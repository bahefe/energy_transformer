import os
from tqdm.auto import tqdm
import time
import json
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Import your updated ET class (with three blocks: [1, 10, 1]) and Patch
from image_et.core import ET, Patch
from image_et.utils import get_cifar10_datasets, collate_fn_augment, collate_fn_no_augment

def soft_cross_entropy_loss(outputs, soft_targets, label_smoothing=0.0):
    """Soft cross entropy loss with label smoothing support"""
    if label_smoothing > 0:
        num_classes = outputs.size(1)
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / num_classes
    return -torch.sum(soft_targets * torch.log_softmax(outputs, dim=1), dim=1).mean()


def main(args):
    results = {
        'args': vars(args),
        'epoch_stats': [],
        'test_acc': None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Device selection
    if torch.cuda.is_available():
        device_type = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    accelerator = Accelerator(
        mixed_precision='fp16',
        dynamo_backend="inductor",
        gradient_accumulation_steps=2,
        cpu=(device_type == "cpu")
    )

    # Manually handle MPS device if used
    if device_type == "mps":
        device = torch.device("mps")
        accelerator._device = device
    else:
        device = accelerator.device

    # Build model
    patch_fn = Patch(dim=args.patch_size)
    model = ET(
        x=torch.randn(1, 3, 32, 32),      # Dummy input for shape inference
        patch=patch_fn,
        num_classes=args.num_classes,
        tkn_dim=args.tkn_dim,
        qk_dim=args.qk_dim,
        nheads=args.nheads,
        hn_mult=args.hn_mult,
        attn_beta=args.attn_beta,
        attn_bias=args.attn_bias,
        hn_bias=args.hn_bias,
        # time_steps and blocks are removed here because
        # the ET class now explicitly uses three blocks with [1, 10, 1] steps
    ).to(device)

    # Load CIFAR-10 datasets
    trainset, valset, testset, unnormalize_fn = get_cifar10_datasets(
        root=args.data_path,
        val_ratio=0.1
    )

    # Create DataLoaders
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn_augment
    )
    val_loader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn_no_augment
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn_no_augment
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay
    )

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_stats = {'epoch': epoch + 1}
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        # Progress bar (only main process)
        if accelerator.is_local_main_process:
            pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}"
            )

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(images)
            
            # Soft CE vs. standard CE
            if labels.ndim == 1:
                loss = F.cross_entropy(outputs, labels, label_smoothing=args.label_smoothing)
                hard_labels = labels
            else:
                loss = soft_cross_entropy_loss(outputs, labels, args.label_smoothing)
                hard_labels = labels.argmax(dim=1)
            
            accelerator.backward(loss)
            optimizer.step()

            # Metrics
            preds = outputs.argmax(dim=1)
            batch_correct = (preds == hard_labels).sum().item()
            batch_total = hard_labels.size(0)

            train_correct += batch_correct
            train_total += batch_total
            total_loss += loss.item()

            # Update progress bar
            if accelerator.is_local_main_process:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * batch_correct / batch_total:.2f}%"
                })

        if accelerator.is_local_main_process:
            pbar.close()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_stats.update({
            'train_loss': total_loss / len(train_loader),
            'train_acc': 100 * train_correct / train_total,
            'val_acc': 100 * val_correct / val_total,
            'epoch_time': time.time() - epoch_start
        })
        results['epoch_stats'].append(epoch_stats)

        accelerator.print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        accelerator.print(f"Train Loss: {epoch_stats['train_loss']:.4f} | "
                          f"Train Acc: {epoch_stats['train_acc']:.2f}% | "
                          f"Val Acc: {epoch_stats['val_acc']:.2f}%")
        accelerator.print("="*50)

        # Perform optional block swap if indicated
        if args.swap_interval is not None and ((epoch+1) % args.swap_interval == 0):
            model.swap_blocks(args.swap_strategy)

    # Final test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100 * test_correct / test_total
    results['test_acc'] = test_acc
    results['total_time'] = time.time() - start_time

    accelerator.print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # Save any recorded swap history (if your ET class implements it)
    results["swap_history"] = accelerator.unwrap_model(model).swap_history

    ################################################################
    # Generate a shortened hyperparams string for filenames
    ################################################################
    hyperparams = vars(args)
    # We will track only a few parameters (no longer tracking time_steps/blocks)
    abbreviations = {
        'patch_size': 'ps',
        'tkn_dim': 'td',
        'qk_dim': 'qd',
        'nheads': 'nh',
        'hn_mult': 'hm',
        'attn_beta': 'ab',
        'attn_bias': 'ab',
        'hn_bias': 'hb',
        'swap_interval': 'si',
        'swap_strategy': 'ss',
        'batch_size': 'bs',
        'epochs': 'ep',
        'lr': 'lr',
        'b1': 'b1',
        'b2': 'b2',
        'weight_decay': 'wd',
        'label_smoothing': 'ls'
    }

    # Decide which parameters to include in filename
    selected_params = [
        'patch_size', 'tkn_dim', 'qk_dim', 'nheads', 'hn_mult', 
        'attn_beta', 'attn_bias', 'hn_bias', 'swap_interval', 
        'swap_strategy', 'batch_size', 'epochs', 'lr', 'b1', 'b2', 
        'weight_decay', 'label_smoothing'
    ]

    param_parts = []
    for k in selected_params:
        v = hyperparams.get(k)
        if v is None:
            continue
        abbrev = abbreviations.get(k, k)
        # Formatting
        if isinstance(v, bool):
            param_part = f"{abbrev}{1 if v else 0}"
        elif isinstance(v, float):
            if v.is_integer():
                param_part = f"{abbrev}{int(v)}"
            else:
                param_part = f"{abbrev}{v:.4f}".rstrip('0').rstrip('.')
        else:
            param_part = f"{abbrev}{v}"
        param_parts.append(param_part)

    # Filename
    hyper_str = "_".join(param_parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"results_{timestamp}_{hyper_str}.json")
    model_file = os.path.join(output_dir, f"model_{timestamp}_{hyper_str}.pth")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(accelerator.unwrap_model(model).state_dict(), model_file)
    
    accelerator.print(f"\nResults saved to {results_file}")
    accelerator.print(f"Model saved to {model_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ET Classifier Training")

    # Model parameters
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--tkn-dim", type=int, default=256)
    parser.add_argument("--qk-dim", type=int, default=64)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--hn-mult", type=float, default=4.0)
    parser.add_argument("--attn-beta", type=float, default=0.125)
    parser.add_argument("--attn-bias", action="store_true")
    parser.add_argument("--hn-bias", action="store_true")

    # Removed --time-steps and --blocks because ET now has [1, 10, 1] fixed
    parser.add_argument("--swap-interval", type=int, default=None)
    parser.add_argument("--swap-strategy", type=int, default=1)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    
    # Data parameters
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    args.num_classes = 10  # CIFAR-10
    main(args)
