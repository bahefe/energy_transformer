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
from image_et.core import ET, Patch
from image_et.utils import GetCIFAR, collate_fn_augment, collate_fn_no_augment

def soft_cross_entropy_loss(outputs, soft_targets, label_smoothing=0.0):
    """Soft cross entropy loss with label smoothing support"""
    if label_smoothing > 0:
        num_classes = outputs.size(1)
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / num_classes
    return -torch.sum(soft_targets * torch.log_softmax(outputs, dim=1), dim=1).mean()

def main(args):
    # Initialize tracking dictionary
    results = {
        'args': vars(args),
        'epoch_stats': [],
        'test_acc': None,
        'timestamp': datetime.now().isoformat()
    }
    
    if torch.cuda.is_available():
        device_type = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU only
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    accelerator = Accelerator(
        mixed_precision='no',
        dynamo_backend="no",
        gradient_accumulation_steps=2,
        cpu=(device_type == "cpu")
    )

    # Manual device handling for MPS
    if device_type == "mps":
        device = torch.device("mps")
        accelerator._device = device
    else:
        device = accelerator.device

    # Create model with swap parameters included.
    patch_fn = Patch(dim=args.patch_size)
    model = ET(
        torch.randn(1, 3, 32, 32),
        patch_fn,
        num_classes=args.num_classes,
        tkn_dim=args.tkn_dim,
        qk_dim=args.qk_dim,
        nheads=args.nheads,
        hn_mult=args.hn_mult,
        attn_beta=args.attn_beta,
        attn_bias=args.attn_bias,
        hn_bias=args.hn_bias,
        time_steps=args.time_steps,
        blocks=args.blocks,
        swap_interval=args.swap_interval,  # New parameter
        swap_strategy=args.swap_strategy   # New parameter
    ).to(device)

    # Load datasets
    trainset, valset, testset, unnormalize_fn = GetCIFAR(
        root=args.data_path,
        which="cifar10",
        val_ratio=0.1
    )

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn_augment)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn_no_augment)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate_fn_no_augment)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay
    )

    # Prepare components with Accelerator
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # Training loop
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_stats = {'epoch': epoch+1}
        
        # Training phase
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        # Create progress bar
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
            
            if labels.ndim == 1:
                loss = F.cross_entropy(outputs, labels, label_smoothing=args.label_smoothing)
                hard_labels = labels
            else:
                loss = soft_cross_entropy_loss(outputs, labels, args.label_smoothing)
                hard_labels = labels.argmax(dim=1)
            
            accelerator.backward(loss)
            optimizer.step()

            # Update metrics
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

        # Close progress bar
        if accelerator.is_local_main_process:
            pbar.close()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # Record epoch statistics
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

    # Final test evaluation
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

    # Generate filename-friendly hyperparameter string
    hyperparams = vars(args)
    excluded_params = ['data_path', 'num_workers', 'num_classes']
    abbreviations = {
        'patch_size': 'ps',
        'tkn_dim': 'td',
        'qk_dim': 'qd',
        'nheads': 'nh',
        'hn_mult': 'hm',
        'attn_beta': 'ab',
        'attn_bias': 'ab',
        'hn_bias': 'hb',
        'time_steps': 'ts',
        'blocks': 'bl',
        'swap_interval': 'si',  # New abbreviation for swap_interval
        'swap_strategy': 'ss',  # New abbreviation for swap_strategy
        'epochs': 'ep',
        'batch_size': 'bs',
        'lr': 'lr',
        'b1': 'b1',
        'b2': 'b2',
        'weight_decay': 'wd',
        'label_smoothing': 'ls'
    }

    param_parts = []
    for k, v in hyperparams.items():
        if k in excluded_params:
            continue
        abbrev = abbreviations.get(k, k)
        # Handle different value types
        if isinstance(v, bool):
            param_parts.append(f"{abbrev}{1 if v else 0}")
        elif isinstance(v, float):
            if v.is_integer():
                param_parts.append(f"{abbrev}{int(v)}")
            else:
                # Format to 4 decimal places without scientific notation
                param_parts.append(f"{abbrev}{v:.4f}".rstrip('0').rstrip('.'))
        elif v is None:
            param_parts.append(f"{abbrev}None")
        else:
            param_parts.append(f"{abbrev}{v}")

    # Create filename components
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyper_str = "_".join(param_parts)
    max_length = 100  # Keep filename reasonable
    safe_hyper_str = (hyper_str[:max_length] + '..') if len(hyper_str) > max_length else hyper_str
    
    # Save paths
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f"results_{timestamp}_{safe_hyper_str}.json")
    model_file = os.path.join(output_dir, f"model_{timestamp}_{safe_hyper_str}.pth")
    
    # Save results and model
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
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=4)
    # New swap parameters
    parser.add_argument("--swap-interval", type=int, default=None,
                        help="Interval (in iterations) at which to swap blocks in the model.")
    parser.add_argument("--swap-strategy", type=int, default=1,
                        help="Swap strategy to use (1, 2, 3, or 4).")
    
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
    args.num_classes = 10  # Fixed for CIFAR-10
    main(args)
