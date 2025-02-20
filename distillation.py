import os
import time
import json
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from image_et.core import ET, Patch
from image_et.utils import get_cifar10_datasets, collate_fn_augment, collate_fn_no_augment

def load_block_weights_from_random_teacher(student_model):
    """
    Simulate a teacher checkpoint by randomly initializing block weights
    and mapping them from a teacher state (with a '_orig_mod.blocks.4.' prefix)
    to the student's single block (with a 'blocks.0.' prefix).
    """
    # Create a simulated teacher state with random values for block weights.
    teacher_state = {
        "_orig_mod.blocks.4.0.gamma": torch.randn(1),
        "_orig_mod.blocks.4.0.bias": torch.randn(128),
        "_orig_mod.blocks.4.1.hn.proj.weight": torch.randn(512, 128),
        "_orig_mod.blocks.4.1.attn.wq": torch.randn(8, 64, 128),
        "_orig_mod.blocks.4.1.attn.wk": torch.randn(8, 64, 128),
    }
    
    print("Simulated teacher state (randomly initialized) keys:")
    for k, v in teacher_state.items():
        print(f"  {k}: shape {v.shape}")
    
    student_state = student_model.state_dict()
    state_dict = {}
    block_teacher_prefix = "_orig_mod.blocks.4."
    block_student_prefix = "blocks.0."
    block_keys = [k for k in teacher_state if k.startswith(block_teacher_prefix)]
    
    print("\nMapping block keys from simulated teacher state:")
    for key in block_keys:
        new_key = key.replace(block_teacher_prefix, block_student_prefix)
        teacher_tensor = teacher_state[key]
        if new_key in student_state:
            student_tensor = student_state[new_key]
            print(f"Mapping teacher {key} (shape: {teacher_tensor.shape}) to student {new_key} (shape: {student_tensor.shape})")
        else:
            print(f"Warning: Student key {new_key} not found!")
        state_dict[new_key] = teacher_tensor

    missing_keys, unexpected_keys = student_model.load_state_dict(state_dict, strict=False)
    print("\nAfter loading mapped block weights:")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    
    print("\nFinal student model block state and norms:")
    for k, v in student_model.state_dict().items():
        if k.startswith("blocks.0"):
            print(f"  {k}: shape {v.shape}, norm {v.norm():.4f}")
    
    return student_model

def soft_cross_entropy_loss(outputs, soft_targets, label_smoothing=0.0):
    """Soft cross entropy loss with label smoothing support"""
    if label_smoothing > 0:
        num_classes = outputs.size(1)
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / num_classes
    return -torch.sum(soft_targets * torch.log_softmax(outputs, dim=1), dim=1).mean()

def main(args):
    results = {
        "args": vars(args),
        "epoch_stats": [],
        "test_acc": None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Setup device and accelerator.
    if torch.cuda.is_available():
        device_type = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    
    accelerator = Accelerator(
        mixed_precision="fp16",
        dynamo_backend="inductor",
        gradient_accumulation_steps=2,
        cpu=(device_type == "cpu")
    )
    
    if device_type == "mps":
        device = torch.device("mps")
        accelerator._device = device
    else:
        device = accelerator.device

    # Initialize the ET model using a dummy input.
    patch = Patch(dim=args.patch_size)
    dummy_input = torch.randn(1, 3, 32, 32)
    model = ET(
        dummy_input,
        patch,
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
        swap_interval=args.swap_interval,
        swap_strategy=args.swap_strategy
    ).to(device)
    
    # Instead of loading a teacher checkpoint, map a random teacher block to the student.
    model = load_block_weights_from_random_teacher(model)

    # Load CIFAR10 datasets.
    trainset, valset, testset, unnormalize_fn = get_cifar10_datasets(
        root=args.data_path,
        val_ratio=0.1
    )
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn_augment)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn_no_augment)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate_fn_no_augment)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay
    )
    
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_stats = {"epoch": epoch+1}
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()
        
        if accelerator.is_local_main_process:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, bar_format="{l_bar}{bar:20}{r_bar}")
        
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
            
            preds = outputs.argmax(dim=1)
            batch_correct = (preds == hard_labels).sum().item()
            batch_total = hard_labels.size(0)
            train_correct += batch_correct
            train_total += batch_total
            total_loss += loss.item()
            
            if accelerator.is_local_main_process:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * batch_correct / batch_total:.2f}%"
                })
        if accelerator.is_local_main_process:
            pbar.close()
        
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
            "train_loss": total_loss / len(train_loader),
            "train_acc": 100 * train_correct / train_total,
            "val_acc": 100 * val_correct / val_total,
            "epoch_time": time.time() - epoch_start
        })
        results["epoch_stats"].append(epoch_stats)
        
        accelerator.print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        accelerator.print(f"Train Loss: {epoch_stats['train_loss']:.4f} | Train Acc: {epoch_stats['train_acc']:.2f}% | Val Acc: {epoch_stats['val_acc']:.2f}%")
        accelerator.print("=" * 50)
        
        if args.swap_interval is not None and ((epoch+1) % args.swap_interval == 0):
            model.swap_blocks(args.swap_strategy)
    
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
    results["test_acc"] = test_acc
    results["total_time"] = time.time() - start_time
    accelerator.print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    results["swap_history"] = accelerator.unwrap_model(model).swap_history
    
    hyperparams = vars(args)
    abbreviations = {
        "blocks": "bl",
        "time_steps": "ts",
        "batch_size": "bs",
        "swap_interval": "si",
        "swap_strategy": "ss",
    }
    selected_params = ["blocks", "time_steps", "batch_size"]
    if hyperparams.get("swap_interval") is not None:
        selected_params.extend(["swap_interval", "swap_strategy"])
    param_parts = []
    for k in selected_params:
        v = hyperparams.get(k)
        if v is None:
            continue
        abbrev = abbreviations.get(k, k)
        if isinstance(v, bool):
            param_part = f"{abbrev}{1 if v else 0}"
        elif isinstance(v, float):
            if v.is_integer():
                param_part = f"{abbrev}{int(v)}"
            else:
                param_part = f"{abbrev}{v:.4f}".rstrip("0").rstrip(".")
        else:
            param_part = f"{abbrev}{v}"
        param_parts.append(param_part)
    
    hyper_str = "_".join(param_parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"results_{timestamp}_{hyper_str}.json")
    model_file = os.path.join(output_dir, f"model_{timestamp}_{hyper_str}.pth")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(accelerator.unwrap_model(model).state_dict(), model_file)
    
    accelerator.print(f"\nResults saved to {results_file}")
    accelerator.print(f"Model saved to {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ET Distillation with Random Block Mapping")
    # Model parameters
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--tkn-dim", type=int, default=128)
    parser.add_argument("--qk-dim", type=int, default=64)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--hn-mult", type=float, default=4.0)
    parser.add_argument("--attn-beta", type=float, default=0.125)
    parser.add_argument("--attn-bias", action="store_true")
    parser.add_argument("--hn-bias", action="store_true")
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--swap-interval", type=int, default=None)
    parser.add_argument("--swap-strategy", type=int, default=1)
    # Training parameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    # Data parameters
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    args.num_classes = 10
    main(args)
