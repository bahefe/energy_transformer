import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from accelerate import Accelerator
from image_et.core import ET, Patch

# Import collate functions from your utils file.
from image_et.utils import collate_fn_augment, collate_fn_no_augment

def soft_cross_entropy_loss(outputs, soft_targets, label_smoothing=0.0):
    """
    Computes the cross entropy loss between soft targets and model outputs,
    with optional label smoothing. This is useful when using MixUp/CutMix,
    where soft_targets have shape [batch_size, num_classes].
    """
    if label_smoothing > 0:
        num_classes = outputs.size(1)
        # Adjust the target distribution with smoothing
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / num_classes
    log_preds = torch.log_softmax(outputs, dim=1)
    loss = - torch.sum(soft_targets * log_preds, dim=1)
    return loss.mean()

def main(args):
    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Create the Patch module and model
    patch_fn = Patch(dim=args.patch_size)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    model = ET(
        dummy_input,
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
    )

    # Load CIFAR10 dataset
    train_dataset = CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_dataset = CIFAR10(
        root=args.data_path,
        train=False,
        download=True,
        transform=ToTensor()
    )
    accelerator.print('Data accessed')

    # Create DataLoaders with the appropriate collate functions.
    # The training loader applies batch-level augmentation.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_augment
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_no_augment
    )
    accelerator.print('Loaders ready')

    # Set up the optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay
    )

    # Prepare the model, optimizer, and dataloaders with Accelerator.
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    accelerator.print('Before training loop')

    # Training loop
    for epoch in range(1, args.epochs + 1):
        accelerator.print(f"\nStarting Epoch {epoch}/{args.epochs}")
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            outputs = model(images)

            # If labels are hard (1D tensor), use F.cross_entropy with label smoothing.
            # Otherwise, assume soft labels (e.g., from MixUp/CutMix) and use our custom loss.
            if labels.ndim == 1:
                loss = F.cross_entropy(outputs, labels, label_smoothing=args.label_smoothing)
                hard_labels = labels
            else:
                loss = soft_cross_entropy_loss(outputs, labels, label_smoothing=args.label_smoothing)
                hard_labels = labels.argmax(dim=1)

            accelerator.backward(loss)
            optimizer.step()

            # Compute accuracy using hard labels.
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(hard_labels).sum().item()
            batch_total = hard_labels.size(0)
            batch_acc = 100.0 * batch_correct / batch_total

            total_loss += loss.item()
            correct += batch_correct
            total += batch_total

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                accelerator.print(
                    f"Epoch [{epoch}/{args.epochs}], Iteration [{batch_idx}/{len(train_loader)}]: "
                    f"Batch Accuracy = {batch_acc:.2f}%"
                )

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total

        accelerator.print(f"Epoch {epoch}/{args.epochs} completed")
        accelerator.print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        accelerator.print(f"Val Acc: {val_acc:.2f}%")
        accelerator.print(f"Epoch Time: {time.time() - epoch_start_time:.2f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ET Classifier with Label Smoothing")
    parser.add_argument("--num-classes", type=int, default=10)
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    
    args = parser.parse_args()
    main(args)
