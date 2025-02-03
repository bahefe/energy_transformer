import os
import torch
import argparse
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from time import time

from image_et import (
    ET,  # Modified classification model
    Patch,
    GetCIFAR,
    count_parameters,
    device,
    str2bool,
    get_latest_file,
    make_dir
)

def main(args):
    # Setup directories
    MODEL_FOLDER = os.path.join(args.result_dir, "models")
    
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        make_dir(args.result_dir)
        make_dir(MODEL_FOLDER)

    # Initialize model with classification head
    patch_fn = Patch(dim=args.patch_size)
    model = ET(
        x=torch.randn(1, 3, 32, 32),  # Dummy input for initialization
        patch=patch_fn,
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

    if accelerator.is_main_process:
        print(f"Number of parameters: {count_parameters(model)}", flush=True)

    # Load CIFAR dataset with labels
    trainset, testset, _ = GetCIFAR(args.data_path, args.data_name)

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Loss function for classification
    criterion = torch.nn.CrossEntropyLoss()
    
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay,
    )

    # Training setup
    start_epoch = 1
    if accelerator.is_main_process:
        latest_checkpoint = get_latest_file(MODEL_FOLDER, ".pth")
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}", flush=True)
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["opt"])

    model, opt, train_loader, test_loader = accelerator.prepare(
        model, opt, train_loader, test_loader
    )

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time()


        if args.model_type == "swapped" and epoch % 5 == 0 and epoch != 0:
            if accelerator.is_main_process:
                seed = torch.randint(0, 1000000, (1,)).item()
                print(f"\n--- Shuffling layers at epoch {epoch} with seed {seed} ---")
            else:
                seed = 0
            
            seed_tensor = torch.tensor(seed, device=device)
            accelerator.broadcast(seed_tensor)
            seed = seed_tensor.item()
            
            torch.manual_seed(seed)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.shuffle_layers()
            
            accelerator.wait_for_everyone()
            model = accelerator.prepare(model)

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            
            opt.zero_grad()
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)
            
            # Backward pass
            accelerator.backward(loss)
            opt.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, labels in test_loader:
                x, labels = x.to(device), labels.to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        # Sync metrics across processes
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total

        # Print metrics
        if accelerator.is_main_process:
            epoch_time = time() - start_time
            print(f"Epoch: {epoch}/{args.epochs} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Acc: {val_acc:.2f}%")

        # Save checkpoint
        if epoch % args.ckpt_every == 0 and accelerator.is_main_process:
            ckpt = {
                "epoch": epoch,
                "model": accelerator.unwrap_model(model).state_dict(),
                "opt": opt.state_dict(),
                "args": args,
            }
            torch.save(ckpt, os.path.join(MODEL_FOLDER, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ET for Classification")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--tkn-dim", type=int, default=128)
    parser.add_argument("--qk-dim", type=int, default=64)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--hn-mult", type=float, default=4.0)
    parser.add_argument("--attn-beta", type=float, default=0.125)
    parser.add_argument("--attn-bias", type=str2bool, default=False)
    parser.add_argument("--hn-bias", type=str2bool, default=False)
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--result-dir", type=str, default="./classification_results")
    parser.add_argument("--data-path", type=str, default="../data")
    parser.add_argument("--data-name", type=str, default="cifar10")
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    # In the __main__ section's argparse setup
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="standard",
        choices=["standard", "swapped"],
        help="Model architecture type: 'standard' (no swap) or 'swapped' (swap every 5 epochs)"
    )
    args = parser.parse_args()
    
    main(args)