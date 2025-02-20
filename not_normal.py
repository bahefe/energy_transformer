import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from image_et.core import ET, Patch
from image_et.utils import get_cifar10_datasets, collate_fn_no_augment

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the model (dummy input to set dimensions).
    patch = Patch(dim=4)
    dummy_input = torch.randn(1, 3, 32, 32)
    model = ET(
        dummy_input,
        patch,
        num_classes=10,
        tkn_dim=128,
        qk_dim=64,
        nheads=8,
        hn_mult=4.0,
        attn_beta=0.125,
        attn_bias=False,
        hn_bias=False,
        time_steps=1,
        blocks=12,
        swap_interval=None,
        swap_strategy=None
    ).to(device)
    
    # Load the checkpoint.
    checkpoint_path = "model_20250218_171155_bl12_ts1_bs128_si10_ss1.pth"  # Replace with your actual checkpoint path.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    
    # Load the state dict and remove the '_orig_mod.' prefix.
    original_state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
        else:
            new_key = key
        new_state_dict[new_key] = value

    # Load the transformed state dict into the model.
    model.load_state_dict(new_state_dict)
    print(f"Loaded model from {checkpoint_path}")
    
    # Prepare the datasets and dataloaders.
    trainset, valset, testset, _ = get_cifar10_datasets(root="./data", val_ratio=0.1)
    
    # Training loader uses the training set.
    train_loader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Validation loader (no augmentation).
    val_loader = DataLoader(
        valset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_no_augment
    )
    
    # Test loader (no augmentation).
    test_loader = DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_no_augment
    )
    
    # Define the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Continue training for 10 epochs.
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        # Use tqdm progress bar for the training loop.
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            # Compute loss for backward pass.
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        train_acc = 100 * correct_train / total_train
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Accuracy: {train_acc:.2f}% - Validation Accuracy: {val_acc:.2f}%")
    
    # Evaluate the model on the test set after training.
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy after training: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
