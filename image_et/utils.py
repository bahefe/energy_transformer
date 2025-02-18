import torch
import numpy as np
from functools import partial
from torch.utils.data import random_split, DataLoader, default_collate
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import v2

# ------------------------------
# CIFAR mean/std (for both 10 and 100)
# ------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ------------------------------
# Transforms: match the first file
# ------------------------------
# Augmented pipeline
transforms_augmentation = v2.Compose([
    v2.Resize((32, 32)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # Note: We will handle the final Normalize step inside GetCIFAR,
    #       so that we can switch between CIFAR-10 and CIFAR-100 properly.
])

# No-augmentation pipeline
transforms_no_augment = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # Same note about Normalize as above.
])

# Batchwise augmentation (CutMix / MixUp)
# (assuming CIFAR-10 with 10 classes; modify if you need CIFAR-100)
cutmix_or_mixup = v2.RandomChoice([
    v2.CutMix(num_classes=10),
    v2.MixUp(num_classes=10)
])

def collate_fn_augment(batch):
    """Apply CutMix or MixUp to a batch."""
    return cutmix_or_mixup(*default_collate(batch))

def collate_fn_no_augment(batch):
    """No batchwise augmentation."""
    return default_collate(batch)

# ------------------------------
# Main data-loading function
# ------------------------------
def GetCIFAR(root, which: str = "cifar10", val_ratio: float = 0.1):
    """
    Loads CIFAR-10 or CIFAR-100 with v2 transforms that match the
    first file. Normalization is performed last, based on the correct
    dataset statistics.
    """
    which = which.lower()
    if which == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        dataset_class = CIFAR10
        num_classes = 10
    elif which == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        dataset_class = CIFAR100
        num_classes = 100
    else:
        raise NotImplementedError("Dataset not available.")

    # Add dataset-specific normalization at the end of each pipeline
    transform_train = v2.Compose([
        transforms_augmentation,
        v2.Normalize(mean=mean, std=std)
    ])
    transform_test = v2.Compose([
        transforms_no_augment,
        v2.Normalize(mean=mean, std=std)
    ])

    # Load datasets
    trainset_full = dataset_class(
        root=root,
        train=True,
        download=True,
        transform=transform_train
    )

    testset = dataset_class(
        root=root,
        train=False,
        download=True,
        transform=transform_test
    )

    # Create validation split
    total_train = len(trainset_full)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])

    # Wrap valset in the same test transform
    # (random_split only changes indices, not transforms).
    # We can do this if we want the same pipeline for valset as for testset,
    # or just keep the parted transform. If you want them exactly the same:
    valset.dataset.transform = transform_test

    # Prepare unnormalize function
    std_arr, mean_arr = map(lambda z: np.array(z)[None, :, None, None], (std, mean))
    return trainset, valset, testset, partial(unnormalize, mean=mean_arr, std=std_arr)

# ------------------------------
# Utility functions
# ------------------------------
def unnormalize(x, mean, std):
    return x * std + mean

def gen_mask_id(num_patch, mask_size, batch_size: int):
    batch_id = torch.arange(batch_size)[:, None]
    mask_id = torch.randn(batch_size, num_patch).argsort(-1)[:, :mask_size]
    return batch_id, mask_id

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    return False
