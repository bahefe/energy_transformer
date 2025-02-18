import torch
import numpy as np
from functools import partial
from torch.utils.data import random_split, DataLoader, default_collate
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

# -----------------------------------------------------------------------------
# 1. Basic Constants and Utility Functions
# -----------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# -----------------------------------------------------------------------------
# 2. Image-Level Transforms
# -----------------------------------------------------------------------------
# (A) Augmentation pipeline for training images
#     Includes: resizing to 32x32, AutoAugment, converting to float,
#     and normalizing with the correct CIFAR-10 statistics.
transforms_augmentation = v2.Compose([
    v2.Resize((32, 32)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
    v2.ToImage(),                           # Convert to PIL-like image
    v2.ToDtype(torch.float32, scale=True),  # Scale [0..255] => [0..1], cast to float
    v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# (B) No augmentation for validation/test images
#     Also does the standard resize, convert to float, and normalization.
transforms_no_augment = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# -----------------------------------------------------------------------------
# 3. Batch-Level Augmentation (CutMix/MixUp)
# -----------------------------------------------------------------------------
# This will be applied *inside the collate function* for the training set.
cutmix_or_mixup = v2.RandomChoice([
    v2.CutMix(num_classes=10),
    v2.MixUp(num_classes=10)
])

def collate_fn_augment(batch):
    """
    Collate function that applies CutMix or MixUp
    to a batch of (images, labels).
    """
    return cutmix_or_mixup(*default_collate(batch))

def collate_fn_no_augment(batch):
    """
    Simple collate function with no batchwise augmentation.
    Used for validation/test sets.
    """
    return default_collate(batch)

# -----------------------------------------------------------------------------
# 4. Dataset Loading Function
# -----------------------------------------------------------------------------
def get_cifar10_datasets(root="./data/cifar-10-python", val_ratio=0.1):
    """
    Loads CIFAR-10 with image-level transformations applied
    and splits into train/val/test. The training set uses
    transforms_augmentation, while validation/test use transforms_no_augment.

    By default, a 'val_ratio' fraction is taken from the training set
    as the validation set.
    """
    # Training set with augmentation
    trainset_full = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transforms_augmentation
    )
    # Test set with no augmentation
    testset = CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transforms_no_augment
    )

    # Split train -> train + validation
    total_train = len(trainset_full)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])

    # Override transform for valset so that it uses no augmentation
    # (We want validation to be as close to test as possible)
    valset.dataset.transform = transforms_no_augment

    # Prepare an unnormalize function if desired (for debugging/visualization)
    std_arr, mean_arr = map(lambda z: np.array(z)[None, :, None, None], (CIFAR10_STD, CIFAR10_MEAN))
    unnorm_func = partial(unnormalize, mean=mean_arr, std=std_arr)

    return trainset, valset, testset, unnorm_func

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
