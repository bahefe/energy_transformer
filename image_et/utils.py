import torch
import numpy as np
from functools import partial
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split, DataLoader, default_collate
from torchvision.transforms import v2

# Predefined normalization constants
CIFAR10_STD = (0.2470, 0.2435, 0.2616)  # Adjusted for correct unnormalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)

CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)

# --------------------------------------------------
# Updated v2 Transforms with normalization placeholders
transforms_augmentation = v2.Compose([
    v2.Resize((32, 32)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
    v2.RandAugment(num_ops=2),
    v2.RandomErasing(0.15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

transforms_no_augment = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Batchwise augmentation (CutMix / MixUp)
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
# --------------------------------------------------

def GetCIFAR(root, which: str = "cifar10", val_ratio: float = 0.1):
    """
    Loads CIFAR dataset with v2 transforms and proper normalization
    """
    which = which.lower()
    if which == "cifar10":
        std, mean = CIFAR10_STD, CIFAR10_MEAN
        dataset_class = CIFAR10
    elif which == "cifar100":
        std, mean = CIFAR100_STD, CIFAR100_MEAN
        dataset_class = CIFAR100
    else:
        raise NotImplementedError("Dataset not available.")

    # Training transforms with augmentation
    transform_train = v2.Compose([
        transforms_augmentation,
        v2.Normalize(mean=mean, std=std),
    ])

    # Validation/Test transforms without augmentation
    transform_test = v2.Compose([
        transforms_no_augment,
        v2.Normalize(mean=mean, std=std),
    ])

    # Load datasets
    trainset_full = dataset_class(
        root,
        train=True,
        download=True,
        transform=transform_train
    )

    testset = dataset_class(
        root,
        train=False,
        download=True,
        transform=transform_test
    )

    # Create validation split
    total_train = len(trainset_full)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])

    # Prepare unnormalize function with correct parameter order
    std_arr, mean_arr = map(lambda z: np.array(z)[None, :, None, None], (std, mean))
    return trainset, valset, testset, partial(unnormalize, mean=mean_arr, std=std_arr)


def gen_mask_id(num_patch, mask_size, batch_size: int):
    batch_id = torch.arange(batch_size)[:, None]
    mask_id = torch.randn(batch_size, num_patch).argsort(-1)[:, :mask_size]
    return batch_id, mask_id


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unnormalize(x, std, mean):
    x = x * std + mean
    return x


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


