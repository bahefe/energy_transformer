import torch
import numpy as np
from functools import partial
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split, DataLoader, default_collate

# Predefined normalization constants
CIFAR10_STD = (0.4914, 0.4822, 0.4465)
CIFAR10_MU = (0.2023, 0.1994, 0.2010)

CIFAR100_STD = (0.5071, 0.4867, 0.4408)
CIFAR100_MU = (0.2675, 0.2565, 0.2761)

# --------------------------------------------------
# Data Augmentation transforms from your other project.
# Here we use torchvision.transforms.v2 which provides
# newer APIs for augmentation.
import torch
from torchvision.transforms import v2

transforms_augmentation = v2.Compose(
    [
        v2.Resize((32, 32)),
        v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
        v2.RandAugment(num_ops=2),
        v2.RandomErasing(0.15),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

transforms_no_augment = v2.Compose(
    [
        v2.Resize((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# Batchwise augmentation (CutMix / MixUp)
cutmix_or_mixup = v2.RandomChoice([v2.CutMix(num_classes=10), v2.MixUp(num_classes=10)])


def collate_fn_augment(batch):
    """Apply CutMix or MixUp to a batch."""
    return cutmix_or_mixup(*default_collate(batch))


def collate_fn_no_augment(batch):
    """No batchwise augmentation."""
    return default_collate(batch)
# --------------------------------------------------

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


def GetCIFAR(root, which: str = "cifar10", val_ratio: float = 0.1, use_v2_aug: bool = False):
    """
    Loads the CIFAR dataset with an option to use the v2 augmentation transforms.
    
    Args:
        root (str): The path where the data will be stored.
        which (str): Either "cifar10" or "cifar100".
        val_ratio (float): The ratio of the training set to be used as validation.
        use_v2_aug (bool): If True, uses the v2 augmentation transforms.
                           Otherwise, uses the standard transforms.
    
    Returns:
        trainset, valset, testset, unnormalize: The train, validation, test datasets and 
                                                a function to unnormalize images.
    """
    which = which.lower()
    if which == "cifar10":
        std, mean = CIFAR10_STD, CIFAR10_MU

        # Choose transform based on use_v2_aug flag.
        if use_v2_aug:
            transform_train = transforms_augmentation
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                ]
            )

        trainset_full = CIFAR10(
            root,
            train=True,
            download=True,
            transform=transform_train,
        )

        testset = CIFAR10(
            root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                ]
            ),
        )

    elif which == "cifar100":
        std, mean = CIFAR100_STD, CIFAR100_MU

        if use_v2_aug:
            transform_train = transforms_augmentation
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                ]
            )

        trainset_full = CIFAR100(
            root,
            train=True,
            download=True,
            transform=transform_train,
        )

        testset = CIFAR100(
            root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                ]
            ),
        )
    else:
        raise NotImplementedError("Dataset not available.")

    # Create a validation split (val_ratio of the training data)
    total_train = len(trainset_full)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])

    # Prepare unnormalize function with std and mean reshaped for images
    std_arr, mean_arr = map(lambda z: np.array(z)[None, :, None, None], (std, mean))

    return trainset, valset, testset, partial(unnormalize, std=std_arr, mean=mean_arr)


