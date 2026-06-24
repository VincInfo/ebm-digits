"""MNIST loaders: optional downscale (e.g. 8x8) and [-1, 1] scaling for EBMs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    root: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    img_size: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/test DataLoaders with images [1, img_size, img_size] in [-1, 1]."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )

    train_ds = datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, test_loader
