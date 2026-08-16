"""CIFAR-10 loading, normalized to [-1, 1] to match the tanh decoder output.

The AE trains on the full train set for a good representation; the persistent
assignment then uses a *fixed* N-image subset (default N=6000, the report's
coverage regime), so N is held constant across the bottleneck-dimension sweep.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # -> [-1, 1]
])


def cifar_loaders(root: str, batch: int, n_particles: int, workers: int = 2,
                  n_train: int | None = None):
    train = datasets.CIFAR10(root, train=True, download=True, transform=_TF)
    test = datasets.CIFAR10(root, train=False, download=True, transform=_TF)

    if n_train is not None and n_train < len(train):
        g = torch.Generator().manual_seed(1)
        train = Subset(train, torch.randperm(len(train), generator=g)[:n_train].tolist())

    # fixed particle subset (deterministic) -> stable x_i <-> z_i identity
    g = torch.Generator().manual_seed(0)
    p_idx = torch.randperm(len(train), generator=g)[:n_particles].tolist()
    particles = Subset(train, p_idx)

    ae_loader = DataLoader(train, batch, shuffle=True, num_workers=workers,
                           pin_memory=True)
    enc_loader = DataLoader(particles, batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test, batch, shuffle=False, num_workers=workers)
    return ae_loader, enc_loader, test_loader, len(train)
