"""CelebA loading (flwrlabs/celeba, cached offline via HuggingFace `datasets`),
normalized to [-1, 1] to match the tanh decoder output -- mirrors gga/data.py's
cifar_loaders() interface so the rest of the pipeline (gaussianize, decoder)
is dataset-agnostic.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_HF_REPO = "flwrlabs/celeba"
_HF_CONFIG = "img_align+identity+attr"


def _transform(image_size: int):
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # -> [-1, 1]
    ])


class _CelebASplit(Dataset):
    def __init__(self, hf_split, image_size: int):
        self.ds = hf_split
        self.tf = _transform(image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]["image"].convert("RGB")
        return self.tf(image), 0


def celeba_loaders(cache_dir: str, batch: int, n_particles: int, workers: int = 4,
                   n_train: int | None = None, image_size: int = 64):
    os.environ.setdefault("HF_HOME", cache_dir)
    from datasets import load_dataset
    hf = load_dataset(_HF_REPO, _HF_CONFIG)

    train = _CelebASplit(hf["train"], image_size)
    test = _CelebASplit(hf["test"], image_size)

    if n_train is not None and n_train < len(train):
        g = torch.Generator().manual_seed(1)
        train = Subset(train, torch.randperm(len(train), generator=g)[:n_train].tolist())

    g = torch.Generator().manual_seed(0)
    p_idx = torch.randperm(len(train), generator=g)[:n_particles].tolist()
    particles = Subset(train, p_idx)

    ae_loader = DataLoader(train, batch, shuffle=True, num_workers=workers,
                           pin_memory=True, persistent_workers=workers > 0)
    enc_loader = DataLoader(particles, batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test, batch, shuffle=False, num_workers=workers)
    return ae_loader, enc_loader, test_loader, len(train)
