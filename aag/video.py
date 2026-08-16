"""Video helpers shared by the generator scripts.

Three things differ from images and each has bitten once already:
  * LPIPS is 2D -> score every frame, average.
  * the segment cache must stay uint8 on GPU; 88k segments as fp32 is ~70GB.
  * save_image cannot take a 5D tensor -> lay frames out as a strip.
"""
from __future__ import annotations

import torch
from torchvision.utils import save_image


def lpips_any(perceptual, a, b):
    """LPIPS for images or video. (B,C,T,H,W) is scored frame-by-frame."""
    if a.dim() == 5:
        B, C, T, H, W = a.shape
        a = a.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        b = b.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    return perceptual(a, b)


class Uint8Targets:
    """Keeps targets as uint8 on device, yielding float [-1,1] per batch."""

    def __init__(self, u8: torch.Tensor):
        self.u8 = u8                                  # (N,3,T,H,W) or (N,3,H,W)

    def __len__(self):
        return self.u8.shape[0]

    @property
    def shape(self):
        return self.u8.shape

    def __getitem__(self, idx):
        return self.u8[idx].float().div_(127.5).sub_(1.0)


def load_video_targets(cache_root: str, n: int, device="cuda"):
    """Segment cache -> Uint8Targets in particle order (the order z indexes)."""
    import numpy as np
    from aag.datasets import _segment_loaders  # reuses the same seeded permutation
    segs = np.load(f"{cache_root}/segments.npy", mmap_mode="r")
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(segs), generator=g)[:n].numpy()
    out = torch.empty(len(perm), 3, *segs.shape[1:3], segs.shape[3], dtype=torch.uint8)
    # (T,H,W,3) -> (3,T,H,W), copied in chunks to bound host memory
    for i0 in range(0, len(perm), 512):
        chunk = perm[i0:i0 + 512]
        arr = torch.from_numpy(segs[np.sort(chunk)].copy())
        order = torch.argsort(torch.argsort(torch.as_tensor(chunk)))
        out[i0:i0 + len(chunk)] = arr[order].permute(0, 4, 1, 2, 3)
    return Uint8Targets(out.to(device))


def save_video_grid(video, path, n_show=8, n_frames=8):
    """(B,3,T,H,W) -> a PNG strip: one row per sample, n_frames columns."""
    v = video[:n_show].clamp(-1, 1)
    T = v.shape[2]
    idx = torch.linspace(0, T - 1, min(n_frames, T)).long()
    v = v[:, :, idx]                                   # (B,3,f,H,W)
    B, C, f, H, W = v.shape
    strip = v.permute(0, 2, 1, 3, 4).reshape(B * f, C, H, W)
    save_image((strip + 1) / 2, path, nrow=f)
