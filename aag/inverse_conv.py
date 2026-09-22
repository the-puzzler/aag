"""From-scratch convolutional inverse E(x) -> z for AAG (user, 2026-10-02).

The DINO-feature inverse capped at half the code variance on CelebA and a quarter of the latent variance on
ImageNet. Frozen semantic features are the wrong basis for this map: after the assignment's transport the
latent is no longer the encoder latent of the image, it is a shuffled copy of it, so the map image -> z has
to be learned outright rather than read off a generic representation.

This is the mirror of Generator256: five stride-2 stages 256 -> 8 with residual blocks, then a small head to
the target. Unconditional even for class-conditional generators, since z is independent of the class by
construction and the image already carries it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two 3x3 convs with GroupNorm, identity skip (1x1 if channels change)."""

    def __init__(self, cin: int, cout: int, groups: int = 32):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class DownStage(nn.Module):
    def __init__(self, cin: int, cout: int, n_res: int):
        super().__init__()
        self.down = nn.Conv2d(cin, cout, 3, 2, 1)
        self.res = nn.ModuleList([ResBlock(cout, cout) for _ in range(n_res)])

    def forward(self, x):
        x = self.down(x)
        for r in self.res:
            x = r(x)
        return x


class ConvInverse(nn.Module):
    """(B, 3, 256, 256) in [-1, 1] -> (B, out_dim) standardised target.

    head='grid': the last feature map is 8x8 and the target is read off it with a 1x1 conv then flattened,
    which matches a generator whose z is a spatial grid (coordinate j of z means grid cell j).
    head='pool': global average pool then MLP, for flat targets such as a bottleneck code.
    """

    def __init__(self, out_dim: int, width: float = 1.0, n_res: int = 2, base_channels=(64, 128, 256, 512, 512),
                 head: str = "pool", grid: int = 8, dropout: float = 0.0):
        super().__init__()
        ch = [max(32, int(round(c * width / 32)) * 32) for c in base_channels]
        self.stem = nn.Conv2d(3, ch[0], 3, 1, 1)
        self.stages = nn.ModuleList()
        cin = ch[0]
        for c in ch:                                   # 256 -> 128 -> 64 -> 32 -> 16 -> 8
            self.stages.append(DownStage(cin, c, n_res)); cin = c
        self.out_norm = nn.GroupNorm(min(32, cin), cin)
        self.head_kind, self.grid, self.out_dim = head, grid, out_dim
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if head == "grid":
            assert out_dim % (grid * grid) == 0, f"grid head needs out_dim divisible by {grid * grid}"
            self.head = nn.Conv2d(cin, out_dim // (grid * grid), 1)
        else:
            self.head = nn.Sequential(nn.Linear(cin, 4 * cin), nn.SiLU(), nn.Linear(4 * cin, out_dim))

    def forward(self, x):
        h = self.stem(x)
        for s in self.stages:
            h = s(h)
        h = F.silu(self.out_norm(h))
        if self.head_kind == "grid":
            if h.shape[-1] != self.grid:
                h = F.adaptive_avg_pool2d(h, self.grid)
            return self.head(self.drop(h)).flatten(1)
        return self.head(self.drop(h.mean(dim=(2, 3))))


def conv_kwargs_from_ckpt(ck: dict) -> dict:
    k = dict(ck.get("conv_kwargs", {}))
    k.setdefault("out_dim", ck["mu"].numel())
    return k
