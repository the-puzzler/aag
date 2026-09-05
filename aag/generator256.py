"""Direct-to-pixel generator for 256x256: a spatial z grid -> image, optionally
modulated by a class embedding.

z is a transported copy of a SPATIAL AE latent (DC-AE f32c32 gives 32x8x8), and
the assignment whitens with rotate=False precisely so coordinate j of z still
means grid cell j of h. So the generator does not start from a Linear over a
flat 2048-vector: it reshapes z back to (C, g, g) and decodes with the same
parameter-free DC-AE upsample shortcuts the AE path uses (aag.ae.DCAEUpBlock),
which is what made 192x-compression decoders trainable there.

Conditioning is adaptive GroupNorm (ADM-style): a class embedding passes through
an MLP and produces a per-block (scale, shift) on the normalised activations.
With cond=None every AdaGN falls back to plain GroupNorm, so the unconditional
CelebA-HQ generator and the class-conditional ImageNet one are one module.

The class enters ONLY here. There is no classifier-free guidance in AAG, so class
adherence rests entirely on z being independent of the class -- which is what
the interleaved hierarchy transport in run_assignment_classes.py is for.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaGroupNorm(nn.Module):
    def __init__(self, ch: int, cond_dim: int | None, groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch), ch, affine=cond_dim is None)
        self.proj = nn.Linear(cond_dim, 2 * ch) if cond_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        x = self.norm(x)
        if self.proj is None or cond is None:
            return x
        scale, shift = self.proj(cond)[:, :, None, None].chunk(2, 1)
        return x * (1 + scale) + shift


class CondResBlock(nn.Module):
    """Two 3x3 convs with AdaGN, identity skip (1x1 if channels change)."""

    def __init__(self, cin: int, cout: int, cond_dim: int | None):
        super().__init__()
        self.n1 = AdaGroupNorm(cin, cond_dim)
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.n2 = AdaGroupNorm(cout, cond_dim)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)
        # zero-init the last conv so every block starts as identity; deep decoders
        # otherwise blow activations up through the parameter-free shortcuts
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x, cond)))
        h = self.c2(F.silu(self.n2(h, cond)))
        return self.skip(x) + h


class UpShortcut(nn.Module):
    """DC-AE channel-to-space upsample: parameter-free, tile-or-average channels
    to 4*cout then pixel_shuffle(2). Same as aag.ae.DCAEUpBlock.skip."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.need = cout * 4

    def forward(self, x):
        c = x.shape[1]
        if c < self.need:
            x = x.repeat(1, -(-self.need // c), 1, 1)[:, :self.need]
        elif c > self.need:
            x = x.view(x.shape[0], self.need, c // self.need, *x.shape[2:]).mean(2)
        return F.pixel_shuffle(x, 2)


class UpStage(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim, n_res: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1)
        self.short = UpShortcut(cin, cout)
        self.res = nn.ModuleList([CondResBlock(cout, cout, cond_dim) for _ in range(n_res)])

    def forward(self, x, cond):
        h = self.conv(self.up(x)) + self.short(x)
        for r in self.res:
            h = r(h, cond)
        return h


class Generator256(nn.Module):
    """z (B, C*g*g) [+ class id] -> (B, 3, image_size, image_size) in [-1, 1].

    channel schedule runs from the grid resolution up to image_size; the default
    for g=8 -> 256 is five doublings at [512, 512, 256, 128, 64] with 2 res
    blocks each, ~60M params. `width` scales every stage.
    """

    def __init__(self, dim_z: int, grid: int = 8, image_size: int = 256,
                 n_classes: int = 0, cond_dim: int = 512, width: float = 1.0,
                 n_res: int = 2, base_channels=(512, 512, 256, 128, 64)):
        super().__init__()
        # grid=0: z is a FLAT latent with no spatial layout (a 1-D tokenizer such as
        # TiTok, 32 tokens x 16). Reshaping it to a grid would be arbitrary, so a
        # learned linear stem lays it out on an 8x8 grid instead; everything after
        # the stem is unchanged.
        self.flat = grid == 0
        if self.flat:
            grid = 8
            self.grid, self.cz = grid, max(16, dim_z // (grid * grid))
        else:
            assert dim_z % (grid * grid) == 0, "dim_z must be C * grid^2"
            self.grid, self.cz = grid, dim_z // (grid * grid)
        n_up = (image_size // grid).bit_length() - 1
        assert grid << n_up == image_size, "image_size must be grid * 2^k"
        chans = [max(16, int(c * width)) for c in base_channels[:n_up]]
        if len(chans) < n_up:
            chans += [chans[-1]] * (n_up - len(chans))
        self.n_classes = n_classes
        cd = cond_dim if n_classes else None
        if n_classes:
            self.emb = nn.Embedding(n_classes, cond_dim)
            self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cond_dim),
                                          nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.lay = nn.Linear(dim_z, self.cz * grid * grid) if self.flat else None
        self.stem = nn.Conv2d(self.cz, chans[0], 3, 1, 1)
        self.pre = nn.ModuleList([CondResBlock(chans[0], chans[0], cd) for _ in range(n_res)])
        stages, cin = [], chans[0]
        for c in chans:
            stages.append(UpStage(cin, c, cd, n_res)); cin = c
        self.stages = nn.ModuleList(stages)
        self.out_norm = AdaGroupNorm(cin, cd)
        self.out = nn.Conv2d(cin, 3, 3, 1, 1)

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.cond_mlp(self.emb(y)) if (self.n_classes and y is not None) else None
        if self.lay is not None:
            z = self.lay(z)
        h = self.stem(z.view(z.shape[0], self.cz, self.grid, self.grid))
        for r in self.pre:
            h = r(h, cond)
        for s in self.stages:
            h = s(h, cond)
        return torch.tanh(self.out(F.silu(self.out_norm(h, cond))))
