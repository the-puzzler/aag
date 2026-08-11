"""Convolutional autoencoder for CIFAR-10 with a configurable latent width.

Section 1.1 of the report: "Train an ordinary reconstruction autoencoder and
encode the full training set."  The AE is deliberately plain — its only job is
to supply a useful representation.  The generative machinery lives entirely in
the offline Gaussian assignment + direct decoder that come afterwards.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """32x32x3 -> latent_dim.  Three stride-2 conv blocks then a linear head."""

    def __init__(self, latent_dim: int, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.SiLU(),        # 16x16
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.SiLU(),  # 8x8
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.BatchNorm2d(ch * 4), nn.SiLU(),  # 4x4
        )
        self.head = nn.Linear(ch * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.head(h.flatten(1))


class Decoder(nn.Module):
    """latent_dim -> 32x32x3, mirror of the encoder."""

    def __init__(self, latent_dim: int, ch: int = 64):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(latent_dim, ch * 4 * 4 * 4)
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch * 4), nn.SiLU(),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.SiLU(),  # 8x8
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.SiLU(),          # 16x16
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),                                              # 32x32
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.ch * 4, 4, 4)
        return torch.tanh(self.net(h))  # images are normalized to [-1, 1]


class ResidualBlock(nn.Module):
    """Two-convolution residual block that preserves shape and channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResidualDownBlock(nn.Module):
    """Stride-2 residual block used by the stronger encoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResidualUpBlock(nn.Module):
    """Resize-convolution residual block used by the stronger decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.act(self.main(x) + self.skip(x))


class ResidualEncoder(nn.Module):
    """image_size x image_size x3 -> latent_dim, bottleneck-preserving residual blocks.

    Downsamples by stride-2 blocks until the spatial size reaches 4x4, so the
    same head shape (ch*4 channels @ 4x4) works for any image_size that is a
    power of two >= 8 -- image_size=32 (3 blocks) reproduces the original CIFAR
    architecture exactly; image_size=64 adds one more block.
    """

    def __init__(self, latent_dim: int, ch: int = 64, image_size: int = 32):
        super().__init__()
        n_down = (image_size // 4).bit_length() - 1
        if 4 << n_down != image_size:
            raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
        out_ch = [ch * 2 ** min(i, 2) for i in range(n_down)]
        in_ch = [ch] + out_ch[:-1]

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            *(ResidualDownBlock(i, o) for i, o in zip(in_ch, out_ch)),
            ResidualBlock(out_ch[-1]),
        )
        self.head = nn.Linear(out_ch[-1] * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(self.stem(x))
        return self.head(h.flatten(1))


class ResidualDecoder(nn.Module):
    """latent_dim -> image_size x image_size x3, mirrors ResidualEncoder."""

    def __init__(self, latent_dim: int, ch: int = 64, image_size: int = 32):
        super().__init__()
        n_down = (image_size // 4).bit_length() - 1
        if 4 << n_down != image_size:
            raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
        enc_out_ch = [ch * 2 ** min(i, 2) for i in range(n_down)]
        bottleneck_ch = enc_out_ch[-1]
        dec_out_ch = list(reversed(enc_out_ch[:-1])) + [ch]
        dec_in_ch = [bottleneck_ch] + dec_out_ch[:-1]

        self.ch = ch
        self.bottleneck_ch = bottleneck_ch
        self.fc = nn.Linear(latent_dim, bottleneck_ch * 4 * 4)
        self.pre = ResidualBlock(bottleneck_ch)
        self.net = nn.Sequential(
            *(ResidualUpBlock(i, o) for i, o in zip(dec_in_ch, dec_out_ch)),
            nn.Conv2d(ch, 3, 3, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.bottleneck_ch, 4, 4)
        return torch.tanh(self.net(self.pre(h)))


class SpatialResidualDownBlock(nn.Module):
    """Residual downsampling with a parameter-free space-to-depth skip."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        skip_channels = in_channels * 4
        if skip_channels % out_channels:
            raise ValueError(
                f"cannot group {skip_channels} skip channels into {out_channels}"
            )
        self.out_channels = out_channels
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.act = nn.SiLU()

    def skip(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pixel_unshuffle(x, 2)
        batch, channels, height, width = x.shape
        group = channels // self.out_channels
        return x.view(batch, self.out_channels, group, height, width).mean(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class SpatialResidualUpBlock(nn.Module):
    """Nearest-neighbour resize-convolution block with a learned residual skip."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))
        return self.act(self.main(x) + skip)


class SpatialResidualEncoder(nn.Module):
    """image_size x image_size encoder whose flat code retains a 4x4 spatial
    organization. The stem does one stride-2 halving; additional halvings
    (image_size=64 needs one more than image_size=32) repeat at ch*2 width,
    so image_size=32 reproduces the original CIFAR architecture exactly."""

    def __init__(self, latent_channels: int, ch: int = 64, image_size: int = 32,
                width_mult: int = 2):
        super().__init__()
        n_down = (image_size // 4).bit_length() - 1
        if 4 << n_down != image_size:
            raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
        n_blocks = n_down - 1  # halvings after the stem's own halving
        out_ch = [ch * width_mult] * (n_blocks - 1) + [latent_channels]
        in_ch = [ch] + out_ch[:-1]

        self.latent_channels = latent_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),                  # image_size/2
            nn.SiLU(),
        )
        self.features = nn.Sequential(
            *(SpatialResidualDownBlock(i, o) for i, o in zip(in_ch, out_ch))
        )
        self.latent_norm = nn.BatchNorm1d(latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(self.stem(x))
        batch = z.shape[0]
        z = self.latent_norm(z.view(batch, self.latent_channels, -1))
        return z.flatten(1)


class SpatialResidualDecoder(nn.Module):
    """Decode a flattened Cx4x4 code while retaining its spatial topology,
    mirroring SpatialResidualEncoder for any supported image_size."""

    def __init__(self, latent_channels: int, ch: int = 64, image_size: int = 32,
                width_mult: int = 2):
        super().__init__()
        n_down = (image_size // 4).bit_length() - 1
        if 4 << n_down != image_size:
            raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
        n_blocks = n_down - 1
        enc_out_ch = [ch * width_mult] * (n_blocks - 1) + [latent_channels]
        enc_in_ch = [ch] + enc_out_ch[:-1]
        dec_out_ch = list(reversed(enc_in_ch))
        dec_in_ch = [latent_channels] + dec_out_ch[:-1]

        self.latent_channels = latent_channels
        self.net = nn.Sequential(
            *(SpatialResidualUpBlock(i, o) for i, o in zip(dec_in_ch, dec_out_ch)),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, 3, 3, 1, 1),                       # image_size
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.shape[0], self.latent_channels, 4, 4)
        return torch.tanh(self.net(z))


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int, ch: int = 64, architecture: str = "plain",
                image_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.architecture = architecture
        if architecture == "plain" and image_size != 32:
            raise ValueError("architecture='plain' assumes image_size=32")
        if architecture == "plain":
            self.enc = Encoder(latent_dim, ch)
            self.dec = Decoder(latent_dim, ch)
        elif architecture == "residual":
            self.enc = ResidualEncoder(latent_dim, ch, image_size)
            self.dec = ResidualDecoder(latent_dim, ch, image_size)
        elif architecture == "spatial":
            if latent_dim % 16:
                raise ValueError("spatial architecture requires latent_dim divisible by 16")
            latent_channels = latent_dim // 16
            self.enc = SpatialResidualEncoder(latent_channels, ch, image_size)
            self.dec = SpatialResidualDecoder(latent_channels, ch, image_size)
        elif architecture == "hybrid":
            # spatial's structure-preserving 4x4 grid bottleneck, widened to
            # residual's channel capacity (ch*4 instead of spatial's ch*2) --
            # tests whether the two levers combine for mutual benefit.
            if latent_dim % 16:
                raise ValueError("hybrid architecture requires latent_dim divisible by 16")
            latent_channels = latent_dim // 16
            self.enc = SpatialResidualEncoder(latent_channels, ch, image_size, width_mult=4)
            self.dec = SpatialResidualDecoder(latent_channels, ch, image_size, width_mult=4)
        else:
            raise ValueError(
                f"unknown autoencoder architecture {architecture!r}; "
                "expected 'plain', 'residual', 'spatial', or 'hybrid'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def reconstruction_loss(prediction, target, *, topk_percent: float = 100.0):
    """MSE, optionally restricted to each image's largest elementwise errors."""
    if not 0 < topk_percent <= 100:
        raise ValueError("topk_percent must be in (0, 100]")
    squared_error = (prediction - target).square()
    if topk_percent == 100:
        return squared_error.mean()
    per_image = squared_error.flatten(1)
    k = max(1, math.ceil(per_image.shape[1] * topk_percent / 100))
    return per_image.topk(k, dim=1, sorted=False).values.mean()


def train_autoencoder(ae, loader, *, epochs, lr, device, topk_percent=100.0,
                      log=print):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    ae.train()
    for ep in range(epochs):
        running, running_full_mse, n = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            xr = ae(x)
            loss = reconstruction_loss(xr, x, topk_percent=topk_percent)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * x.size(0)
            running_full_mse += nn.functional.mse_loss(xr.detach(), x).item() * x.size(0)
            n += x.size(0)
        if topk_percent == 100:
            log(f"  [AE] epoch {ep+1}/{epochs}  recon_mse={running/n:.5f}")
        else:
            log(f"  [AE] epoch {ep+1}/{epochs}  top{topk_percent:g}%_mse="
                f"{running/n:.5f}  recon_mse={running_full_mse/n:.5f}")
    return ae


@torch.no_grad()
def encode_all(ae, loader, device):
    """Encode the full dataset into the persistent particle initialization h_i."""
    ae.eval()
    hs = []
    for x, _ in loader:
        hs.append(ae.enc(x.to(device)).cpu())
    return torch.cat(hs, 0)
