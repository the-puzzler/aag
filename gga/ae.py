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
    """32x32x3 -> latent_dim using bottleneck-preserving residual blocks."""

    def __init__(self, latent_dim: int, ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            ResidualDownBlock(ch, ch),                    # 16x16
            ResidualDownBlock(ch, ch * 2),                # 8x8
            ResidualDownBlock(ch * 2, ch * 4),            # 4x4
            ResidualBlock(ch * 4),
        )
        self.head = nn.Linear(ch * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(self.stem(x))
        return self.head(h.flatten(1))


class ResidualDecoder(nn.Module):
    """latent_dim -> 32x32x3 using residual resize-convolution blocks."""

    def __init__(self, latent_dim: int, ch: int = 64):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(latent_dim, ch * 4 * 4 * 4)
        self.pre = ResidualBlock(ch * 4)
        self.net = nn.Sequential(
            ResidualUpBlock(ch * 4, ch * 2),              # 8x8
            ResidualUpBlock(ch * 2, ch),                  # 16x16
            ResidualUpBlock(ch, ch),                      # 32x32
            nn.Conv2d(ch, 3, 3, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.ch * 4, 4, 4)
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
    """CIFAR encoder whose flat code retains a 4x4 spatial organization."""

    def __init__(self, latent_channels: int, ch: int = 64):
        super().__init__()
        self.latent_channels = latent_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),                  # 16x16
            nn.SiLU(),
        )
        self.features = nn.Sequential(
            SpatialResidualDownBlock(ch, ch * 2),       # 8x8
            SpatialResidualDownBlock(ch * 2, latent_channels),  # 4x4
        )
        self.latent_norm = nn.BatchNorm1d(latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(self.stem(x))
        batch = z.shape[0]
        z = self.latent_norm(z.view(batch, self.latent_channels, -1))
        return z.flatten(1)


class SpatialResidualDecoder(nn.Module):
    """Decode a flattened Cx4x4 code while retaining its spatial topology."""

    def __init__(self, latent_channels: int, ch: int = 64):
        super().__init__()
        self.latent_channels = latent_channels
        self.net = nn.Sequential(
            SpatialResidualUpBlock(latent_channels, ch * 2),  # 8x8
            SpatialResidualUpBlock(ch * 2, ch),               # 16x16
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, 3, 3, 1, 1),                       # 32x32
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.shape[0], self.latent_channels, 4, 4)
        return torch.tanh(self.net(z))


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int, ch: int = 64, architecture: str = "plain"):
        super().__init__()
        self.latent_dim = latent_dim
        self.architecture = architecture
        if architecture == "plain":
            self.enc = Encoder(latent_dim, ch)
            self.dec = Decoder(latent_dim, ch)
        elif architecture == "residual":
            self.enc = ResidualEncoder(latent_dim, ch)
            self.dec = ResidualDecoder(latent_dim, ch)
        elif architecture == "spatial":
            if latent_dim % 16:
                raise ValueError("spatial architecture requires latent_dim divisible by 16")
            latent_channels = latent_dim // 16
            self.enc = SpatialResidualEncoder(latent_channels, ch)
            self.dec = SpatialResidualDecoder(latent_channels, ch)
        else:
            raise ValueError(
                f"unknown autoencoder architecture {architecture!r}; "
                "expected 'plain', 'residual', or 'spatial'"
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
