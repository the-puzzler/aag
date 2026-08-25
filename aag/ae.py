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


class DCAEUpBlock(nn.Module):
    """Upsample block with DC-AE Residual Autoencoding (arXiv 2410.10733).

    SpatialResidualUpBlock uses a LEARNED 1x1 projection for its skip. DC-AE
    instead makes the identity path parameter-free, so the convolutions only have
    to learn the correction rather than the whole mapping -- the paper's stated
    fix for "the optimization difficulty of high spatial-compression
    autoencoders", which at 192x compression is exactly our regime.

    Their upsample shortcut is channel-to-space with channel DUPLICATION:
    H/2 x W/2 x 2C  -> pixel_shuffle -> H x W x C/2, duplicated and concatenated
    to H x W x C. Implemented here as tile-then-shuffle, which is the same thing
    and generalises when the channel counts are not an exact factor of two --
    matching how SpatialResidualUpBlock3d in this file already does it.

    The encoder side needs no change: SpatialResidualDownBlock.skip is already
    parameter-free space-to-depth with channel averaging, which is DC-AE's
    downsample shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.need = out_channels * 4          # what pixel_shuffle(2) consumes
        self.act = nn.SiLU()

    def skip(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c < self.need:                     # duplicate
            reps = -(-self.need // c)
            x = x.repeat(1, reps, 1, 1)[:, :self.need]
        elif c > self.need:                   # average groups
            x = x.view(x.shape[0], self.need, c // self.need, *x.shape[2:]).mean(2)
        return nn.functional.pixel_shuffle(x, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


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


class DCAEDecoder(SpatialResidualDecoder):
    """SpatialResidualDecoder with DC-AE parameter-free upsample shortcuts."""

    def __init__(self, latent_channels: int, ch: int = 64, image_size: int = 32,
                 width_mult: int = 2):
        super().__init__(latent_channels, ch, image_size, width_mult)
        swapped = []
        for m in self.net:
            if isinstance(m, SpatialResidualUpBlock):
                i = m.main[1].in_channels
                o = m.main[1].out_channels
                swapped.append(DCAEUpBlock(i, o))
            else:
                swapped.append(m)
        self.net = nn.Sequential(*swapped)


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
        elif architecture == "dcae":
            # DC-AE (arXiv 2410.10733) residual autoencoding: same spatial grid
            # bottleneck as 'hybrid', but the decoder's upsample skips are
            # parameter-free channel-to-space instead of learned 1x1 convs.
            if latent_dim % 16:
                raise ValueError("dcae architecture requires latent_dim divisible by 16")
            latent_channels = latent_dim // 16
            self.enc = SpatialResidualEncoder(latent_channels, ch, image_size, width_mult=4)
            self.dec = DCAEDecoder(latent_channels, ch, image_size, width_mult=4)
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
                "expected 'plain', 'residual', 'spatial', 'hybrid', or 'dcae'"
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




# --------------------------------------------------------------------------- #
# 3D (video) variants -- T frames x H x W
#
# Temporal kernels differ by whether that block downsamples time:
#   downsample -> k=4 s=2 p=1  (T -> T/2)
#   keep       -> k=3 s=1 p=1  (T -> T; k=4 would shrink T=2 to 1)
# --------------------------------------------------------------------------- #
class ResidualBlock3d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, 1, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, 1, 1), nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


def _plan(image_size: int, frames: int):
    """-> (channel widths, per-block 'does this block halve time', final T)"""
    n_down = (image_size // 4).bit_length() - 1
    if 4 << n_down != image_size:
        raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
    chs, halve, t = [], [], frames
    for i in range(n_down):
        chs.append(64 * 2 ** min(i, 2))
        h = t > 2
        halve.append(h)
        if h:
            t //= 2
    return chs, halve, t, n_down


class ResidualEncoder3d(nn.Module):
    """(B,3,T,H,W) -> latent_dim."""

    def __init__(self, latent_dim: int, ch: int = 64, image_size: int = 64, frames: int = 16):
        super().__init__()
        base, halve, t_out, n_down = _plan(image_size, frames)
        chs = [ch * (c // 64) for c in base]
        layers, cin = [], 3
        for cout, h in zip(chs, halve):
            k = (4, 4, 4) if h else (3, 4, 4)
            st = (2, 2, 2) if h else (1, 2, 2)
            layers += [nn.Conv3d(cin, cout, k, st, (1, 1, 1)), nn.GroupNorm(8, cout), nn.SiLU(),
                       ResidualBlock3d(cout)]
            cin = cout
        self.net = nn.Sequential(*layers)
        self.t_out, self.s_out, self.c_out = t_out, image_size // (2 ** n_down), cin
        self.fc = nn.Linear(cin * t_out * self.s_out * self.s_out, latent_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class AdaLNDecoder3d(nn.Module):
    """z -> (B,3,T,H,W), with the condition injected by adaptive GroupNorm.

    Concatenating the condition onto z feeds it in once, at the input, after
    which a deep decoder dilutes it. AdaLN instead predicts a per-channel
    (scale, shift) from c at EVERY norm layer, so the condition keeps steering
    the whole stack. It also matches what z and c actually are here: z is the
    sample being decoded, c shapes the decoding function -- concat pretends they
    are interchangeable inputs.

    Norms are affine=False; the affine part comes from c. Each modulation MLP's
    final layer is zero-initialised so the block starts at scale=1, shift=0 --
    i.e. identical to the unconditioned network, and conditioning is learned
    rather than a random perturbation at init.
    """

    def __init__(self, dim_z: int, cond_dim: int, ch: int = 64, image_size: int = 64,
                 frames: int = 16, hidden: int = 256):
        super().__init__()
        base, halve, t_out, n_down = _plan(image_size, frames)
        chs = [ch * (c // 64) for c in base]
        self.t0, self.s0, self.c0 = t_out, image_size // (2 ** n_down), chs[-1]
        self.fc = nn.Linear(dim_z, self.c0 * self.t0 * self.s0 * self.s0)
        dec_out = list(reversed(chs[:-1])) + [ch]
        dec_in = [self.c0] + dec_out[:-1]
        up = list(reversed(halve))

        self.ups, self.norms, self.mods, self.blocks = (nn.ModuleList() for _ in range(4))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU())
        for cin, cout, h in zip(dec_in, dec_out, up):
            k = (4, 4, 4) if h else (3, 4, 4)
            st = (2, 2, 2) if h else (1, 2, 2)
            self.ups.append(nn.ConvTranspose3d(cin, cout, k, st, (1, 1, 1)))
            self.norms.append(nn.GroupNorm(8, cout, affine=False))
            mod = nn.Linear(hidden, 2 * cout)
            nn.init.zeros_(mod.weight); nn.init.zeros_(mod.bias)   # start as identity
            self.mods.append(mod)
            self.blocks.append(ResidualBlock3d(cout))
        self.out = nn.Conv3d(ch, 3, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, z, cond):
        w = self.cond_mlp(cond)
        x = self.fc(z).view(-1, self.c0, self.t0, self.s0, self.s0)
        for up, norm, mod, blk in zip(self.ups, self.norms, self.mods, self.blocks):
            x = norm(up(x))
            scale, shift = mod(w).chunk(2, dim=1)
            x = x * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
            x = blk(self.act(x))
        return torch.tanh(self.out(x))


class SpatialCondDecoder3d(nn.Module):
    """z -> (B,3,T,H,W) with the condition injected as SPATIAL feature maps.

    The condition here is a frame encoded by the `hybrid` 2D AE, whose code is
    really C_c channels on a 4x4 grid -- it knows *where* things are. Flattening
    it into a vector for AdaLN discards that: AdaLN emits one scale/shift per
    channel for the whole map, so the decoder is told "this kind of scene" but
    not "wall on the left, corridor ahead". For continuing a specific layout,
    that is exactly the information that should transfer.

    So the condition grid is tiled across the decoder's time axis and
    concatenated as extra channels at the first stage, where it lines up with
    the decoder's own (C, t0, 4, 4) grid with no resampling.

    AdaLN modulation is KEPT alongside it: the two are complementary -- spatial
    channels carry layout, the global modulation carries scene-level style.
    """

    def __init__(self, dim_z: int, cond_channels: int, ch: int = 64, image_size: int = 64,
                 frames: int = 16, cond_grid: int = 4, hidden: int = 256):
        super().__init__()
        base, halve, t_out, n_down = _plan(image_size, frames)
        chs = [ch * (c // 64) for c in base]
        self.t0, self.s0, self.c0 = t_out, image_size // (2 ** n_down), chs[-1]
        if cond_grid != self.s0:
            raise ValueError(f"condition grid {cond_grid} must match decoder start {self.s0}")
        self.cond_channels = cond_channels
        self.fc = nn.Linear(dim_z, self.c0 * self.t0 * self.s0 * self.s0)
        # the condition's channels ride alongside z's at the first stage
        self.merge = nn.Conv3d(self.c0 + cond_channels, self.c0, 1)

        dec_out = list(reversed(chs[:-1])) + [ch]
        dec_in = [self.c0] + dec_out[:-1]
        up = list(reversed(halve))
        self.ups, self.norms, self.mods, self.blocks = (nn.ModuleList() for _ in range(4))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_channels * cond_grid * cond_grid, hidden),
                                      nn.SiLU())
        for cin, cout, h in zip(dec_in, dec_out, up):
            k = (4, 4, 4) if h else (3, 4, 4)
            st = (2, 2, 2) if h else (1, 2, 2)
            self.ups.append(nn.ConvTranspose3d(cin, cout, k, st, (1, 1, 1)))
            self.norms.append(nn.GroupNorm(8, cout, affine=False))
            mod = nn.Linear(hidden, 2 * cout)
            nn.init.zeros_(mod.weight); nn.init.zeros_(mod.bias)
            self.mods.append(mod)
            self.blocks.append(ResidualBlock3d(cout))
        self.out = nn.Conv3d(ch, 3, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, z, cond):
        b = z.shape[0]
        cg = cond.view(b, self.cond_channels, self.s0, self.s0)          # (B,Cc,4,4)
        cg = cg.unsqueeze(2).expand(-1, -1, self.t0, -1, -1)             # tile over time
        x = self.fc(z).view(b, self.c0, self.t0, self.s0, self.s0)
        x = self.merge(torch.cat([x, cg], dim=1))                        # spatial injection
        w = self.cond_mlp(cond)
        for up, norm, mod, blk in zip(self.ups, self.norms, self.mods, self.blocks):
            x = norm(up(x))
            scale, shift = mod(w).chunk(2, dim=1)
            x = x * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
            x = blk(self.act(x))
        return torch.tanh(self.out(x))


class ResidualDecoder3d(nn.Module):
    """latent_dim -> (B,3,T,H,W). Mirrors ResidualEncoder3d.

    Doubles as the direct-to-pixel video generator (z -> video), which is why
    it lives here rather than in the generator script.
    """

    def __init__(self, latent_dim: int, ch: int = 64, image_size: int = 64, frames: int = 16):
        super().__init__()
        base, halve, t_out, n_down = _plan(image_size, frames)
        chs = [ch * (c // 64) for c in base]
        self.t0, self.s0, self.c0 = t_out, image_size // (2 ** n_down), chs[-1]
        self.fc = nn.Linear(latent_dim, self.c0 * self.t0 * self.s0 * self.s0)
        dec_out = list(reversed(chs[:-1])) + [ch]
        dec_in = [self.c0] + dec_out[:-1]
        up = list(reversed(halve))          # mirror the encoder's time schedule
        layers = []
        for cin, cout, h in zip(dec_in, dec_out, up):
            k = (4, 4, 4) if h else (3, 4, 4)
            st = (2, 2, 2) if h else (1, 2, 2)
            layers += [nn.ConvTranspose3d(cin, cout, k, st, (1, 1, 1)),
                       nn.GroupNorm(8, cout), nn.SiLU(), ResidualBlock3d(cout)]
        layers += [nn.Conv3d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z).view(-1, self.c0, self.t0, self.s0, self.s0)
        return torch.tanh(self.net(h))


def _space_to_depth3d(x, ft, fh, fw):
    """Parameter-free (B,C,T,H,W) -> (B,C*ft*fh*fw,T/ft,H/fh,W/fw) rearrange."""
    b, c, t, h, w = x.shape
    x = x.view(b, c, t // ft, ft, h // fh, fh, w // fw, fw)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(b, c * ft * fh * fw, t // ft, h // fh, w // fw)


class SpatialResidualDownBlock3d(nn.Module):
    """3D residual downsample with a parameter-free space-to-depth skip.

    Always uses kernel=3 (never 4): with padding=1 a kernel of 4 shrinks a
    length-2 temporal axis to 1, which silently breaks the round trip.
    """

    def __init__(self, in_channels, out_channels, halve_t: bool):
        super().__init__()
        self.halve_t, self.out_channels = halve_t, out_channels
        self.ft = 2 if halve_t else 1
        skip_channels = in_channels * self.ft * 4
        if skip_channels % out_channels:
            raise ValueError(f"cannot group {skip_channels} skip channels into {out_channels}")
        stride = (2, 2, 2) if halve_t else (1, 2, 2)
        # GroupNorm is load-bearing: without it activations grow through the
        # deeper 3D stack and the decoder's tanh saturates within one epoch,
        # zeroing the gradient (the 2D blocks get away with no norm).
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(min(8, out_channels), out_channels), nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
        )
        self.act = nn.SiLU()

    def skip(self, x):
        x = _space_to_depth3d(x, self.ft, 2, 2)
        b, c, t, h, w = x.shape
        return x.view(b, self.out_channels, c // self.out_channels, t, h, w).mean(2)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


def _depth_to_space3d(x, ft, fh, fw):
    """Inverse of _space_to_depth3d: (B,C*ft*fh*fw,T,H,W) -> (B,C,T*ft,H*fh,W*fw)."""
    b, c, t, h, w = x.shape
    c_out = c // (ft * fh * fw)
    x = x.view(b, c_out, ft, fh, fw, t, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, c_out, t * ft, h * fh, w * fw)


class SpatialResidualUpBlock3d(nn.Module):
    """Nearest-resize + conv3d, mirroring SpatialResidualDownBlock3d."""

    def __init__(self, in_channels, out_channels, double_t: bool):
        super().__init__()
        self.scale = (2 if double_t else 1, 2, 2)
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=self.scale, mode="nearest"),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(8, out_channels), out_channels), nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
        )
        # DC-AE residual autoencoding: mirror the encoder's parameter-free
        # space->channel with channel->space here, so the identity path is free
        # and the convs only learn the correction. Channels are tiled (not
        # projected) up to the count depth-to-space consumes.
        self.ft_up = self.scale[0]
        self.need = out_channels * self.ft_up * 4
        self.in_channels = in_channels
        self.act = nn.SiLU()

    def skip(self, x):
        c = x.shape[1]
        if c < self.need:
            reps = -(-self.need // c)                      # ceil
            x = x.repeat(1, reps, 1, 1, 1)[:, :self.need]
        elif c > self.need:
            x = x.view(x.shape[0], self.need, c // self.need,
                       *x.shape[2:]).mean(2)
        return _depth_to_space3d(x, self.ft_up, 2, 2)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


def _spatial_plan3d(image_size, frames, t_out):
    """-> (n_blocks after the stem, per-block 'does this block halve time')."""
    n_down = (image_size // 4).bit_length() - 1
    if 4 << n_down != image_size:
        raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
    n_blocks = n_down - 1
    ratio = frames // t_out
    if t_out * ratio != frames or (ratio & (ratio - 1)):
        raise ValueError(f"frames/t_out must be a power of two, got {frames}/{t_out}")
    n_t = ratio.bit_length() - 1
    if n_t > n_blocks:
        raise ValueError(f"need {n_t} temporal halvings but only {n_blocks} blocks")
    return n_blocks, [i < n_t for i in range(n_blocks)]


class SpatialResidualEncoder3d(nn.Module):
    """(B,3,T,H,W) -> flat code that retains a C x t_out x 4 x 4 topology.

    Unlike ResidualEncoder3d there is no fc collapse: position stays structural
    rather than being re-encoded into global coordinates, so the same
    latent_dim buys far more reconstructable detail.
    """

    def __init__(self, latent_channels: int, ch: int = 64, image_size: int = 64,
                 frames: int = 16, t_out: int = 4, width_mult: int = 2):
        super().__init__()
        n_blocks, halve = _spatial_plan3d(image_size, frames, t_out)
        out_ch = [ch * width_mult] * (n_blocks - 1) + [latent_channels]
        in_ch = [ch] + out_ch[:-1]
        self.latent_channels, self.t_out = latent_channels, t_out
        self.stem = nn.Sequential(nn.Conv3d(3, ch, 3, (1, 2, 2), 1), nn.SiLU())
        self.features = nn.Sequential(
            *(SpatialResidualDownBlock3d(i, o, h) for i, o, h in zip(in_ch, out_ch, halve))
        )
        self.latent_norm = nn.BatchNorm1d(latent_channels)

    def forward(self, x):
        z = self.features(self.stem(x))
        z = self.latent_norm(z.view(z.shape[0], self.latent_channels, -1))
        return z.flatten(1)


class SpatialResidualDecoder3d(nn.Module):
    """Flat C x t_out x 4 x 4 code -> (B,3,T,H,W). Doubles as the generator."""

    def __init__(self, latent_channels: int, ch: int = 64, image_size: int = 64,
                 frames: int = 16, t_out: int = 4, width_mult: int = 2):
        super().__init__()
        n_blocks, halve = _spatial_plan3d(image_size, frames, t_out)
        enc_out = [ch * width_mult] * (n_blocks - 1) + [latent_channels]
        enc_in = [ch] + enc_out[:-1]
        dec_out = list(reversed(enc_in))
        dec_in = [latent_channels] + dec_out[:-1]
        self.latent_channels, self.t_out = latent_channels, t_out
        self.net = nn.Sequential(
            *(SpatialResidualUpBlock3d(i, o, d)
              for i, o, d in zip(dec_in, dec_out, reversed(halve))),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(ch, 3, 3, 1, 1),
        )

    def forward(self, z):
        z = z.view(z.shape[0], self.latent_channels, self.t_out, 4, 4)
        return torch.tanh(self.net(z))


class VideoAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int, ch: int = 64, image_size: int = 64, frames: int = 16,
                 architecture: str = "residual", t_out: int = 4, width_mult: int = 2):
        super().__init__()
        self.latent_dim, self.architecture = latent_dim, architecture
        if architecture == "residual":
            self.enc = ResidualEncoder3d(latent_dim, ch, image_size, frames)
            self.dec = ResidualDecoder3d(latent_dim, ch, image_size, frames)
        elif architecture == "spatial":
            per_frame = t_out * 16          # t_out x 4 x 4 grid positions
            if latent_dim % per_frame:
                raise ValueError(
                    f"spatial architecture needs latent_dim divisible by {per_frame} "
                    f"(t_out={t_out} x 4 x 4), got {latent_dim}")
            c = latent_dim // per_frame
            self.enc = SpatialResidualEncoder3d(c, ch, image_size, frames, t_out, width_mult)
            self.dec = SpatialResidualDecoder3d(c, ch, image_size, frames, t_out, width_mult)
        else:
            raise ValueError(f"unknown video architecture {architecture!r}; "
                             "expected 'residual' or 'spatial'")

    def forward(self, x):
        return self.dec(self.enc(x))
