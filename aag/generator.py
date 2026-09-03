"""Transformer generator: z attends to the context, then a conv decoder reads
the z token.

The flat-vector generator (ResidualDecoder over [z, cond, action]) puts 74% of
its 107.5M parameters in the first Linear, purely to absorb the 6144-dim
context, and gives z no way to interrogate that context -- the two are simply
adjacent in one long vector, and the model is free to lean on whichever is
easier to fit.

Here the 24 context frames are 24 tokens, z is one token, the action is one
token, and self-attention lets z query the history before anything is decoded.
Only the z token's output is decoded, so whatever the frame is built from has
to have passed through that query.

Interface matches the conv generator: forward() takes the same flat
[z | cond | action_tail] row so the training script does not care which is in
use.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ae import ResidualDecoder


class TransformerGenerator(nn.Module):
    def __init__(self, dim_z: int, ctx_frames: int, ctx_dim: int, act_dim: int,
                 d_model: int = 512, depth: int = 6, heads: int = 8,
                 mlp_ratio: float = 4.0, ch: int = 192, image_size: int = 64):
        super().__init__()
        self.dim_z, self.ctx_frames, self.ctx_dim = dim_z, ctx_frames, ctx_dim
        self.act_dim = act_dim

        self.emb_z = nn.Linear(dim_z, d_model)
        self.emb_ctx = nn.Linear(ctx_dim, d_model)
        self.emb_act = nn.Linear(act_dim, d_model)
        # one learned position per context frame, plus distinct slots for the z
        # and action tokens. Temporal order matters, so this is not optional.
        self.pos = nn.Parameter(torch.zeros(1, ctx_frames + 2, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=int(d_model * mlp_ratio),
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        # the conv decoder is unchanged; it just reads d_model instead of 6490
        self.dec = ResidualDecoder(d_model, ch=ch, image_size=image_size)

    def forward(self, row: torch.Tensor) -> torch.Tensor:
        B = row.shape[0]
        z = row[:, :self.dim_z]
        c = row[:, self.dim_z:self.dim_z + self.ctx_frames * self.ctx_dim]
        a = row[:, self.dim_z + self.ctx_frames * self.ctx_dim:]

        t_z = self.emb_z(z).unsqueeze(1)                              # (B,1,D)
        t_c = self.emb_ctx(c.view(B, self.ctx_frames, self.ctx_dim))  # (B,24,D)
        t_a = self.emb_act(a).unsqueeze(1)                            # (B,1,D)
        x = torch.cat([t_z, t_c, t_a], 1) + self.pos
        x = self.norm(self.blocks(x))
        return self.dec(x[:, 0])          # decode ONLY the z token


class PixelContextEncoder(nn.Module):
    """Context frames as PIXELS, encoded to ctx_dim by a learned encoder.

    Replaces the frozen AE latents that TransformerGenerator's context tokens
    were read from. It emits ctx_dim per frame, so it drops into the existing
    `emb_ctx = Linear(ctx_dim, d_model)` with no change to the generator: the
    only difference is where the context vectors come from.

    Why bother. The AE is a reconstruction bottleneck trained for reconstruction,
    not prediction, and one encode-decode destroys 6.55 mean |pixel| against a
    real consecutive-frame step of 7.16 -- 91%. Velocity lives in the DIFFERENCE
    between consecutive context vectors, so a channel that noisy is a poor place
    to read speed from, which is a candidate mechanism for the measured failure
    to modulate motion magnitude (model own-motion ~1.1-1.3 regardless of scene,
    against real scene motion of 1.40 and 7.94).

    And in an autoregressive rollout it removes the AE from the loop entirely.
    With AE context, every refeed step is generated pixels -> ae.enc -> latent
    through an encoder never trained on generated images. Here the generated
    pixels go straight back in through an encoder trained on exactly that path.

    No recency weighting is applied. The AE context was scaled by
    sqrt(gamma**i) so a plain L2 would be the recency-weighted L2 -- a property
    the TRANSPORT needed. The generator has a learned positional embedding per
    context slot, so ordering is already represented, and the encoder is free to
    learn any per-slot scaling it wants.
    """

    def __init__(self, ctx_dim: int = 256, ch: int = 64, image_size: int = 64,
                 in_ch: int = 3):
        super().__init__()
        c1, c2, c3, c4 = ch, ch * 2, ch * 4, ch * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 4, 2, 1), nn.GroupNorm(8, c1), nn.GELU(),   # 32
            nn.Conv2d(c1, c2, 4, 2, 1), nn.GroupNorm(8, c2), nn.GELU(),      # 16
            nn.Conv2d(c2, c3, 4, 2, 1), nn.GroupNorm(8, c3), nn.GELU(),      # 8
            nn.Conv2d(c3, c4, 4, 2, 1), nn.GroupNorm(8, c4), nn.GELU(),      # 4
        )
        self.head = nn.Linear(c4 * (image_size // 16) ** 2, ctx_dim)
        self.ctx_dim = ctx_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) in [-1,1] -> (B, T, ctx_dim)."""
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, *frames.shape[2:])
        x = self.net(x).flatten(1)
        return self.head(x).view(B, T, self.ctx_dim)
