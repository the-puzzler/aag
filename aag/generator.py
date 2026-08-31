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
