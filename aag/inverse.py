"""Inverse model E(x) -> z for AAG (user idea 2026-09-29).

AAG already provides paired data in both directions (z_i <-> x_i). An encoder trained ONLY on the real assigned
pairs, then frozen, gives a pointwise off-anchor constraint for fresh z with no external teacher:
    E(G(z_f)) ~= z_f          (cycle consistency)
The distributional term (MMD / critic) says fresh outputs must lie on the right image distribution; the cycle
term says each fresh z must keep its own location/identity, so the gaps cannot be shuffled or collapsed.

E = frozen DINOv2 ViT-B/14 tokens (CLS + 256 patch tokens) -> learned attention-pooling head -> target.
Target: the full whitened z, or -- when the generator has a hard linear bottleneck z -> r -- the r-dim code
c = W0 z + b0 the generator actually reads (W0, b0 frozen from the base checkpoint); either way standardised
per dimension by the anchor statistics so an MSE of 1.0 = a constant predictor and 0 = perfect inversion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseHead(nn.Module):
    def __init__(self, dim_tok: int = 768, n_tok: int = 257, n_query: int = 8, out_dim: int = 512, depth: int = 2, heads: int = 8):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim_tok)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.queries = nn.Parameter(torch.zeros(1, n_query, dim_tok)); nn.init.trunc_normal_(self.queries, std=0.02)
        self.ln_kv = nn.LayerNorm(dim_tok)
        self.cross = nn.MultiheadAttention(dim_tok, heads, batch_first=True)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dim_tok), "attn": nn.MultiheadAttention(dim_tok, heads, batch_first=True),
                "ln2": nn.LayerNorm(dim_tok), "mlp": nn.Sequential(nn.Linear(dim_tok, 4 * dim_tok), nn.GELU(), nn.Linear(4 * dim_tok, dim_tok))}))
        self.ln_out = nn.LayerNorm(n_query * dim_tok)
        self.proj = nn.Linear(n_query * dim_tok, out_dim)
        self.n_query, self.dim_tok, self.out_dim = n_query, dim_tok, out_dim

    def forward(self, cls: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        tok = torch.cat([cls[:, None], patch], 1)
        if tok.shape[1] != self.pos.shape[1]:
            raise ValueError(f"expected {self.pos.shape[1]} tokens, got {tok.shape[1]}")
        kv = self.ln_kv(tok + self.pos)
        q = self.queries.expand(tok.shape[0], -1, -1)
        h = q + self.cross(q, kv, kv, need_weights=False)[0]
        for b in self.blocks:
            x = b["ln1"](h); h = h + b["attn"](x, x, x, need_weights=False)[0]
            h = h + b["mlp"](b["ln2"](h))
        return self.proj(self.ln_out(h.flatten(1)))


class InverseModel:
    """Frozen E with its target definition. `cycle_loss(x, z)` takes IMAGES: with the conv encoder E sees them
    directly, with the DINO head they pass through the frozen backbone first."""

    def __init__(self, ck: dict, dev, dino=None):
        self.encoder = ck.get("encoder", "dino")
        self.kind = ck["kind"]                                   # "z" or "code"
        self.W = ck["W"].to(dev) if ck.get("W") is not None else None
        self.b = ck["b"].to(dev) if ck.get("b") is not None else None
        self.mu, self.sd = ck["mu"].to(dev), ck["sd"].to(dev)
        if self.encoder == "conv":
            from aag.inverse_conv import ConvInverse, conv_kwargs_from_ckpt
            self.net = ConvInverse(**conv_kwargs_from_ckpt(ck)).to(dev)
            self.dino = None
        else:
            self.net = InverseHead(**ck["head_kwargs"]).to(dev)
            self.dino = dino
            assert dino is not None, "the DINO-feature inverse needs the frozen backbone"
        self.net.load_state_dict(ck["head"]); self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.r2_val = ck.get("r2_val")

    def raw_target(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W.t() + self.b if self.kind == "code" else z

    def target(self, z: torch.Tensor) -> torch.Tensor:
        return (self.raw_target(z.float()) - self.mu) / self.sd

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder == "conv":
            return self.net(x)
        cls, patch = self.dino(x)
        return self.net(cls.float(), patch.float())

    def cycle_loss(self, x, z) -> torch.Tensor:
        """mean squared error in standardised target space: 1.0 = constant predictor, 0 = perfect inversion"""
        return F.mse_loss(self.predict(x).float(), self.target(z))


def load_inverse(path: str, dev, dino=None) -> InverseModel:
    return InverseModel(torch.load(path, map_location="cpu", weights_only=False), dev, dino)
