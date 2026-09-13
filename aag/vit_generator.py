"""ViT-style one-pass generator (user request 2026-09-17): learned position tokens for a 16x16 grid of 16-px patches,
z -> linear rank-r bottleneck -> a few z tokens; mode 'self' = z tokens prepended and plain self-attention over all
tokens; mode 'cross' = position tokens self-attend and cross-attend to the z tokens in every block. Linear patch head
+ a small full-resolution conv smoother (its last conv is `.out`, used by the trainer's adaptive adversarial weight).
Same forward(z, y) interface as Generator256; unconditional only."""
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, cross=False):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.nc = nn.LayerNorm(dim); self.nkv = nn.LayerNorm(dim); self.xattn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, ctx=None):
        h = self.n1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        if self.cross:
            q = self.nc(x); kv = self.nkv(ctx); x = x + self.xattn(q, kv, kv, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViTGenerator(nn.Module):
    def __init__(self, dim_z: int, image_size: int = 256, patch: int = 16, dim: int = 768, depth: int = 12, heads: int = 12,
                 z_bottleneck: int = 64, n_ztok: int = 8, mode: str = "self", mlp_ratio: float = 4.0, smooth_ch: int = 32):
        super().__init__()
        assert mode in ("self", "cross"); self.mode = mode; self.patch = patch; self.g = image_size // patch; self.dim = dim
        r = z_bottleneck if z_bottleneck > 0 else dim_z
        self.bott = nn.Linear(dim_z, r) if z_bottleneck > 0 else nn.Identity()
        self.n_ztok = n_ztok; self.ztok = nn.Linear(r, n_ztok * dim)
        self.zpos = nn.Parameter(torch.zeros(1, n_ztok, dim)); self.pos = nn.Parameter(torch.zeros(1, self.g * self.g, dim))
        nn.init.trunc_normal_(self.zpos, std=0.02); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, cross=(mode == "cross")) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, patch * patch * 3)
        self.smooth = nn.Sequential(nn.Conv2d(3, smooth_ch, 3, 1, 1), nn.SiLU()); self.out = nn.Conv2d(smooth_ch, 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start as the pure patch prediction

    def forward(self, z, y=None):
        B = z.shape[0]
        zt = self.ztok(self.bott(z)).view(B, self.n_ztok, self.dim) + self.zpos
        x = self.pos.expand(B, -1, -1)
        if self.mode == "self":
            x = torch.cat([zt, x], 1)
            for b in self.blocks: x = b(x)
            x = x[:, self.n_ztok:]
        else:
            for b in self.blocks: x = b(x, zt)
        p = self.head(self.norm(x))                                            # (B, g*g, patch*patch*3)
        img = p.view(B, self.g, self.g, self.patch, self.patch, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, self.g * self.patch, self.g * self.patch)
        return torch.tanh(img + self.out(self.smooth(img)))


def vit_kwargs_from_ckpt(ck):
    return dict(patch=ck.get("vit_patch", 16), dim=ck.get("vit_dim", 768), depth=ck.get("vit_depth", 12), heads=ck.get("vit_heads", 12),
                z_bottleneck=ck.get("z_bottleneck", 64), n_ztok=ck.get("vit_ztokens", 8), mode=ck.get("vit_mode", "self"))
