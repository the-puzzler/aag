"""Hybrid generator: ViT trunk (z tokens prepended, self-attention) over a 16x16 grid of learned position tokens,
then the conv upsampler of Generator256 (its UpStage / CondResBlock / AdaGroupNorm) from 16x16 to the image.
Isolates token mixing from the linear patch head that the pure ViT uses. Unconditional; forward(z, y) interface."""
import torch, torch.nn as nn, torch.nn.functional as F
from aag.generator256 import CondResBlock, UpStage, AdaGroupNorm
from aag.vit_generator import Block


class HybridGenerator(nn.Module):
    def __init__(self, dim_z: int, image_size: int = 256, tok_grid: int = 16, dim: int = 512, depth: int = 8, heads: int = 8,
                 z_bottleneck: int = 64, n_ztok: int = 8, width: float = 1.4, n_res: int = 2, base_channels=(512, 256, 128, 64), cz: int = 64,
                 n_classes: int = 0):
        super().__init__()
        # class conditioning: one learned class token appended to the z tokens (the trunk sees it in self-attention)
        self.n_classes = n_classes
        self.cls_emb = nn.Embedding(n_classes, dim) if n_classes else None
        self.g, self.dim, self.n_ztok, self.cz = tok_grid, dim, n_ztok, cz
        r = z_bottleneck if z_bottleneck > 0 else dim_z
        self.bott = nn.Linear(dim_z, r) if z_bottleneck > 0 else nn.Identity()
        self.ztok = nn.Linear(r, n_ztok * dim)
        self.zpos = nn.Parameter(torch.zeros(1, n_ztok, dim)); self.pos = nn.Parameter(torch.zeros(1, tok_grid * tok_grid, dim))
        nn.init.trunc_normal_(self.zpos, std=0.02); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, 4.0, cross=False) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.to_map = nn.Linear(dim, cz)
        n_up = (image_size // tok_grid).bit_length() - 1
        assert tok_grid << n_up == image_size
        chans = [max(32, int(round(c * width / 32)) * 32) for c in base_channels[:n_up]]
        if len(chans) < n_up: chans += [chans[-1]] * (n_up - len(chans))
        self.stem = nn.Conv2d(cz, chans[0], 3, 1, 1)
        self.pre = nn.ModuleList([CondResBlock(chans[0], chans[0], None) for _ in range(n_res)])
        stages, cin = [], chans[0]
        for c in chans:
            stages.append(UpStage(cin, c, None, n_res)); cin = c
        self.stages = nn.ModuleList(stages)
        self.out_norm = AdaGroupNorm(cin, None); self.out = nn.Conv2d(cin, 3, 3, 1, 1)

    def forward(self, z, y=None):
        B = z.shape[0]
        zt = self.ztok(self.bott(z)).view(B, self.n_ztok, self.dim) + self.zpos
        n_pre = self.n_ztok
        if self.cls_emb is not None:
            assert y is not None, "class-conditional hybrid needs y"
            zt = torch.cat([zt, self.cls_emb(y).unsqueeze(1)], 1); n_pre += 1
        x = torch.cat([zt, self.pos.expand(B, -1, -1)], 1)
        for b in self.blocks: x = b(x)
        x = self.to_map(self.norm(x[:, n_pre:]))                                        # (B, g*g, cz)
        h = self.stem(x.transpose(1, 2).reshape(B, self.cz, self.g, self.g))
        for r in self.pre: h = r(h, None)
        for s in self.stages: h = s(h, None)
        return torch.tanh(self.out(F.silu(self.out_norm(h, None))))


def hybrid_kwargs_from_ckpt(ck):
    return dict(tok_grid=ck.get("vit_patch", 16), dim=ck.get("vit_dim", 512), depth=ck.get("vit_depth", 8), heads=ck.get("vit_heads", 8),
                z_bottleneck=ck.get("z_bottleneck", 64), n_ztok=ck.get("vit_ztokens", 8), width=ck.get("width", 1.4), n_res=ck.get("n_res", 2),
                n_classes=int(ck.get("n_classes", 0)))
