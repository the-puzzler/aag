"""The context window used when rolling a VPT generator forward on its own output.

There are two kinds of context and they slide differently, and getting this
wrong at inference silently measures a model that was never trained:

  ae_latent  the window holds AE LATENTS, recency-scaled by sqrt(gamma**i).
             Sliding means encoding the generated frame with the frozen AE. This
             is what the original generators used.
  pixel      the window holds FRAMES. Sliding means appending the generated
             frame; the encoder (a fresh PixelContextEncoder, or the AE's own
             encoder fine-tuned) is re-run on the window every step, and the AE
             never appears in the loop. No recency scaling -- sqrt(gamma**i)
             existed so a plain L2 would be the recency-weighted L2, which the
             TRANSPORT needed; the generator has a learned positional embedding
             per context slot instead.

Which one a checkpoint wants is recorded IN the checkpoint, so callers should
build this from the checkpoint rather than deciding for themselves.
"""
from __future__ import annotations

import numpy as np
import torch

from aag.generator import PixelContextEncoder


class ContextWindow:
    def __init__(self, ckpt: dict, ae, dev, ctx_frames: int, ctx_dim: int,
                 image_size: int = 64, gamma: float = 0.95):
        self.dev, self.CTX, self.DIM = dev, ctx_frames, ctx_dim
        self.image_size = image_size
        self.recw = torch.tensor(np.sqrt(gamma ** np.arange(ctx_frames - 1, -1, -1)),
                                 dtype=torch.float32, device=dev).view(1, ctx_frames, 1)
        self.ae = ae
        self.pixel_ctx = bool(ckpt.get("pixel_context"))
        self.ft_ae = bool(ckpt.get("finetune_ae_enc"))
        self.mode = "pixel" if (self.pixel_ctx or self.ft_ae) else "ae_latent"
        self.enc = None
        if self.pixel_ctx:
            self.enc = PixelContextEncoder(ctx_dim=ctx_dim,
                                           ch=int(ckpt.get("pix_ch", 64)),
                                           image_size=image_size).to(dev).eval()
            sd = ckpt.get("enc_pix_state_dict")
            if sd is None:
                raise SystemExit("checkpoint says pixel_context but carries no "
                                 "enc_pix_state_dict -- the encoder it was "
                                 "trained with is not recoverable")
            self.enc.load_state_dict(sd)
        elif self.ft_ae:
            # the AE's encoder, but the FINE-TUNED weights, not the ones on disk
            self.enc = ae.enc
            sd = ckpt.get("enc_pix_state_dict")
            if sd is not None:
                self.enc.load_state_dict(sd)
            else:
                print("  WARNING checkpoint says finetune_ae_enc but carries no "
                      "encoder state; falling back to the AE's ORIGINAL encoder, "
                      "which is not what it trained with", flush=True)
            self.enc.eval()
        self._w = None

    def describe(self) -> str:
        if self.mode == "ae_latent":
            return "context: frozen AE latents, recency-scaled (AE in the loop)"
        which = "fresh PixelContextEncoder" if self.pixel_ctx else "fine-tuned AE encoder"
        return f"context: pixel window through the {which} (no AE in the loop)"

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B,T,3,H,W) -> (B, T*DIM)."""
        B, T = frames.shape[:2]
        if self.pixel_ctx:
            return self.enc(frames).reshape(B, -1)
        out = self.enc(frames.reshape(B * T, *frames.shape[2:]))
        return out.reshape(B, -1)

    def init(self, cond_row: torch.Tensor | None, frames: torch.Tensor | None):
        """cond_row (B, CTX*DIM) for ae_latent; frames (B,CTX,3,H,W) for pixel."""
        if self.mode == "ae_latent":
            if cond_row is None:
                raise ValueError("ae_latent context needs cond_row")
            B = cond_row.shape[0]
            self._w = cond_row.view(B, self.CTX, self.DIM) / self.recw
        else:
            if frames is None:
                raise ValueError("pixel context needs the real context frames")
            self._w = frames.clone()

    def vector(self) -> torch.Tensor:
        if self.mode == "ae_latent":
            B = self._w.shape[0]
            return (self._w * self.recw).reshape(B, -1)
        return self.encode(self._w)

    def push(self, pred: torch.Tensor):
        """Slide one generated frame (B,3,H,W) in [-1,1] into the window."""
        p = pred.clamp(-1, 1).float()
        if self.mode == "ae_latent":
            # .float() matters: under autocast pred is bf16 and the AE's conv
            # weights are fp32
            hn = self.ae.enc(p).view(p.shape[0], 1, self.DIM).float()
            self._w = torch.cat([self._w[:, 1:], hn], 1)
        else:
            self._w = torch.cat([self._w[:, 1:], p.unsqueeze(1)], 1)
