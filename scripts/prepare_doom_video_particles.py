#!/usr/bin/env python
"""Particles for first-frame-conditioned video generation.

    target  h = VideoAE(chunk)        (256,)  the 16-frame clip
    cond    c = FrameAE(frame 0)      (64,)   its own first frame

Two different autoencoders on purpose: the video AE carries the clip's motion
and content, while the condition must be computable at inference from a single
image the user supplies -- which is exactly what the per-frame AE does.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

from aag.ae import AutoEncoder, VideoAutoEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ae", required=True)
    ap.add_argument("--frame-ae", required=True)
    ap.add_argument("--cache", default="/data/doom/cache_train")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda"
    vk = torch.load(args.video_ae, map_location=dev, weights_only=False)
    vae = VideoAutoEncoder(vk["latent_dim"], ch=vk["channels"], image_size=vk["image_size"],
                           frames=vk.get("frames", 16), architecture="spatial",
                           t_out=vk.get("t_out", 4), width_mult=4).to(dev).eval()
    vae.load_state_dict(vk["model_state_dict"])

    fk = torch.load(args.frame_ae, map_location=dev, weights_only=False)
    fae = AutoEncoder(fk["latent_dim"], ch=fk["channels"], architecture=fk["architecture"],
                      image_size=fk["image_size"]).to(dev).eval()
    fae.load_state_dict(fk["model_state_dict"])
    print(f"video AE dim={vk['latent_dim']} (ep{vk.get('epochs')}, test_mse={vk.get('test_mse'):.5f})", flush=True)
    print(f"frame AE dim={fk['latent_dim']} (ep{fk.get('epochs')}, test_mse={fk.get('test_mse'):.5f})", flush=True)

    segs = np.load(f"{args.cache}/segments.npy", mmap_mode="r")
    rec = np.load(f"{args.cache}/clip_ids.npy")
    acts = np.load(f"{args.cache}/action_seqs.npy")
    N = len(rec)
    H = torch.empty(N, vk["latent_dim"])
    C = torch.empty(N, fk["latent_dim"])
    with torch.no_grad():
        for i in range(0, N, args.batch):
            sl = slice(i, min(i + args.batch, N))
            x = torch.from_numpy(np.ascontiguousarray(segs[sl])).to(dev)
            x = x.permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1.0)   # (b,3,T,H,W)
            H[sl] = vae.enc(x).cpu()
            C[sl] = fae.enc(x[:, :, 0]).cpu()                            # frame 0 only
            if i % (args.batch * 200) == 0:
                print(f"  {i:,}/{N:,}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"h_target": H, "cond_first_frame": C,
                "episode": torch.from_numpy(rec), "action_seqs": torch.from_numpy(acts),
                "video_ae": args.video_ae, "frame_ae": args.frame_ae}, args.out / "particles.pt")
    meta = dict(n=N, dim=vk["latent_dim"], cond_dim=fk["latent_dim"],
                n_episodes=int(len(np.unique(rec))))
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
