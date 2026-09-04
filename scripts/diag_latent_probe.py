#!/usr/bin/env python
"""Does the LATENT carry more? A decoder-free probe.

`diag_ae_hf_fidelity.py` correlates the RECONSTRUCTION's high-frequency band
against the real one, and that is encoder x decoder: a decoder painting
uncorrelated texture over an UNCHANGED encoder produces a falling correlation
just as surely as an encoder that lost information. So it can show that an
adversary bought nothing, but it cannot say which half is responsible, and it
must not be read as "the encoder got worse".

This probe removes the decoder. Ridge-regress the latent z (256 numbers) onto
the target, fit on a train split and score R^2 on a held-out split:

    z -> hf(x)     linearly decodable HIGH-FREQUENCY information
    z -> x         linearly decodable information overall

Higher held-out R^2 means more of that information is present in the latent. It
is a lower bound -- a linear probe cannot see nonlinearly-coded information --
but it is measured on both models identically, so the COMPARISON is fair even
where the absolute number is pessimistic.

This is the right instrument for AAG generally, because z is a transported copy
of h_target = AE_enc(frame): only what the ENCODER kept can reach z, and
`diag_ae_floor`'s reconstruction error is decoder-mediated and therefore cannot
compare two AEs whose decoders differ in character.

The ridge penalty is swept per model and the best held-out score is reported for
each, so neither model is handicapped by a penalty tuned for the other.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aag.ae import AutoEncoder
from aag.datasets import open_segments


def load_enc(path: Path, dev: str):
    c = torch.load(path, map_location=dev, weights_only=False)
    ae = AutoEncoder(c["latent_dim"], ch=c["channels"],
                     architecture=c["architecture"], image_size=c["image_size"],
                     grid=c.get("grid", 4)).to(dev).eval()
    sd = c["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    ae.load_state_dict(sd)
    return ae, c


def hf(x: torch.Tensor) -> torch.Tensor:
    return x - F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="nearest")


def ridge_r2(Z: torch.Tensor, Y: torch.Tensor, n_tr: int, lams) -> tuple:
    """Held-out R^2 of a ridge fit z -> Y, best over the penalty sweep."""
    Ztr, Zte = Z[:n_tr], Z[n_tr:]
    Ytr, Yte = Y[:n_tr], Y[n_tr:]
    # standardise inputs on the train split: the two encoders have different
    # latent scales and a shared penalty would otherwise not mean the same thing
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True).clamp_min(1e-6)
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    ym = Ytr.mean(0, keepdim=True)
    Ytr_c = Ytr - ym
    Ztr = torch.cat([Ztr, torch.ones_like(Ztr[:, :1])], 1)
    Zte = torch.cat([Zte, torch.ones_like(Zte[:, :1])], 1)
    G = Ztr.T @ Ztr
    C = Ztr.T @ Ytr_c
    eye = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    # SS_tot uses the TRAIN mean, so R^2 measures prediction, not the test
    # split's own centring
    ss_tot = ((Yte - ym) ** 2).sum()
    best = (-1e9, None)
    for lam in lams:
        W = torch.linalg.solve(G + lam * eye, C)
        pred = Zte @ W
        r2 = float(1.0 - ((Yte - ym - pred) ** 2).sum() / ss_tot)
        if r2 > best[0]:
            best = (r2, lam)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path,
                    default=Path("/data/aag_results/results_vpt/assign_12d_lag1/"
                                 "assignment.pt"))
    ap.add_argument("--old-ae", type=Path,
                    default=Path("/data/aag_results/results_vpt/"
                                 "ae_dcae_ch192_dim256_cont/checkpoints/"
                                 "ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt"))
    ap.add_argument("--new-ae", type=Path, required=True)
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--train-frac", type=float, default=0.75)
    args = ap.parse_args()

    dev = "cuda"
    encs = {}
    for nm, p in (("old", args.old_ae), ("new", args.new_ae)):
        m, c = load_enc(p, dev)
        encs[nm] = m
        print(f"{nm}: {p.name}  epochs={c.get('epochs')} "
              f"gan_weight={c.get('gan_weight')}")

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    segs = open_segments(args.cache)
    rng = np.random.default_rng(0)
    cand = rng.permutation(len(ci))
    gui = np.load(f"{args.cache}/gui.npy", mmap_mode="r")
    cand = cand[~np.asarray(gui[ci[cand], fi[cand]]).astype(bool)]
    sel = cand[:args.n]

    Zs = {nm: [] for nm in encs}
    Xs, Hs = [], []
    with torch.no_grad():
        for i in range(0, len(sel), 256):
            b = sel[i:i + 256]
            x = np.stack([np.asarray(segs[int(ci[p])][int(fi[p])]) for p in b])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2).float()
                 .div_(127.5).sub_(1.0).to(dev))
            Xs.append(x.flatten(1).double())
            Hs.append(hf(x).flatten(1).double())
            for nm, m in encs.items():
                Zs[nm].append(m.enc(x).flatten(1).double())
    X = torch.cat(Xs)
    H = torch.cat(Hs)
    n_tr = int(len(sel) * args.train_frac)
    lams = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]

    print(f"\nlinear probe from the latent, fit on {n_tr:,} gui-free frames, "
          f"R^2 on {len(sel)-n_tr:,} held out")
    print(f"  {'model':6s} {'R2 z->hf(x)':>13s} {'R2 z->x':>10s}")
    out = {}
    for nm in ("old", "new"):
        Z = torch.cat(Zs[nm])
        r2h, _ = ridge_r2(Z, H, n_tr, lams)
        r2x, _ = ridge_r2(Z, X, n_tr, lams)
        out[nm] = (r2h, r2x)
        print(f"  {nm:6s} {r2h:13.4f} {r2x:10.4f}")
    dh = out["new"][0] - out["old"][0]
    dx = out["new"][1] - out["old"][1]
    print(f"\n  new-old  hf {dh:+.4f}   full {dx:+.4f}")
    print()
    if dh > 0.005:
        print("  THE LATENT CARRIES MORE high-frequency information. Any loss of")
        print("  reconstruction fidelity is the decoder's doing, and the encoder")
        print("  gain is real and available to the assignment.")
    elif dh < -0.005:
        print("  THE LATENT CARRIES LESS. The adversary degraded the ENCODER, not")
        print("  just the rendering -- the worst case, since z can only carry what")
        print("  the encoder kept.")
    else:
        print("  THE LATENT IS UNCHANGED within the probe's resolution. The")
        print("  encoder neither gained nor lost; the reconstruction differences")
        print("  are the DECODER's texture prior alone, and the assignment would")
        print("  see the same information it sees today.")


if __name__ == "__main__":
    main()
