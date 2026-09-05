#!/usr/bin/env python
"""Did the REPRESENTATION change, not just its information content?

`diag_latent_probe.py` answers "how much is linearly decodable from z", and that
is a statement about CONTENT. It is not a statement about the representation. An
encoder can reorganise the latent completely -- rotate it, rescale it, change
which directions carry what, change how gaussian the marginals are -- while the
linearly-decodable content stays identical. The probe is blind to all of that,
and all of it matters to AAG, because the assignment transports z and its cost
depends on the geometry it is transporting.

So this compares the two latent spaces directly, on the same frames:

  CKA(h_old, h_new)   representational similarity, invariant to rotation and
                      isotropic scaling. 1.0 means the same geometry up to those
                      symmetries; lower means genuinely reorganised.
  R^2 h_new <- h_old  can a LINEAR map take one space to the other? If yes, the
  R^2 h_old <- h_new  spaces are the same up to an affine change of basis, and
                      anything downstream that is itself affine-invariant is
                      unaffected. If no, the change is nonlinear and real.
  per-dim corr        |corr(h_old[i], h_new[i])| dimension by dimension. Low
                      values with high CKA means the axes moved but the space
                      did not -- which still invalidates cached latents.

And the properties AAG actually cares about, for each space independently:

  eff. rank           participation ratio of the covariance spectrum, i.e. how
                      many of the 256 dimensions are really used.
  mean |off-diag rho| how decorrelated the dimensions already are before the
                      assignment does anything.
  |kurtosis - 3|      how far the marginals are from gaussian, averaged over
                      dimensions. The assignment gaussianises, so a latent that
                      starts closer costs it less.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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


def linear_cka(A: torch.Tensor, B: torch.Tensor) -> float:
    """CKA with a linear kernel. Invariant to rotation and isotropic scaling."""
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    # ||A^T B||_F^2 / (||A^T A||_F ||B^T B||_F)
    num = (A.T @ B).pow(2).sum()
    da = (A.T @ A).pow(2).sum().sqrt()
    db = (B.T @ B).pow(2).sum().sqrt()
    return float(num / (da * db + 1e-12))


def lin_map_r2(src: torch.Tensor, dst: torch.Tensor, n_tr: int,
               lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)) -> float:
    """Held-out R^2 of a ridge map src -> dst. 1.0 = affinely equivalent."""
    Xtr, Xte = src[:n_tr], src[n_tr:]
    Ytr, Yte = dst[:n_tr], dst[n_tr:]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0, keepdim=True)
    Xtr = torch.cat([Xtr, torch.ones_like(Xtr[:, :1])], 1)
    Xte = torch.cat([Xte, torch.ones_like(Xte[:, :1])], 1)
    G, C = Xtr.T @ Xtr, Xtr.T @ (Ytr - ym)
    eye = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    ss_tot = ((Yte - ym) ** 2).sum()
    best = -1e9
    for lam in lams:
        W = torch.linalg.solve(G + lam * eye, C)
        best = max(best, float(1.0 - ((Yte - ym - Xte @ W) ** 2).sum() / ss_tot))
    return best


def geometry(H: torch.Tensor) -> dict:
    Hc = H - H.mean(0, keepdim=True)
    sd = Hc.std(0).clamp_min(1e-8)
    cov = (Hc.T @ Hc) / (len(Hc) - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    # participation ratio: (sum l)^2 / sum l^2, the standard "how many
    # dimensions are actually used" summary of a spectrum
    eff_rank = float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12))
    corr = cov / (sd[:, None] * sd[None, :])
    d = corr.shape[0]
    off = corr[~torch.eye(d, dtype=torch.bool, device=corr.device)].abs().mean()
    z = Hc / sd
    kurt = (z.pow(4).mean(0) - 3.0).abs().mean()
    return {"eff_rank": eff_rank, "off_diag": float(off),
            "kurt_dev": float(kurt), "mean_std": float(sd.mean())}


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

    Hs = {nm: [] for nm in encs}
    with torch.no_grad():
        for i in range(0, len(sel), 256):
            b = sel[i:i + 256]
            x = np.stack([np.asarray(segs[int(ci[p])][int(fi[p])]) for p in b])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2).float()
                 .div_(127.5).sub_(1.0).to(dev))
            for nm, m in encs.items():
                Hs[nm].append(m.enc(x).flatten(1).double())
    Ho, Hn = torch.cat(Hs["old"]), torch.cat(Hs["new"])
    n_tr = int(len(sel) * args.train_frac)
    print(f"\nlatents on {len(sel):,} gui-free frames, dim {Ho.shape[1]}")

    cka = linear_cka(Ho, Hn)
    r2_fwd = lin_map_r2(Ho, Hn, n_tr)
    r2_bwd = lin_map_r2(Hn, Ho, n_tr)
    oc = Ho - Ho.mean(0, keepdim=True)
    nc = Hn - Hn.mean(0, keepdim=True)
    per_dim = ((oc * nc).mean(0) /
               (oc.std(0) * nc.std(0) + 1e-12)).abs()

    print("\nHOW DIFFERENT ARE THE TWO SPACES")
    print(f"  linear CKA(old, new)        {cka:.4f}   1.0 = same up to rotation/scale")
    print(f"  R^2  new <- old (linear)    {r2_fwd:.4f}")
    print(f"  R^2  old <- new (linear)    {r2_bwd:.4f}")
    print(f"  per-dim |corr| mean         {float(per_dim.mean()):.4f}  "
          f"median {float(per_dim.median()):.4f}  "
          f"min {float(per_dim.min()):.4f}")
    print(f"  dims with |corr| < 0.5      {int((per_dim < 0.5).sum())} of {len(per_dim)}")

    print("\nGEOMETRY OF EACH SPACE (what the assignment has to transport)")
    go, gn = geometry(Ho), geometry(Hn)
    print(f"  {'':22s} {'old':>10s} {'new':>10s}")
    for k, lab in (("eff_rank", "effective rank"),
                   ("off_diag", "mean |off-diag rho|"),
                   ("kurt_dev", "mean |kurtosis-3|"),
                   ("mean_std", "mean per-dim std")):
        print(f"  {lab:22s} {go[k]:10.4f} {gn[k]:10.4f}")

    print()
    if cka > 0.98 and min(r2_fwd, r2_bwd) > 0.98:
        print("  SAME REPRESENTATION up to an affine change of basis. Cached")
        print("  latents are still invalid (the numbers moved), but the assignment")
        print("  would be transporting the same geometry.")
    elif cka > 0.9:
        print("  MOSTLY THE SAME SPACE, partially rotated. Re-encoding is required")
        print("  and the assignment must be re-run, but the transport problem it")
        print("  faces is similar in character.")
    else:
        print("  GENUINELY REORGANISED. This is a different representation, not a")
        print("  perturbed one -- the earlier 'unchanged latent' claim, which was")
        print("  about linearly-decodable CONTENT, does not cover this.")


if __name__ == "__main__":
    main()
