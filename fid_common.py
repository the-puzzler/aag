"""Shared FID plumbing: Inception-v3 (standard FID weights, via pytorch-fid)
feature extraction + Frechet distance, operating on images in [0,1] float."""
from __future__ import annotations

import numpy as np
import torch
from scipy import linalg
from pytorch_fid.inception import InceptionV3

_MODEL = None


def get_inception(device):
    global _MODEL
    if _MODEL is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        _MODEL = InceptionV3([block_idx]).to(device).eval()
    return _MODEL


@torch.no_grad()
def get_activations(imgs01, device, batch=200):
    """imgs01: (N,3,H,W) float tensor in [0,1] (any H,W; Inception resizes)."""
    model = get_inception(device)
    acts = []
    for i in range(0, imgs01.shape[0], batch):
        b = imgs01[i:i + batch].to(device)
        out = model(b)[0]
        out = out.squeeze(-1).squeeze(-1).cpu().numpy()
        acts.append(out)
    return np.concatenate(acts, 0)


def activation_stats(acts):
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)



def fid_from_acts(acts_a, acts_b):
    mu_a, sigma_a = activation_stats(acts_a)
    mu_b, sigma_b = activation_stats(acts_b)
    return calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


def fid_from_stats(mu_a, sigma_a, acts_b):
    mu_b, sigma_b = activation_stats(acts_b)
    return calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
