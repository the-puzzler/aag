"""The VPT action representation: 10 binary controls + 2 continuous mouse axes.

This is the representation the interactive model is conditioned on, and it is
deliberately the SAME 11 controls a player actually has:

    idx  0..3   W A S D          movement
    idx  4      space            jump
    idx  5      left shift       sneak
    idx  6      left ctrl        sprint
    idx  7      E                inventory  (opens a GUI, so it changes the screen)
    idx  8      attack           left mouse button
    idx  9      use              right mouse button
    idx 10..11  dx, dy           mouse motion == where the player is looking

Why this replaces the 81-way categorical
----------------------------------------
The old condition was `action = move(9)*9 + turn(3)*3 + tilt(3)`.  That index is
a deterministic FUNCTION of the vector above, so it carries no information the
vector lacks, while costing three things that were all measured:

  * it cannot express attack, use or E at all -- there is no code point for them,
    so a model conditioned on it can never respond to a mouse click;
  * it collapses mouse magnitude, folding a 5 px nudge and a 500 px flick into
    one class, past a 5 px deadzone;
  * as an integer it puts opposites at adjacent code points ("left" next to
    "look down"), which is what made a turn command read as a tilt.

The 81-way index is still kept in the particle files as `action`, because the
exact-class transport groupings need it.  It is no longer generator input.

Two uses, two scalings -- this is the part that is easy to get wrong
-------------------------------------------------------------------
`action_raw` is the physical vector: binaries in {0,1}, dx/dy in raw pixels.
Every grouping and every diagnostic derives from it, so it is the thing to store.

`action_vec` is `action_raw` mapped for use as a DISTANCE (the k-NN
neighbourhoods of the conditional transport).  Two rules:

  * Binaries stay at a bounded 0/1.  They are NOT standardised.  Standardising a
    rare binary divides by sqrt(p(1-p)), so E at a 0.41% press rate becomes a
    +/-15.6 swing and single-handedly decides every neighbourhood it appears in;
    the original builder dropped E rather than fix this.  At 0/1, differing in E
    costs exactly as much as differing in W -- per-EVENT parity, which is the
    right notion for a switch, rather than per-column variance parity.
  * Mouse is signed-log1p (so magnitude survives, unlike the deadzone class),
    standardised, then scaled so the two mouse columns TOGETHER hold MOUSE_W
    times the total variance of the ten binaries.  MOUSE_W=8 is measured, not
    picked: mouse magnitude explains 16.4% of frame-to-frame pixel variance
    against 2.1% for all keys combined.  Raw and unweighted, dx alone is 81% of
    the L2 variance and a nearest-k on actions is really a nearest-k on dx.

The normalisation constants are fitted once on the particle set and MUST be
saved with it: at inference the live controller has to be encoded with the same
constants or the generator sees a differently-scaled condition than it trained
on.  `fit_action_norm` / `apply_action_norm` / `encode_live` are that contract.
"""
from __future__ import annotations

import numpy as np

# column layout of action_raw / action_vec
ACTION_NAMES = ["W", "A", "S", "D", "space", "shift", "ctrl", "E",
                "attack", "use", "dx", "dy"]
BINARY_COLS = list(range(10))
MOUSE_COLS = [10, 11]
A_DIM = 12

# keys.npy column order, from prepare_vpt_cache.KEYS
KEYS_ORDER = ["W", "A", "S", "D", "space", "shift", "ctrl", "E"]

MOUSE_W = 8.0        # mouse:all-keys variance ratio, from the pixel-change fit
DX_DEADZONE = 5.0    # matches prepare_vpt_cache, so `turn`/`tilt` stay reproducible
DY_DEADZONE = 5.0


def build_action_raw(keys, mouse, clicks) -> np.ndarray:
    """(N,8) keys + (N,2) mouse + (N,2) clicks -> (N,12) float32 physical vector.

    keys columns are KEYS_ORDER (W A S D space shift ctrl E); clicks are
    [attack, use] as written by patch_vpt_clicks.py.
    """
    keys = np.asarray(keys)
    mouse = np.asarray(mouse)
    clicks = np.asarray(clicks)
    if keys.shape[-1] != 8:
        raise ValueError(f"expected 8 key columns, got {keys.shape[-1]}")
    if clicks.shape[-1] != 2:
        raise ValueError(f"expected 2 click columns, got {clicks.shape[-1]}")
    out = np.empty(keys.shape[:-1] + (A_DIM,), np.float32)
    out[..., 0:8] = keys.astype(np.float32)
    out[..., 8:10] = clicks.astype(np.float32)
    out[..., 10:12] = mouse.astype(np.float32)
    return out


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def fit_action_norm(raw: np.ndarray, mouse_w: float = MOUSE_W) -> dict:
    """Fit the mouse standardisation + the mouse:key variance balance.

    Returned dict must be saved alongside the particles and carried into the
    generator checkpoint -- see encode_live.
    """
    raw = np.asarray(raw, np.float32).reshape(-1, A_DIM)
    m = _signed_log1p(raw[:, MOUSE_COLS])
    mu = m.mean(0)
    sd = m.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    p = raw[:, BINARY_COLS].mean(0)
    key_var_total = float((p * (1.0 - p)).sum())
    # two mouse columns are to hold mouse_w * key_var_total between them
    scale = float(np.sqrt(max(mouse_w * key_var_total, 1e-8) / len(MOUSE_COLS)))
    return {"mouse_mean": mu.astype(np.float32).tolist(),
            "mouse_sd": sd.astype(np.float32).tolist(),
            "key_rates": p.astype(np.float32).tolist(),
            "key_var_total": key_var_total,
            "mouse_scale": scale,
            "mouse_w": float(mouse_w),
            "names": list(ACTION_NAMES)}


def apply_action_norm(raw: np.ndarray, norm: dict) -> np.ndarray:
    """(N,12) raw -> (N,12) action_vec, using SAVED constants."""
    raw = np.asarray(raw, np.float32)
    flat = raw.reshape(-1, A_DIM)
    v = np.empty_like(flat)
    v[:, BINARY_COLS] = flat[:, BINARY_COLS]          # bounded 0/1, not standardised
    m = _signed_log1p(flat[:, MOUSE_COLS])
    m = (m - np.asarray(norm["mouse_mean"], np.float32)) / np.asarray(norm["mouse_sd"], np.float32)
    v[:, MOUSE_COLS] = m * np.float32(norm["mouse_scale"])
    return v.reshape(raw.shape)


def encode_live(pressed, dx: float, dy: float, norm: dict) -> np.ndarray:
    """Encode one live controller state for interactive inference.

    `pressed` is any container of names from ACTION_NAMES[:10] (case-insensitive),
    e.g. {"W", "shift", "attack"}.  Returns a (12,) float32 action_vec built with
    the same constants the generator trained on.
    """
    want = {str(s).lower() for s in pressed}
    raw = np.zeros((1, A_DIM), np.float32)
    for i in BINARY_COLS:
        if ACTION_NAMES[i].lower() in want:
            raw[0, i] = 1.0
    raw[0, 10] = float(dx)
    raw[0, 11] = float(dy)
    return apply_action_norm(raw, norm)[0]


def action_index_from_raw(raw: np.ndarray) -> np.ndarray:
    """Reproduce prepare_vpt_cache's 81-way index from action_raw.

    Kept so the categorical the exact-class transport groups on can be rebuilt
    (and checked against the cached action_seqs) rather than trusted blindly.
    """
    raw = np.asarray(raw, np.float32).reshape(-1, A_DIM)
    w, a, s, d = (raw[:, i] > 0.5 for i in range(4))
    fb = w.astype(np.int64) - s.astype(np.int64)
    lr = d.astype(np.int64) - a.astype(np.int64)
    lut = {(0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, -1): 3, (0, 1): 4,
           (1, -1): 5, (1, 1): 6, (-1, -1): 7, (-1, 1): 8}
    move = np.array([lut[(int(f), int(l))] for f, l in zip(fb, lr)], np.int64)
    dx, dy = raw[:, 10], raw[:, 11]
    turn = np.where(np.abs(dx) < DX_DEADZONE, 0, np.where(dx > 0, 1, 2))
    tilt = np.where(np.abs(dy) < DY_DEADZONE, 0, np.where(dy > 0, 1, 2))
    return move * 9 + turn * 3 + tilt


def action_marginals(raw: np.ndarray, mag_quantiles=(1 / 3, 2 / 3)) -> dict:
    """Every low-dimensional function of the action the generator could key on.

    The project's hard-won lesson is that decorrelating z from the action vector
    AS A WHOLE leaves it correlated with coarse marginals of that vector -- the
    joint 81-way class read 0.593 while "turning vs not" read 1.272.  So the
    marginals are enumerated explicitly here rather than discovered later, and
    the conditional transport interleaves a firing against each.

    Returns {name: (N,) int64 group ids}: one per binary control, the signed
    mouse direction per axis (3 groups each, using the cache's deadzones), and
    magnitude terciles per axis among the frames that clear the deadzone.
    """
    raw = np.asarray(raw, np.float32).reshape(-1, A_DIM)
    out = {}
    for i in BINARY_COLS:
        out[ACTION_NAMES[i].lower()] = (raw[:, i] > 0.5).astype(np.int64)
    for axis, dz, nm in ((10, DX_DEADZONE, "dx"), (11, DY_DEADZONE, "dy")):
        v = raw[:, axis]
        av = np.abs(v)
        out[f"{nm}sign"] = np.where(av < dz, 0, np.where(v > 0, 1, 2)).astype(np.int64)
        live = av >= dz
        g = np.zeros(len(v), np.int64)              # 0 = inside the deadzone
        if live.any():
            qs = np.quantile(av[live], mag_quantiles)
            g[live] = 1 + np.searchsorted(qs, av[live], side="right")
        out[f"{nm}mag"] = g
    # any-click and any-key: the coarsest marginals, and coarse marginals were
    # the ones that leaked worst on the action side
    out["anyclick"] = ((raw[:, 8] > 0.5) | (raw[:, 9] > 0.5)).astype(np.int64)
    out["moving"] = (raw[:, 0:4].max(1) > 0.5).astype(np.int64)
    return out
