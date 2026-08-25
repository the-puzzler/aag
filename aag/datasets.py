"""Dataset dispatch so one pipeline serves CelebA and CIFAR-10.

Both loaders expose the SAME contract: a deterministic "particle" subset whose
order defines the persistent x_i <-> z_i identity. Every downstream tensor
(latents h, assignment z, conditions) must be built in that order or the pairs
silently misalign -- a bug that once looked exactly like mode collapse.

Conditions differ in kind:
  celeba  -> 40 binary attributes, groups only approximable by k-NN in Hamming space
  cifar10 -> 10 exact, disjoint class groups
"""
from __future__ import annotations

import torch

# The 18 discrete Doom actions, reconstructed by running the GameNGen-repro
# action-space builder (gameNgen-repro/ViZDoomPPO/common/utils.py) with the
# scenario's button order [ATTACK, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT,
# TURN_RIGHT, TURN_LEFT]: all button combinations, minus mutually-exclusive
# pairs, ATTACK usable only alone, no-op removed.
DOOM_ACTION_NAMES = [
    "Turn Left", "Turn Right", "Right", "Right+Turn Left", "Right+Turn Right",
    "Left", "Left+Turn Left", "Left+Turn Right", "Forward", "Forward+Turn Left",
    "Forward+Turn Right", "Forward+Right", "Forward+Right+Turn Left",
    "Forward+Right+Turn Right", "Forward+Left", "Forward+Left+Turn Left",
    "Forward+Left+Turn Right", "Attack",
]

SPECS = {
    "celeba":  dict(image_size=64, n_cond=40, discrete=False, default_root="/data/hf_cache"),
    "cifar10": dict(image_size=32, n_cond=10, discrete=True,  default_root="data"),
    # doom: one 16-frame segment per DISTINCT episode, so N == independent
    # samples (unlike ucf101, whose segments are ~9 near-duplicates per clip).
    # Conditions are the 18 agent actions (modal action over the segment).
    "doom":    dict(image_size=64, n_cond=18, discrete=True, frames=16, video=True,
                    default_root="/data/doom/cache_train"),
    # doom_frames: the SAME cache read one frame at a time, for the per-frame 2D
    # autoencoder a world model wants (temporal modelling belongs in the
    # dynamics, not the encoder). ~3M frames, so no sample bound at all.
    "doom_frames": dict(image_size=64, n_cond=18, discrete=True, video=False,
                        default_root="/data/doom/cache_train"),
}


def spec(dataset: str) -> dict:
    if dataset not in SPECS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(SPECS)}")
    return SPECS[dataset]


class ShardedSegments:
    """Read-only view over segments_00000.npy, segments_00001.npy, ... as one array.

    The VPT builder writes shards rather than a single file: a lone 770 GB mapping
    accumulates dirty pages for the whole run and left this box hard-hanging twice,
    whereas one live shard bounds the dirty set. Reading is unaffected either way
    (clean mapped pages are freely reclaimable), so this exists purely so the
    loaders below do not care which layout they were given.

    Shards are mmapped lazily on first touch and cached, so a loader that only
    visits part of the dataset never maps the rest.
    """

    def __init__(self, root: str, shard_size: int):
        import glob
        import numpy as np
        self._paths = sorted(glob.glob(f"{root}/segments_[0-9]*.npy"))
        if not self._paths:
            raise FileNotFoundError(f"no segment shards under {root}")
        self._shard = int(shard_size)
        self._open: dict = {}
        last = np.load(self._paths[-1], mmap_mode="r")
        self._tail = last.shape
        self.shape = (self._shard * (len(self._paths) - 1) + last.shape[0],) + last.shape[1:]
        self.dtype = last.dtype

    def _get(self, si: int):
        a = self._open.get(si)
        if a is None:
            import numpy as np
            a = np.load(self._paths[si], mmap_mode="r")
            self._open[si] = a
        return a

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        # only the access patterns the loaders use: seg[j] and seg[c, f]
        if isinstance(key, tuple):
            j, rest = key[0], key[1:]
            si, so = divmod(int(j), self._shard)
            return self._get(si)[(so,) + rest]
        si, so = divmod(int(key), self._shard)
        return self._get(si)[so]


def open_segments(root: str):
    """Single-file cache if present, else the sharded layout."""
    import json
    import os
    import numpy as np
    single = f"{root}/segments.npy"
    if os.path.exists(single):
        return np.load(single, mmap_mode="r")
    shard = 8192
    try:
        shard = int(json.load(open(f"{root}/meta.json")).get("shard_size", shard))
    except Exception:
        pass
    return ShardedSegments(root, shard)


class VideoSegments(torch.utils.data.Dataset):
    """uint8 segment cache -> (3,T,H,W) float in [-1,1], plus class label.

    The cache is memory-mapped and converted per item: holding all ~88k
    segments as fp32 would be ~70GB, versus 17GB as uint8 on disk.
    Item order IS particle order.
    """

    def __init__(self, root: str, indices=None):
        import numpy as np
        self.segs = open_segments(root)                              # (N,T,H,W,3)
        self.labels = np.load(f"{root}/labels.npy")
        self.idx = np.arange(len(self.labels)) if indices is None else indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = torch.from_numpy(self.segs[j].copy())          # (T,H,W,3) uint8
        x = x.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)   # (3,T,H,W) in [-1,1]
        return x, int(self.labels[j])


class DoomFrames(torch.utils.data.Dataset):
    """One frame at a time out of the chunked uint8 cache.

    `stride` skips near-duplicate neighbours (adjacent frames are ~identical),
    so the default keeps 4 of every 16 frames -- still ~760k frames, at a
    quarter the epoch cost. Returns the frame's own action as the label.
    """

    def __init__(self, root: str, stride: int = 4, indices=None):
        import numpy as np
        self.segs = open_segments(root)                              # (N,T,H,W,3)
        self.acts = np.load(f"{root}/action_seqs.npy")               # (N,T)
        n_valid = len(np.load(f"{root}/labels.npy"))                 # capacity may exceed this
        self.f_idx = np.arange(0, self.segs.shape[1], stride)
        self.n_per = len(self.f_idx)
        flat = np.arange(n_valid * self.n_per)
        self.idx = flat if indices is None else indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        c, f = j // self.n_per, self.f_idx[j % self.n_per]
        x = torch.from_numpy(self.segs[c, f].copy())            # (H,W,3) uint8
        x = x.permute(2, 0, 1).float().div_(127.5).sub_(1.0)    # (3,H,W) in [-1,1]
        return x, int(self.acts[c, f])


def _doom_frame_loaders(root: str, batch: int, n_particles: int, workers: int = 4,
                        n_train: int | None = None, stride: int = 4):
    from torch.utils.data import DataLoader
    full = DoomFrames(root, stride=stride)
    n = len(full)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).numpy()
    p_idx = perm[:min(n_particles, n)]
    t_idx = perm[min(n_particles, n):][:4096]            # never overlaps particles
    particles = DoomFrames(root, stride, p_idx)
    test = DoomFrames(root, stride, t_idx if len(t_idx) else p_idx[:1024])
    return (DataLoader(full, batch, shuffle=True, num_workers=workers, pin_memory=True),
            DataLoader(particles, batch, shuffle=False, num_workers=workers),
            DataLoader(test, batch, shuffle=False, num_workers=workers), n)


def _segment_loaders(root: str, batch: int, n_particles: int, workers: int = 4,
                 n_train: int | None = None):
    from torch.utils.data import DataLoader
    full = VideoSegments(root)
    n = len(full)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    p_idx = perm[:min(n_particles, n)].numpy()
    # held-out tail of the same permutation, so it never overlaps the particles
    t_idx = perm[min(n_particles, n):][:4096].numpy()
    particles = VideoSegments(root, p_idx)
    test = VideoSegments(root, t_idx if len(t_idx) else p_idx[:1024])
    ae_loader = DataLoader(full, batch, shuffle=True, num_workers=workers, pin_memory=True)
    enc_loader = DataLoader(particles, batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test, batch, shuffle=False, num_workers=workers)
    return ae_loader, enc_loader, test_loader, n


def get_loaders(dataset: str, root: str, batch: int, n_particles: int,
                workers: int = 4, n_train: int | None = None,
                image_size: int | None = None):
    """-> (ae_loader, enc_loader, test_loader, n_train_available)"""
    if dataset == "celeba":
        from .celeba_data import celeba_loaders
        return celeba_loaders(root, batch, n_particles, workers, n_train,
                              image_size or SPECS["celeba"]["image_size"])
    if dataset == "cifar10":
        from .data import cifar_loaders
        return cifar_loaders(root, batch, n_particles, min(workers, 2), n_train)
    if dataset == "doom_frames":
        return _doom_frame_loaders(root or SPECS["doom_frames"]["default_root"], batch,
                                   n_particles, workers, n_train)
    if dataset == "doom":
        return _segment_loaders(root or SPECS["doom"]["default_root"], batch,
                            n_particles, workers, n_train)
    raise ValueError(dataset)


def collect_particle_images(dataset: str, root: str, batch: int, n_particles: int,
                            image_size: int | None = None, device="cpu"):
    """Target images in particle order -- the order z is indexed by."""
    _, enc, _, _ = get_loaders(dataset, root, batch, n_particles, image_size=image_size)
    return torch.cat([x for x, _ in enc]).to(device)


def collect_conditions(dataset: str, root: str, batch: int, n_particles: int,
                       device="cpu"):
    """-> (cond_float, group_ids_or_None) in particle order.

    cond_float feeds the generator: (N, n_cond) float, one-hot for cifar10.
    group_ids is (N,) int class labels for discrete datasets, else None -- it is
    what group_rank_transport_step and the group diagnostics consume.
    """
    if dataset == "cifar10":
        _, enc, _, _ = get_loaders(dataset, root, batch, n_particles)
        labels = torch.cat([y for _, y in enc]).to(device)
        onehot = torch.zeros(labels.shape[0], SPECS["cifar10"]["n_cond"], device=device)
        onehot[torch.arange(labels.shape[0], device=device), labels] = 1.0
        return onehot, labels
    if dataset == "doom_frames":
        _, enc, _, _ = get_loaders(dataset, root, batch, n_particles)
        labels = torch.cat([y for _, y in enc]).to(device)
        onehot = torch.zeros(labels.shape[0], SPECS["doom_frames"]["n_cond"], device=device)
        onehot[torch.arange(labels.shape[0], device=device), labels] = 1.0
        return onehot, labels
    if dataset == "doom":
        _, enc, _, _ = get_loaders(dataset, root, batch, n_particles)
        labels = torch.cat([y for _, y in enc]).to(device)
        onehot = torch.zeros(labels.shape[0], SPECS["doom"]["n_cond"], device=device)
        onehot[torch.arange(labels.shape[0], device=device), labels] = 1.0
        return onehot, labels
    if dataset == "celeba":
        a = torch.load("results_celeba/attrs.pt", map_location=device, weights_only=False)
        return a["attrs"].float().to(device), None
    raise ValueError(dataset)
