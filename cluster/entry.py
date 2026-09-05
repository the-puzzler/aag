#!/usr/bin/env python
"""Stage runner launched by odytrain's torchrun mode, one process per GPU.

torchrun starts NUM_GPUS copies of this file with RANK / LOCAL_RANK / WORLD_SIZE /
MASTER_ADDR / MASTER_PORT in the environment. The YAML at /etc/config/config.yaml
lists stages; each stage is a shell command run from /app/aag:

  stages:
    - name: download            # single: LOCAL_RANK 0 runs it, the others wait
      cmd: python scripts/...
    - name: encode
      ddp: true                 # every rank runs it with the torchrun env intact
      cmd: python scripts/encode_hf256.py --rank $RANK --world $WORLD_SIZE ...
    - name: assign_and_train    # parallel group: GPUs split between sub-stages
      parallel:
        - {name: assign,  gpus: "0",   cmd: python scripts/run_assignment_classes.py ...}
        - {name: gen,     gpus: "1-7", ddp: true, cmd: torchrun-style DDP command}

In a parallel group each rank runs the sub-stage whose `gpus` contains its
LOCAL_RANK; for a `ddp` sub-stage the child sees RANK / WORLD_SIZE renumbered
within its GPU set, its own MASTER_PORT, and CUDA_VISIBLE_DEVICES restricted to
that set (LOCAL_RANK renumbered to match), so plain "cuda" in a one-GPU script
means the group's first GPU, never physical GPU 0. This is how one
node stays busy while a single-GPU assignment runs for hours (the user's
rule: more transport is better, so it is never short).

Barriers are files under /dev/shm (single node) because a torch process group
here would occupy MASTER_PORT that the DDP children need. Completed single
stages leave a marker in `state_dir` so a relaunched job with the same config
name skips them -- delete the marker to force a rerun.
"""
from __future__ import annotations

import os, subprocess, sys, time
from pathlib import Path

import yaml

cfg = yaml.safe_load(open(os.environ.get("AAG_CONFIG", "/etc/config/config.yaml")))
local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
world = int(os.environ.get("WORLD_SIZE", 1))
state = Path(cfg.get("state_dir", "/mnt/shared/aag/state")) / cfg["name"]
shm = Path("/dev/shm") / f"aag_{cfg['name']}"
if local == 0:
    state.mkdir(parents=True, exist_ok=True); shm.mkdir(parents=True, exist_ok=True)
base_env = dict(os.environ, **{k: str(v) for k, v in cfg.get("env", {}).items()})


def log(msg, rank0_only=True):
    if local == 0 or not rank0_only:
        print(f"[entry {time.strftime('%H:%M:%S')} gpu{local}] {msg}", flush=True)


def parse_gpus(spec) -> list[int]:
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a_, b_ = part.split("-"); out += list(range(int(a_), int(b_) + 1))
        else:
            out.append(int(part))
    return out


def run(cmd, tag, env):
    log(f"stage {tag}: {cmd}", rank0_only=False)
    t = time.time()
    r = subprocess.run(cmd, shell=True, env=env, cwd="/app/aag")
    if r.returncode:
        print(f"[entry gpu{local}] stage {tag} FAILED rc={r.returncode}", flush=True)
        (shm / f"{tag}.failed").touch()
        sys.exit(r.returncode)
    log(f"stage {tag} done in {(time.time() - t) / 60:.1f} min", rank0_only=False)


def wait_for(marker, tag):
    while not (shm / marker).exists():
        if (shm / f"{tag}.failed").exists() or any(shm.glob("*.failed")):
            sys.exit(1)
        time.sleep(5)


def single(st, tag):
    """Only LOCAL_RANK 0 runs; the rest block on the marker."""
    done = state / f"{tag}.done"
    if local == 0:
        if done.exists() and not st.get("always", False):
            log(f"stage {tag}: already done ({done}), skipping")
        else:
            run(st["cmd"], tag, base_env); done.touch()
        (shm / f"{tag}.done").touch()
    else:
        wait_for(f"{tag}.done", tag)


for i, st in enumerate(cfg["stages"]):
    tag = f"{i:02d}_{st['name']}"
    if "parallel" in st:
        subs = st["parallel"]
        mine = [s for s in subs if local in parse_gpus(s["gpus"])]
        if len(mine) != 1:
            raise SystemExit(f"gpu{local}: parallel stage {tag} must assign each GPU to exactly one sub-stage, got {len(mine)}")
        s = mine[0]; gpus = parse_gpus(s["gpus"]); stag = f"{tag}/{s['name']}"
        # The sub-group sees only its own GPUs, renumbered from 0: a single-GPU script's
        # plain "cuda" then lands on the group's first GPU instead of physical GPU 0
        # (which another sub-stage owns), and torch.cuda.set_device(LOCAL_RANK) stays valid.
        env = dict(base_env, RANK=str(gpus.index(local)), LOCAL_RANK=str(gpus.index(local)),
                   WORLD_SIZE=str(len(gpus)), CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus),
                   MASTER_PORT=str(int(base_env.get("MASTER_PORT", 29500)) + 1 + subs.index(s)))
        if s.get("ddp", False) or gpus.index(local) == 0:
            done = state / f"{stag.replace('/', '_')}.done"
            if done.exists() and not s.get("always", False):
                log(f"stage {stag}: already done, skipping", rank0_only=False)
            else:
                run(s["cmd"], stag, env)
                if gpus.index(local) == 0:
                    done.touch()
        (shm / f"{stag.replace('/', '_')}.gpu{local}.done").touch()
        # everyone waits for every GPU of every sub-stage before the next stage
        for s2 in subs:
            for g in parse_gpus(s2["gpus"]):
                wait_for(f"{tag}/{s2['name']}".replace("/", "_") + f".gpu{g}.done", tag)
    elif st.get("ddp", False):
        run(st["cmd"], tag, base_env)
    else:
        single(st, tag)
log("all stages complete")
