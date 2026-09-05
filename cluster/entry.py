#!/usr/bin/env python
"""Stage runner launched by odytrain's torchrun mode, one process per GPU.

torchrun starts NUM_GPUS copies of this file with RANK / LOCAL_RANK / WORLD_SIZE /
MASTER_ADDR / MASTER_PORT in the environment. Each stage in the YAML at
/etc/config/config.yaml is a shell command:

  stages:
    - name: download      # single: only LOCAL_RANK 0 runs it, the others wait
      cmd: python scripts/...
    - name: generator
      ddp: true           # every rank runs it; the command inherits torchrun's env
      cmd: python scripts/train_generator256_ddp.py ...

The wait is a file barrier under /dev/shm (single node), because a torch
process group in this parent would occupy MASTER_PORT that the DDP child needs.
Completed single stages leave a marker in `state_dir` so a relaunched job with
the same config skips them -- resume by deleting the marker.
"""
from __future__ import annotations

import os, subprocess, sys, time
from pathlib import Path

import yaml

cfg = yaml.safe_load(open(os.environ.get("AAG_CONFIG", "/etc/config/config.yaml")))
rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
world = int(os.environ.get("WORLD_SIZE", 1))
state = Path(cfg.get("state_dir", "/mnt/shared/aag/state")) / cfg["name"]
if rank == 0:
    state.mkdir(parents=True, exist_ok=True)
shm = Path("/dev/shm") / f"aag_{cfg['name']}"
if rank == 0:
    shm.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, **{k: str(v) for k, v in cfg.get("env", {}).items()})


def log(msg):
    if rank == 0:
        print(f"[entry {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, tag):
    log(f"stage {tag}: {cmd}")
    t = time.time()
    r = subprocess.run(cmd, shell=True, env=env, cwd="/app/aag")
    if r.returncode:
        print(f"[entry rank {rank}] stage {tag} FAILED rc={r.returncode}", flush=True)
        (shm / f"{tag}.failed").touch()
        sys.exit(r.returncode)
    log(f"stage {tag} done in {(time.time() - t) / 60:.1f} min")


for i, st in enumerate(cfg["stages"]):
    tag = f"{i:02d}_{st['name']}"
    if st.get("ddp", False):
        run(st["cmd"], tag)
        continue
    done = state / f"{tag}.done"
    if rank == 0:
        if done.exists() and not st.get("always", False):
            log(f"stage {tag}: already done ({done}), skipping")
        else:
            run(st["cmd"], tag)
            done.touch()
        (shm / f"{tag}.done").touch()
    else:
        while not (shm / f"{tag}.done").exists():
            if (shm / f"{tag}.failed").exists():
                sys.exit(1)
            time.sleep(5)
log("all stages complete")
