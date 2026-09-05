#!/usr/bin/env python
"""Block until a file exists and has stopped growing (an upload in progress is not ready).

The assignment is produced on the local box and copied to /mnt/shared by whatever hop is
available (object store or kubectl cp), so the cluster job just waits for it here. Prints
once a minute so the pod log shows it is waiting, not hung; exits non-zero on timeout.
"""
import argparse, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--hours", type=float, default=6.0)
ap.add_argument("--settle", type=int, default=90, help="seconds the size must be unchanged")
a = ap.parse_args()
t0 = time.time(); last = (-1, 0.0)
while time.time() - t0 < a.hours * 3600:
    if os.path.exists(a.path):
        size = os.path.getsize(a.path)
        if size != last[0]:
            last = (size, time.time())
        elif time.time() - last[1] >= a.settle:
            print(f"ready: {a.path} ({size / 1e9:.2f} GB, stable {a.settle}s) after {(time.time() - t0) / 60:.1f} min", flush=True)
            sys.exit(0)
        print(f"  {a.path}: {size / 1e9:.2f} GB, settling...", flush=True)
    else:
        print(f"  waiting for {a.path} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    time.sleep(60)
print(f"timeout after {a.hours} h waiting for {a.path}", flush=True); sys.exit(1)
