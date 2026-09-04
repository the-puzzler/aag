#!/usr/bin/env python
"""Turn train_ae.py's cumulative running means into per-window values.

The log prints running_X / n, a mean over every step so far, which lags hard: a
term that has gone bad 20k steps ago still looks mild because the good early
steps are still in the average. Multiplying back by n and differencing
consecutive lines recovers the mean over just the interval between them, which
is what a trend needs. Uses only the log, so it costs the running job nothing.
"""
import re, sys, pathlib

pat = re.compile(
    r"ep(\d+) step (\d+)/(\d+).*?([\d.]+)m elapsed.*?mse=([\d.]+) lpips=([\d.]+)"
    r"(?:\s+g=([\d.]+) d=([\d.]+) w=([\d.eE+-]+))?")
log = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/data/vpt/ae_gan.log")
rows = []
for line in log.read_text().splitlines():
    m = pat.search(line)
    if m:
        ep, st, tot, el, mse, lp = int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5]), float(m[6])
        g, d, w = (float(m[7]), float(m[8]), float(m[9])) if m[7] else (None, None, None)
        rows.append((ep, st, el, mse, lp, g, d, w))

BATCH = 128
every = 12  # collapse to ~12 log lines per reported window so trends are legible
print(f"{'ep':>3} {'step':>7} {'%':>5} {'min':>6} | "
      f"{'mse(win)':>9} {'lpips(win)':>10} | {'g':>7} {'d':>7} {'w':>9}")
prev = None
for i, r in enumerate(rows):
    if i % every and i != len(rows) - 1:
        continue
    ep, st, el, mse, lp, g, d, w = r
    if prev is not None and prev[0] == ep:
        pst, pmse, plp, pg, pd, pw = prev[1], prev[3], prev[4], prev[5], prev[6], prev[7]
        n0, n1 = pst * BATCH, st * BATCH
        dn = n1 - n0
        wm = (mse * n1 - pmse * n0) / dn
        wl = (lp * n1 - plp * n0) / dn
        wg = (g * n1 - pg * n0) / dn if g is not None and pg is not None else None
        wd = (d * n1 - pd * n0) / dn if d is not None and pd is not None else None
        ww = (w * n1 - pw * n0) / dn if w is not None and pw is not None else None
    else:
        wm, wl, wg, wd, ww = mse, lp, g, d, w
    gs = f"{wg:7.4f} {wd:7.4f} {ww:9.3g}" if wg is not None else " " * 25
    print(f"{ep:3d} {st:7d} {100*st/129918:5.1f} {el:6.1f} | "
          f"{wm:9.5f} {wl:10.5f} | {gs}")
    prev = r
print("\nbaseline it resumed from: test_mse 0.00716  test_lpips 0.10261")
print("mse(win)/lpips(win) are means over the interval since the line above,")
print("not cumulative -- so they show where the run IS, not where it has been.")
