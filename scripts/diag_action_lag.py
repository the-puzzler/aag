"""Per-control action->frame lag. A single global lag may be right for the mouse
and wrong for the keys: mouse-look rotates the camera on the tick it is applied,
but WASD goes through player acceleration, so its visual effect can be delayed
and smeared over several ticks.

For each binary control and each lag L, measure the mean |frame_t - frame_{t-1}|
when the control was held at tick t-L minus when it was not. Restricted to
low-mouse frames, because mouse motion otherwise swamps every key effect
(16.4% of pixel variance against 2.1% for all keys combined). The lag that
maximises the gap is that control's effective lag.
"""
import numpy as np
from aag.datasets import open_segments

C = '/data/vpt/cache_train'
keys = np.load(f'{C}/keys.npy', mmap_mode='r')
mouse = np.load(f'{C}/mouse.npy', mmap_mode='r')
clicks = np.load(f'{C}/clicks.npy', mmap_mode='r')
segs = open_segments('/opt/dlami/nvme/vpt_full')

NAMES = ['W', 'A', 'S', 'D', 'space', 'shift', 'ctrl', 'E', 'attack', 'use']
LAGS = [0, 1, 2, 3]
rng = np.random.default_rng(0)

acc = {n: {L: [[], []] for L in LAGS} for n in NAMES}
nseg = 0
for _ in range(700):
    c = int(rng.integers(0, 831477))
    seg = np.asarray(segs[c]).astype(np.float32)
    d = np.abs(np.diff(seg, axis=0)).mean((1, 2, 3))       # d[i]=|f_{i+1}-f_i|
    k = np.asarray(keys[c]).astype(np.uint8)
    cl = np.asarray(clicks[c]).astype(np.uint8)
    m = np.asarray(mouse[c]).astype(np.float32)
    am = np.abs(m[:, 0]) + np.abs(m[:, 1])
    ctrl = np.concatenate([k, cl], 1)                       # (80,10)
    nseg += 1
    for t in range(4, 78):
        if am[t - 1] > 6:            # low-mouse only, else mouse dominates
            continue
        chg = d[t - 1]                                      # change INTO frame t
        for L in LAGS:
            for ci_, n in enumerate(NAMES):
                acc[n][L][int(ctrl[t - L, ci_] > 0)].append(chg)

print(f'{nseg} segments, change INTO frame t on low-mouse frames only\n')
print(f"{'control':8s} {'rate':>6s} " + " ".join(f'lag{L}' + ' ' * 4 for L in LAGS)
      + "  best")
for n in NAMES:
    row, gaps = [], []
    for L in LAGS:
        off, on = acc[n][L][0], acc[n][L][1]
        if len(on) < 200:
            row.append('   n/a  '); gaps.append(-1e9); continue
        g = np.mean(on) - np.mean(off)
        gaps.append(g); row.append(f'{g:+7.3f} ')
    rate = len(acc[n][0][1]) / max(1, len(acc[n][0][0]) + len(acc[n][0][1]))
    best = LAGS[int(np.argmax(gaps))] if max(gaps) > -1e8 else None
    star = f'  t-{best}' if best is not None else '  -'
    print(f'{n:8s} {rate:6.3f} ' + " ".join(row) + star)

print('\nsigned mouse, for comparison (from the earlier test): peak at t-1')
print('positive gap = holding the control makes the frame change MORE')
