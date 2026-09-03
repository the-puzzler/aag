import numpy as np, torch
from aag.ae import AutoEncoder
from aag.datasets import open_segments

dev = 'cuda'
ac = torch.load('/data/aag_results/results_vpt/ae_dcae_ch192_dim256_cont/checkpoints/'
                'ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt',
                map_location=dev, weights_only=False)
ae = AutoEncoder(ac['latent_dim'], ch=ac['channels'], architecture=ac['architecture'],
                 image_size=ac['image_size'], grid=ac.get('grid', 4)).to(dev).eval()
sd = ac['model_state_dict']
if any(k.startswith('_orig_mod.') for k in sd):
    sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
ae.load_state_dict(sd)

segs = open_segments('/opt/dlami/nvme/vpt_full')
rng = np.random.default_rng(0)
rt, rt2, step = [], [], []
with torch.no_grad():
    for _ in range(40):
        c = int(rng.integers(0, 831477))
        seg = np.asarray(segs[c][:16])
        x = torch.from_numpy(seg).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(dev)
        r = ae.dec(ae.enc(x)).clamp(-1, 1)
        rt.append(float((r - x).abs().mean()) * 127.5)
        r2 = ae.dec(ae.enc(r)).clamp(-1, 1)
        rt2.append(float((r2 - r).abs().mean()) * 127.5)
        step.append(float((x[1:] - x[:-1]).abs().mean()) * 127.5)

print('all in mean |pixel| on the 0-255 scale (comparable to the motion table)\n')
print(f'  AE round-trip error, real frame      {np.mean(rt):6.2f}')
print(f'  AE round-trip error, 2nd pass        {np.mean(rt2):6.2f}   (re-encoding a reconstruction)')
print(f'  real consecutive frame step          {np.mean(step):6.2f}')
print()
print(f'  ratio  AE error / real frame step    {np.mean(rt)/np.mean(step):6.2f}')
print('  model own-motion at ep20 was ~1.1-1.3 on this scale')
