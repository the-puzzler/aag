# AAG 256x256 scale-up on eks-train-prod-aps3 (1 node, 8x B200, project warhol)

## Pipeline (all inside one job, `cluster/configs/aag256.yaml`)
1. download the two HF parquet mirrors to EFS (`/mnt/shared/aag/hf`)
2. encode both datasets with the pretrained DC-AE f32c32-in encoder (8-way sharded, ~min)
3. in parallel: ImageNet class-conditional assignment on GPU 0 (100k steps, checkpoints every 10k)
   || CelebA-HQ unconditional assignment + 7-GPU generator on GPUs 1-7
4. ImageNet class-conditional generator on all 8 GPUs (140M params, 40 epochs)

Results land under `/mnt/shared/aag/results_scale256/{celebahq,imagenet}/` -- sample grids
`samples_epNNN.png`, held-out pairs `heldout_pairs_epNNN.png`, `curve.json`, checkpoints.

## Commands (from the aag worktree)
```bash
aws sso login --sso-session odyssey                 # when kubectl says the token expired
cluster/launch.sh build                             # docker build + push -> odydev.azurecr.io/aag:<git hash>
cluster/launch.sh dry    cluster/configs/aag256.yaml aag256   # print manifest
cluster/launch.sh submit cluster/configs/smoke.yaml  aag-smoke # 5-min sanity job first
cluster/launch.sh submit cluster/configs/aag256.yaml aag256
kubectl -n kubeflow get pytorchjobs | grep aag
kubectl -n kubeflow logs -f -l training.kubeflow.org/job-name=aag256 --tail=100
cd /data/tmp/odyssey-main/tools/odytrain && uv run python delete.py --job_name=aag256
```
Set `AAG_IMAGE_TAG=<tag>` to launch a tag other than the current git hash.

## Resuming
Completed single stages leave markers in `/mnt/shared/aag/state/<config name>/`; a relaunch with
the same config name skips them. Delete a marker to redo that stage. The generator trainers take
`--resume <checkpoint>`; the assignment takes `--resume-z <assignment.pt>`.

## Reading the assignment log
Per eval line: `obj` (transport objective; meaningless once inside the printed floor),
`G` (global Gaussian defect), `disp` (mean transport displacement, a progress meter),
and one independence ratio PER LEVEL (`joint`, `depth7`, ... `dog`). Never quote one level.
Selection between assignment checkpoints is by the generator's held-out pair MSE/LPIPS,
and quality by looking at the sample grids.
