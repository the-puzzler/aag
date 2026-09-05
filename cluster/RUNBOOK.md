# AAG 256x256 scale-up on eks-train-prod-aps3 (1 node, 8x B200, project warhol)

## Pipeline (one node per dataset: `cluster/configs/aag256_celebahq.yaml`, `aag256_imagenet.yaml`)
Split (user's call, 2026-09-05): **the local box makes the assignment, the cluster trains the generator.**

Local box (`ip-10-10-13-216`, RTX PRO 6000; run with `PYTHONPATH=<worktree>`):
1. `scripts/encode_hf256.py --dataset imagenet256 --amp --workers 12 --batch 128` -> `/data/aag_data/imagenet256/particles_dcae_f32c32.pt`
   (GPU-bound, ~220 img/s, ~100 min; the CelebA-HQ particles already exist)
2. `scripts/run_assignment_classes.py` (CelebA-HQ 200k steps; ImageNet 300k with hierarchy levels,
   `--save-every 50000 --keep-checkpoints`) -> `/data/aag_results/results_scale256/<ds>/assign/`
3. copy the assignment to EFS through the CPU-only helper pod (no bucket needed):
   `kubectl -n kubeflow cp <file> kubeflow/aagcp:/mnt/shared/aag/results_scale256/<ds>/assign/<file>.part`
   then `kubectl -n kubeflow exec aagcp -- mv ....part ...` (~8 MB/s; pod spec: busybox + PVC `shared-drive`)

Cluster job (`cluster/entry.py` runs the YAML stages; the YAML is mounted at submit time, so config
changes need no image rebuild -- code changes do):
1. download the HF parquet mirror to EFS (`/mnt/shared/aag/hf`)
2. `wait_assignment`: `cluster/wait_for_file.py $ASSIGN` blocks until the upload exists and has stopped growing
3. FID reference stats
4. generator on all 8 GPUs (CelebA-HQ 38M params / 400 epochs; ImageNet 140M / 40 epochs)

Results land under `/mnt/shared/aag/results_scale256/{celebahq,imagenet}/` -- sample grids
`samples_epNNN.png`, held-out pairs `heldout_pairs_epNNN.png`, `curve.json`, checkpoints.

## Commands (from the aag worktree)
```bash
aws sso login --sso-session odyssey                 # when kubectl says the token expired
cluster/launch.sh build                             # docker build + push -> odydev.azurecr.io/aag:<git hash>
cluster/launch.sh dry    cluster/configs/aag256_celebahq.yaml aag1   # print manifest
# job names are deliberately plain (user: "aag1", "aag2") so they are easy to spot in k9s
cluster/launch.sh submit cluster/configs/aag256_celebahq.yaml aag1
cluster/launch.sh submit cluster/configs/aag256_imagenet.yaml aag2
kubectl -n kubeflow get pytorchjobs | grep aag
kubectl -n kubeflow logs -f -l training.kubeflow.org/job-name=aag2 --tail=100
kubectl -n kubeflow delete pytorchjob aag2       # delete.py needs a TTY
# a job stuck in ImagePullBackOff for ~20 min gets Suspended by the scheduler and its pod removed;
# AAG_IMAGE_TAG=<tag> selects an already-pushed image, AAG_IMAGE_REPO=<repo> the in-region ECR spare
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
