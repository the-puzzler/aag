#!/usr/bin/env bash
# Build+push the image and submit a 1-node B200 job on eks-train-prod-aps3 under project warhol.
#   cluster/launch.sh build                       # docker build + push, prints the tag
#   cluster/launch.sh submit <config.yaml> <job>  # odytrain torchrun launch with that stage config
#   cluster/launch.sh dry <config.yaml> <job>     # print the manifest only
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ODY="${ODYSSEY_MAIN:-/data/tmp/odyssey-main}"          # fresh origin/main worktree: knows aps3
REG=odydev.azurecr.io/aag
# odytrain refuses to attribute jobs to a shared account (ubuntu/root); jobs carry odyssey.systems/user=<this>
export ODYSSEY_USER="${ODYSSEY_USER:-matteopeluso}"
TAG="${AAG_IMAGE_TAG:-$(git -C "$HERE" rev-parse --short HEAD)}"
IMAGE="$REG:$TAG"
case "${1:-}" in
build)
  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f "$HERE/cluster/Dockerfile" -t "$IMAGE" --push "$HERE"
  echo "pushed $IMAGE" ;;
submit|dry)
  CFG="$(realpath "$2")"; JOB="$3"
  EXTRA=(); [[ "$1" == dry ]] && EXTRA+=(--dry_run)
  cd "$ODY/tools/odytrain" && uv run python launch.py \
    --image_path="$IMAGE" --cluster=eks-train-prod-aps3 --gpu_type=b200 \
    --num_nodes=1 --project=warhol --job_name="$JOB" \
    --use_torchrun --launcher_script=/app/aag/cluster/entry.py \
    --config_path="$CFG" --nodisable_ipv4_egress --yes --nohuman_killable "${EXTRA[@]}" ;;
*) sed -n 2,5p "$0"; exit 1 ;;
esac
