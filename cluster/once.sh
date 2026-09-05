#!/usr/bin/env bash
# Run <cmd> on RANK 0 of the current (sub-)group only; other ranks wait for it.
#   cluster/once.sh <marker-name> <cmd...>
# Lets a ddp sub-stage chain a one-GPU step (assignment, FID stats) before its
# multi-GPU step without a separate stage. Markers live in /dev/shm (one node).
set -euo pipefail
M="/dev/shm/aag_once_$1"; shift
if [[ "${RANK:-0}" == "0" ]]; then
  rm -f "$M.failed"
  if "$@"; then touch "$M.done"; else touch "$M.failed"; exit 1; fi
else
  while [[ ! -e "$M.done" ]]; do [[ -e "$M.failed" ]] && exit 1; sleep 5; done
fi
