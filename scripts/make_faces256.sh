#!/bin/bash
# Build the combined faces256 dataset dir: CelebA-HQ train parquets first, then FFHQ, as symlinks
# whose names sort in that order (aag.hf256 defines particle identity by sorted filename order).
# usage: scripts/make_faces256.sh [/data/aag_data/hf]
set -e
ROOT=${1:-/data/aag_data/hf}; OUT=$ROOT/faces-256x256/data; mkdir -p $OUT
i=0
for f in $(ls $ROOT/celeba-hq-256x256/data/train-*.parquet | sort); do ln -sfn $f $OUT/train-$(printf %03d $i)-celebahq-$(basename $f); i=$((i+1)); done
for f in $(ls $ROOT/ffhq-256/data/train-*.parquet | sort); do ln -sfn $f $OUT/train-$(printf %03d $i)-ffhq-$(basename $f); i=$((i+1)); done
# validation: CelebA-HQ's 2k val split (FFHQ has none)
for f in $(ls $ROOT/celeba-hq-256x256/data/validation-*.parquet | sort); do ln -sfn $f $OUT/$(basename $f); done
rm -f $OUT/_manifest_*.json
echo "$i train parquets linked under $OUT"; ls $OUT | head -3; ls $OUT | tail -2
