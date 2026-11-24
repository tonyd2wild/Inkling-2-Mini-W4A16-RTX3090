#!/usr/bin/env bash
# Build the DFlash2 overlay files from the image's own vLLM files plus the diffs in patches/,
# and check every file against SHA256SUMS (stock before patching, patched after).
# Needs: docker, patch, sha256sum. Writes OUT_DIR/*.py (default: the directory of this script's DFLASH2_DIR).
#
# usage: ./prepare_dflash2.sh            (IMAGE=vllm/vllm-openai:inkling-2-mini by default)
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai:inkling-2-mini}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HERE/overlay/dflash2
VL=/usr/local/lib/python3.14/dist-packages/vllm
# in-image path : overlay file name
FILES="model_executor/models/qwen3_dflash.py:qwen3_dflash.py
model_executor/models/qwen3_dflash2.py:qwen3_dflash2.py
v1/worker/gpu/spec_decode/dflash2/speculator.py:dflash2_speculator.py
v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:eagle3_utils.py"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/stock" "$tmp/work"
cid=$(docker create "$IMAGE")
for pair in $FILES; do
  p=${pair%%:*}; f=${pair#*:}
  docker cp "$cid:$VL/$p" "$tmp/stock/$f" >/dev/null
  mkdir -p "$tmp/work/vllm/$(dirname "$p")"; cp "$tmp/stock/$f" "$tmp/work/vllm/$p"
done
docker rm "$cid" >/dev/null

echo "stock files from $IMAGE:"
( cd "$tmp" && grep ' stock/' "$D/SHA256SUMS" | sha256sum -c - ) || {
  echo "The image's files differ from the ones these patches were made against (vLLM 0.1.dev25855+g7cb5dae22)."; exit 1; }

for d in "$D"/patches/*.diff; do ( cd "$tmp/work" && patch -p1 --forward --batch --quiet < "$d" ); done
for pair in $FILES; do
  p=${pair%%:*}; f=${pair#*:}
  cp "$tmp/work/vllm/$p" "$D/$f"
done
echo "patched files:"
( cd "$D" && grep -v ' stock/' SHA256SUMS | sha256sum -c - )
echo "ready: $D"
