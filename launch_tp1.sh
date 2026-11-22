#!/usr/bin/env bash
# Inkling 2 Mini W4A16 on ONE RTX 3090 (24 GB), text-only.
#   Weights take 14.77 GiB, so the KV pool is small: 8,192 context by default,
#   up to 32,768 (KV pool 33,392 tokens). ~74 tok/s single stream, no spec decode.
#
# usage: ./launch_tp1.sh [extra vLLM flags]          MAXLEN=32768 ./launch_tp1.sh
# env:   GPUS=0  MAXLEN=8192  UTIL=0.95  SEQS=4  PORT=8130  NAME=inkling-w4a16
#        MODEL_DIR=~/models/Inkling-2-Mini-W4A16  RESTART=no|unless-stopped  FORCE=1
set -eu
. "$(dirname "$0")/common.sh"
GPUS=${GPUS:-0}
preflight "$GPUS"

# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --ipc=host --restart "$RESTART" -p "$PORT":8000 \
  -v "$MODEL_DIR":/models/model:ro $OVERLAY_MOUNTS "$IMAGE" /models/model \
  --served-model-name inkling-2-mini-w4a16 inkling-2-mini \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --tensor-parallel-size 1 --disable-custom-all-reduce \
  --max-model-len "${MAXLEN:-8192}" --gpu-memory-utilization "${UTIL:-0.95}" \
  --max-num-seqs "${SEQS:-4}" --limit-mm-per-prompt '{"image":0}' "$@"
started
