#!/usr/bin/env bash
# Inkling 2 Mini W4A16 on 2x RTX 3090, tensor parallel 2, vision on.
#   131,072-token context (KV pool ~170K tokens), images accepted.
#   As measured: no speculative decoding (~90 tok/s single stream).
#   DSpark was measured on the text-only lane only; to try it here, append
#   --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
#
# usage: ./launch_tp2.sh [extra vLLM flags]
# env:   GPUS=0,1  MAXLEN=131072  UTIL=0.92  PORT=8130  NAME=inkling-w4a16
#        MODEL_DIR=~/models/Inkling-2-Mini-W4A16  RESTART=no|unless-stopped  FORCE=1
set -eu
. "$(dirname "$0")/common.sh"
GPUS=${GPUS:-0,1}
preflight "$GPUS"

# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --ipc=host --restart "$RESTART" -p "$PORT":8000 \
  -e NCCL_P2P_LEVEL=NVL \
  -v "$MODEL_DIR":/models/model:ro $OVERLAY_MOUNTS "$IMAGE" /models/model \
  --served-model-name inkling-2-mini-w4a16 inkling-2-mini \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len "${MAXLEN:-131072}" --gpu-memory-utilization "${UTIL:-0.92}" "$@"
started
