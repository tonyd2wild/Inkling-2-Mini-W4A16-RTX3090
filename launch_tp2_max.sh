#!/usr/bin/env bash
# Inkling 2 Mini W4A16 on 2x RTX 3090, tensor parallel 2: the max-context lane.
#   NCCL all-reduce over NVLink P2P, DSpark speculative decoding (k=7),
#   text-only, 458,752-token context (KV pool ~484K tokens), BF16 KV.
# Measured: 137.5 tok/s single stream (100.5 with P2P off), needles to 439K.
#
# usage: ./launch_tp2_max.sh [extra vLLM flags]
# env:   GPUS=0,1  MAXLEN=458752  CHUNK=1024  SEQS=4  PORT=8130  NAME=inkling-w4a16
#        MODEL_DIR=~/models/Inkling-2-Mini-W4A16  RESTART=no|unless-stopped  FORCE=1
set -eu
. "$(dirname "$0")/common.sh"
GPUS=${GPUS:-0,1}
preflight "$GPUS"

# NCCL_P2P_LEVEL=NVL: use P2P only over NVLink. Without a bridge NCCL falls back
# to host memory (the 100.5 tok/s case). NCCL_DEBUG=INFO prints "via P2P/CUMEM"
# in the log when NVLink is in use.
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --ipc=host --restart "$RESTART" -p "$PORT":8000 \
  -e NCCL_P2P_LEVEL=NVL -e NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
  -v "$MODEL_DIR":/models/model:ro $OVERLAY_MOUNTS "$IMAGE" /models/model \
  --served-model-name inkling-2-mini-w4a16 inkling-2-mini \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --tensor-parallel-size 2 --disable-custom-all-reduce --language-model-only \
  --max-model-len "${MAXLEN:-458752}" --max-num-batched-tokens "${CHUNK:-1024}" --max-num-seqs "${SEQS:-4}" \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}' "$@"
started
