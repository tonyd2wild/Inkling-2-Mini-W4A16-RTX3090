#!/usr/bin/env bash
# Inkling 2 Mini W4A16 on 2x RTX 3090, tensor parallel 2, with the DFlash2 drafter
# (incoai/Inkling-2-Mini-DFlash2, k=7) instead of DSpark. Same lane as launch_tp2_max.sh otherwise:
# NCCL over NVLink P2P, text-only, 458,752-token context, BF16 KV.
#
# Before the first run: ./prepare_dflash2.sh (builds overlay/dflash2/*.py from the image + patches)
# and download the drafter:  hf download incoai/Inkling-2-Mini-DFlash2 --local-dir ~/models/Inkling-2-Mini-DFlash2
#
# usage: ./launch_tp2_dflash2.sh [extra vLLM flags]
# env:   GPUS=0,1  K=7  MAXLEN=458752  CHUNK=1024  SEQS=4  PORT=8130  NAME=inkling-w4a16
#        MODEL_DIR=~/models/Inkling-2-Mini-W4A16  DRAFT_DIR=~/models/Inkling-2-Mini-DFlash2
#        RESTART=no|unless-stopped  FORCE=1
set -eu
. "$(dirname "$0")/common.sh"
GPUS=${GPUS:-0,1}
K=${K:-7}
DRAFT_DIR=${DRAFT_DIR:-$HOME/models/Inkling-2-Mini-DFlash2}
[ -f "$DRAFT_DIR/model.safetensors" ] || { echo "drafter not found in $DRAFT_DIR (set DRAFT_DIR=...)"; exit 1; }
[ -f "$OV/dflash2/qwen3_dflash2.py" ] || { echo "run ./prepare_dflash2.sh first"; exit 1; }
preflight "$GPUS"

# The 4 DFlash2 files go over the image's generic vLLM code (not the inkling2 package).
VL=/usr/local/lib/python3.14/dist-packages/vllm
D=$OV/dflash2
DFLASH2_MOUNTS="-v $DRAFT_DIR:/models/dflash2:ro \
 -v $D/qwen3_dflash.py:$VL/model_executor/models/qwen3_dflash.py:ro \
 -v $D/qwen3_dflash2.py:$VL/model_executor/models/qwen3_dflash2.py:ro \
 -v $D/dflash2_speculator.py:$VL/v1/worker/gpu/spec_decode/dflash2/speculator.py:ro \
 -v $D/eagle3_utils.py:$VL/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:ro"

# The image has no SM86 build of vLLM's rms_norm / fused_add_rms_norm / rotary_embedding kernels.
# The drafter's norms and RoPE use vLLM's own torch implementations ("native") instead.
# The Inkling target does not use these ops, so this changes nothing for it.
KCFG='{"ir_op_priority":{"rms_norm":["native"],"fused_add_rms_norm":["native"],"rotary_embedding":["native"],"rotary_embedding_query_only":["native"]}}'

# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --ipc=host --restart "$RESTART" -p "$PORT":8000 \
  -e NCCL_P2P_LEVEL=NVL -e NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
  -v "$MODEL_DIR":/models/model:ro $OVERLAY_MOUNTS $DFLASH2_MOUNTS "$IMAGE" /models/model \
  --served-model-name inkling-2-mini-w4a16 inkling-2-mini \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --tensor-parallel-size 2 --disable-custom-all-reduce --language-model-only \
  --max-model-len "${MAXLEN:-458752}" --max-num-batched-tokens "${CHUNK:-1024}" --max-num-seqs "${SEQS:-4}" \
  --kernel-config "$KCFG" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/dflash2\",\"num_speculative_tokens\":$K}" "$@"
started
