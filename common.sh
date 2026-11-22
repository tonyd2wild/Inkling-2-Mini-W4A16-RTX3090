# Shared settings for the launch scripts. Sourced, not run.
# Override any of these in the environment, e.g. PORT=9000 ./launch_tp2_max.sh

NAME=${NAME:-inkling-w4a16}
PORT=${PORT:-8130}
IMAGE=${IMAGE:-vllm/vllm-openai:inkling-2-mini}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Inkling-2-Mini-W4A16}
RESTART=${RESTART:-no}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OV=$REPO_DIR/overlay
# Where the image keeps the inkling2 model code (vLLM 0.1.dev25855+g7cb5dae22, Python 3.14).
V=/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2

# The overlay: 8 files bind-mounted read-only over the image's inkling2 code.
# Nothing in the image or on the host is changed. See overlay/README.md.
OVERLAY_MOUNTS="-v $OV/port/ops_init.py:$V/nvidia/ops/__init__.py:ro \
 -v $OV/port/vision.py:$V/nvidia/vision.py:ro \
 -v $OV/port/linear.py:$V/nvidia/ops/linear.py:ro \
 -v $OV/port/conv_mix.py:$V/nvidia/ops/conv_mix.py:ro \
 -v $OV/port/moe_sm90.py:$V/nvidia/ops/moe_sm90.py:ro \
 -v $OV/w4a16/quant_config.py:$V/quant_config.py:ro \
 -v $OV/w4a16/model.py:$V/nvidia/model.py:ro \
 -v $OV/w4a16/moe.py:$V/nvidia/moe.py:ro"

# preflight GPU_LIST: model files present, chosen GPUs idle (set FORCE=1 to skip
# the GPU check), then remove any old container with the same name.
preflight() {
  local gpus=$1 busy=""
  [ -f "$MODEL_DIR/model.safetensors.index.json" ] && [ -f "$MODEL_DIR/mtp.safetensors" ] || {
    echo "model files not found in $MODEL_DIR (set MODEL_DIR=...)"; exit 1; }
  [ -f "$OV/port/ops_init.py" ] || { echo "overlay not found in $OV"; exit 1; }
  if [ "${FORCE:-0}" != 1 ] && [ -z "$(docker ps -q --filter "name=^${NAME}\$")" ]; then
    for g in ${gpus//,/ }; do
      used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      if [ "${used:-0}" -gt 500 ]; then busy="$busy GPU$g(${used}MiB)"; fi
    done
    if [ -n "$busy" ]; then
      echo "busy:$busy. Free those GPUs first, or set FORCE=1."; exit 1
    fi
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}

started() {
  echo "starting $NAME on :$PORT (about 2.5 min: weights, warmup, CUDA graphs)."
  echo "follow:  docker logs -f $NAME   (ready at 'Application startup complete')"
}
