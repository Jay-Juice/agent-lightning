#!/usr/bin/env bash
set -Eeuo pipefail
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
TAG="${AGL_TRAIN_TAG:-ppo-$(date +%Y%m%d-%H%M%S)}"
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export AGL_RUN_DIR="$AGL_RUNTIME/logs/training-$TAG"
test ! -e "$AGL_RUN_DIR" || { echo "Run exists: $AGL_RUN_DIR" >&2; exit 2; }
mkdir -p "$AGL_RUN_DIR"/{agent,traces}
exec >"$AGL_RUN_DIR/run.log" 2>&1
export CUDA_VISIBLE_DEVICES="${AGL_GPUS:-${AGL_GPU:-4}}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-7](,[0-7])*$ ]] || { echo 'Expected comma-separated physical GPU indices'; exit 2; }
while read -r used; do
  (( used < 1000 )) || { echo 'A selected GPU is occupied'; exit 2; }
done < <(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
export PYTHONPATH="$TOOLS:$REPO:${PYTHONPATH:-}"
# Ray uses Unix-domain sockets; its parent path must stay short (<108 bytes
# including Ray's session/socket suffix). This directory is still on D1.
export RAY_TMPDIR=/media/ubuntu/D1/zsj/ray
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 RAY_USAGE_STATS_ENABLED=0
# The shared 14 TB data disk is >95% used but has hundreds of GB available.
# Keep Ray's spill limit local to this launch, with an absolute free-space guard.
export RAY_local_fs_capacity_threshold=0.99
free_kb=$(df --output=avail /media/ubuntu/D1 | tail -1)
(( free_kb > 150 * 1024 * 1024 )) || { echo 'D1 needs at least 150 GiB free for this run'; exit 2; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
# Do not send private LAN Ray/vLLM traffic through the workstation's Internet proxy.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY='*' no_proxy='*'
export AGL_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
PORT="${AGL_TRAIN_PORT:-18281}"
export AGL_BASE_URL="http://127.0.0.1:$PORT"
MODEL="${AGL_TRAIN_MODEL:-/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-0.6B}"
export AGL_TRAIN_MODEL="$MODEL"
LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-4}"
[[ "$LOCAL_AGENTS" =~ ^[1-9][0-9]*$ ]] || { echo 'AGL_MAX_LOCAL_AGENTS must be positive'; exit 2; }
server_pid='' controller_pid='' monitor_pid=''
cleanup() {
  code=$?
  trap - EXIT INT TERM
  for pid in "$controller_pid" "$server_pid" "$monitor_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
  printf '%s\n' "$code" > "$AGL_RUN_DIR/run.exit"
}
trap cleanup EXIT
if [[ "${AGL_GPU_MONITOR:-0}" == 1 ]]; then
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits -l 2 >"$AGL_RUN_DIR/gpu.csv" 2>"$AGL_RUN_DIR/gpu-monitor.log" &
  monitor_pid=$!
fi
python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as s: s.bind(('127.0.0.1', int(sys.argv[1])))
PY
agl-server host=127.0.0.1 port="$PORT" key="$AGL_KEY" default_proxy.model_name="$MODEL" \
  >"$AGL_RUN_DIR/server.log" 2>&1 &
server_pid=$!
for ((i=0;i<60;i++)); do
  curl -fs "$AGL_BASE_URL/healthz" >/dev/null && break
  kill -0 "$server_pid" || exit 1
  sleep 1
done
curl -fs "$AGL_BASE_URL/healthz" >/dev/null
agl-controller runner_type=local agl_server.url="$AGL_BASE_URL" agl_server.key="$AGL_KEY" \
  local_runner.maximum_size="$LOCAL_AGENTS" local_runner.poll_interval=1 >"$AGL_RUN_DIR/controller.log" 2>&1 &
controller_pid=$!
python -u "$TOOLS/train.py" --model "$MODEL" "$@" >"$AGL_RUN_DIR/trainer.log" 2>&1
echo TRAINING_FINISHED
