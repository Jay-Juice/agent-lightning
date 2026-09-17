#!/usr/bin/env bash
set -Eeuo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
PREPARED="${1:?Pass a freshly prepared checkpoint-roundtrip directory}"
ACTOR_PROOF="${2:-}"
[[ -f "$PREPARED/inputs.json" && -f "$PREPARED/source-config.json" ]]
[[ ! -e "$PREPARED/run.exit" && ! -e "$PREPARED/actor" && ! -e "$PREPARED/critic" ]]
exec >"$PREPARED/run.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$PREPARED/run.exit"' EXIT
export PYTHONPATH="$SOURCE/research:$SOURCE:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=4,5,6,7
export AGL_CAPO_STRICT_PADDING=1 AGL_SWE_RELIABILITY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
unset RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
roles=(actor critic)
summary_args=()
if [[ -n "$ACTOR_PROOF" ]]; then
  [[ -f "$ACTOR_PROOF/completed.json" ]]
  roles=(critic)
  summary_args=(--actor-proof "$ACTOR_PROOF")
fi
for role in "${roles[@]}"; do
  mapfile -t used < <(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ "${#used[@]}" -eq 4 ]]
  for memory in "${used[@]}"; do
    [[ "$memory" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && (( memory < 1000 )) || {
      echo 'GPU 4-7 occupied; refusing to interfere with an existing experiment'; exit 2;
    }
  done
  timeout --signal=TERM --kill-after=60 7200 torchrun --standalone --nproc_per_node=4 \
    "$SOURCE/research/checkpoint_roundtrip.py" run --prepared "$PREPARED" --role "$role"
done
CUDA_VISIBLE_DEVICES= python "$SOURCE/research/checkpoint_roundtrip.py" summarize \
  --prepared "$PREPARED" "${summary_args[@]}"
