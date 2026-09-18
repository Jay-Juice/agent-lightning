#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
OUT=${1:-$RUNTIME/logs/audit-v2-grading-live-20260918-01}
[[ ! -e "$OUT" ]] || { printf 'Refuse existing output: %s\n' "$OUT" >&2; exit 2; }
mkdir "$OUT"
trap 'status=$?; printf "%s\n" "$status" > "$OUT/run.exit"' EXIT
exec >"$OUT/run.log" 2>&1
source "$RUNTIME/admin/activate-agent-lightning-d1.sh"
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void PYTHONDONTWRITEBYTECODE=1
export SMITH_RELIABILITY=1 AGL_SWE_RELIABILITY=1 SMITH_EVAL_TIMEOUT=600
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python research/verify_v2_grading_live.py --source "$PROJECT" \
  --original "$RUNTIME/logs/training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-01" \
  --output "$OUT" "${@:2}"
