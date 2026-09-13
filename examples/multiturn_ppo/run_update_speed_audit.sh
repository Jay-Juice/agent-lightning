#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
TAG="${AGL_UPDATE_AUDIT_TAG:?Set a unique audit tag}"
OUT="$RUNTIME/logs/$TAG"
mkdir "$OUT"
exec >"$OUT/run.log" 2>&1
trap 'echo "$?" > "$OUT/run.exit"' EXIT
source "$RUNTIME/admin/activate-agent-lightning-d1.sh"
export CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$TOOLS:$REPO"
# Refuse to share the trial GPUs with another active CUDA process.
if [[ -n "$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'Trial GPUs are occupied; no process was stopped.'
  exit 2
fi
modes=(baseline actor-micro4)
if [[ "${AGL_UPDATE_AUDIT_BASELINE_ONLY:-0}" == 1 ]]; then
  [[ "${AGL_UPDATE_AUDIT_SKIP_BASELINE:-0}" != 1 ]] || { echo 'Conflicting audit modes'; exit 2; }
  modes=(baseline)
fi
if [[ "${AGL_UPDATE_AUDIT_SKIP_BASELINE:-0}" == 1 ]]; then
  modes=(actor-micro4)
fi
for mode in "${modes[@]}"; do
  flags=()
  [[ "$mode" != actor-micro4 ]] || flags+=(--actor-train-batch 4)
  torchrun --standalone --nnodes=1 --nproc-per-node=4 "$TOOLS/audit_capo_long_context.py" \
    --run "$RUNTIME/logs/training-capo-swe-editor-4b-20260912-01" \
    --output "$OUT/$mode.json" --fused --fresh-model --cohort --actor-zero2 \
    --inference-batch 2 --train-batch 2 --rows 4 \
    --memory-fraction "${AGL_UPDATE_AUDIT_MEMORY_FRACTION:-0.9}" "${flags[@]}" >"$OUT/$mode.log" 2>&1
done
