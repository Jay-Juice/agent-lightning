#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-recovery-20260920
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
OUT=$RUNTIME/logs/swe-full-python-envs-reliable-v2-syntaxfix-20260920-01
[[ ! -e "$OUT" ]] || { printf 'Refuse existing output: %s\n' "$OUT" >&2; exit 2; }
mkdir "$OUT"
trap 'status=$?; printf "%s\n" "$status" > "$OUT/run.exit"' EXIT
exec >"$OUT/run.log" 2>&1
source "$RUNTIME/admin/activate-agent-lightning-d1.sh"
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export SMITH_RELIABILITY=1 AGL_SWE_RELIABILITY=1 SMITH_EVAL_TIMEOUT=600
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
printf 'CPU/Docker recovery environment audit started: %s\n' "$(date -Iseconds)"
python examples/multiturn_ppo/audit_full_python_envs.py \
  --data "$RUNTIME/data/swe-smith-training/python-full-v7" \
  --output "$OUT" --cached-only --workers 2 --min-free-gib 150
printf 'CPU/Docker recovery environment audit finished: %s\n' "$(date -Iseconds)"
