#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGL_GPUS="${AGL_GPUS:-0,1,2,3}"
[[ "$AGL_GPUS" == "0,1,2,3" ]] || { echo 'PI-v2 is reserved for physical GPUs 0-3'; exit 2; }
export AGL_TRAIN_PORT="${AGL_TRAIN_PORT:-18701}"
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:-capo-swe-pythonfull-v7-p2-4b-gpu03-$(date +%Y%m%d-%H%M%S)}"
export AGL_FULL_ENV_AUDIT="${AGL_FULL_ENV_AUDIT:-/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-p1-20260914-01}"
export SMITH_PRIVILEGED_STATE=critic
export SMITH_PRIVILEGED_ENCODING=semantic_hunks_v2
export SMITH_PRIVILEGED_MAX_TOKENS=4096
export SMITH_PRIVILEGED_SAFETY_MARGIN=32
bash "$TOOLS/run_full_python_ppo.sh" \
  agentlightning.privileged_critic.enabled=true \
  agentlightning.privileged_critic.state_schema_version=1 \
  agentlightning.privileged_critic.encoding=semantic_hunks_v2 \
  agentlightning.privileged_critic.max_tokens="$SMITH_PRIVILEGED_MAX_TOKENS" \
  agentlightning.privileged_critic.safety_margin="$SMITH_PRIVILEGED_SAFETY_MARGIN" \
  "$@"
