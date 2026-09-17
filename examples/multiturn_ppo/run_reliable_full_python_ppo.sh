#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
source "$RUNTIME/admin/activate-agent-lightning-d1.sh"
# P0 protocol only: inherit the checked full-data PPO recipe unchanged.
# No training is started by adding this file; occupied GPUs remain protected by
# run_training.sh. Use a new run directory and a separate server port.
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:-capo-swe-pythonfull-v2-reliable-4b-$(date +%Y%m%d-%H%M%S)}"
export AGL_TRAIN_PORT="${AGL_TRAIN_PORT:-18521}"
export SMITH_RELIABILITY=1 AGL_SWE_RELIABILITY=1
export SMITH_AGENT_WALL_TIMEOUT="${SMITH_AGENT_WALL_TIMEOUT:-3600}"
export SMITH_EVAL_TIMEOUT="${SMITH_EVAL_TIMEOUT:-600}"
export AGL_ROLLOUT_TIMEOUT_SECONDS="${AGL_ROLLOUT_TIMEOUT_SECONDS:-5400}"
# 3 * 91.39 GiB + 20 GiB = 294.17 GiB, rounded up. This protects
# checkpoint writing before retention removes the oldest complete checkpoint.
export AGL_MIN_FREE_GIB="${AGL_MIN_FREE_GIB:-295}"
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python "$TOOLS/reliable_launch.py" \
  "$RUNTIME" --minimum-gib "$AGL_MIN_FREE_GIB"
# Unknown grader outcomes (including unclassified 3/124/137) fail closed;
# they are not silently recorded as model failures or resampled to success.
bash "$TOOLS/run_full_python_ppo.sh" \
  agentlightning.reliability.enabled=true \
  agentlightning.reliability.infrastructure_retries=1 \
  agentlightning.reliability.checkpoint_reserve_gib=112 \
  'agentlightning.local.env_map.SMITH_RELIABILITY="1"' \
  'agentlightning.local.env_map.AGL_SWE_RELIABILITY="1"' \
  "agentlightning.local.env_map.SMITH_AGENT_WALL_TIMEOUT=\"$SMITH_AGENT_WALL_TIMEOUT\"" \
  "agentlightning.local.env_map.SMITH_EVAL_TIMEOUT=\"$SMITH_EVAL_TIMEOUT\"" \
  "agentlightning.local.env_map.AGL_ROLLOUT_TIMEOUT_SECONDS=\"$AGL_ROLLOUT_TIMEOUT_SECONDS\"" \
  trainer.max_actor_ckpt_to_keep=2 trainer.max_critic_ckpt_to_keep=2 \
  trainer.val_before_train=true trainer.test_freq=20 "$@"
