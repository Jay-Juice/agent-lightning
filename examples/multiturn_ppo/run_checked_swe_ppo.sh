#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reproduce the 2026-09-12 checked-editor SWE pilot; see SWE_DIAGNOSIS_2026-09-12.md.
export AGL_GPUS="${AGL_GPUS:-0,1,2,3}"
[[ "$AGL_GPUS" == "0,1,2,3" ]] || { echo 'This recipe reserves physical GPUs 0,1,2,3 only'; exit 2; }
export AGL_TRAIN_MODEL="${AGL_TRAIN_MODEL:-/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507}"
export AGL_MAX_LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-8}"
export SMITH_MAX_TURNS="${SMITH_MAX_TURNS:-32}" SMITH_MAX_TOKENS="${SMITH_MAX_TOKENS:-4096}"
export SMITH_CONTEXT="${SMITH_CONTEXT:-65536}" SMITH_OBS_CHAR_CAP="${SMITH_OBS_CHAR_CAP:-32000}"
export SMITH_MODEL_TIMEOUT="${SMITH_MODEL_TIMEOUT:-600}"
export SMITH_VERIFY_SUBMISSION=1 SMITH_ALLOW_REPRO_FILES=1 SMITH_CHECK_SYNTAX=1 SMITH_CHECKED_EDITOR=1
bash "$TOOLS/run_capo_ppo.sh" --steps 2 trainer.total_epochs=2 \
  agentlightning.rollout_timeout_seconds=3600 \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.8 actor_rollout_ref.rollout.val_kwargs.top_k=20 "$@"
