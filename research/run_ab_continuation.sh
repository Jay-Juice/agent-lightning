#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
ARM="${1:?Specify A or B}"
[[ "$ARM" == A || "$ARM" == B ]]
PREPARED=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/ab-continuation-preflight-20260919-01
[[ -f "$PREPARED/completed.json" ]]
mapfile -t spec < <(CUDA_VISIBLE_DEVICES= python - "$PREPARED/$ARM.json" <<'PY'
import json, sys
v=json.load(open(sys.argv[1]))
for key in ('head','gpus','port','tag','seed','checkpoint_root'):
    print(v[key])
print(json.dumps(v['expected_data_ids'],separators=(',',':')))
PY
)
[[ "${#spec[@]}" == 7 ]]
export AGL_GPUS="${spec[1]}" AGL_TRAIN_PORT="${spec[2]}"
export AGL_TRAIN_TAG="${AGL_RESUME_TAG_OVERRIDE:-${spec[3]}}"
export AGL_RAY_TMPDIR="/media/ubuntu/D1/zsj/ray-ab-${ARM,,}-0919"
export AGL_FULL_ENV_AUDIT="${AGL_FULL_ENV_AUDIT:-/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-reliable-v2-20260918-03}"
export AGL_GPU_MONITOR=1 AGL_EXPORT_FINAL_ACTOR=0 AGL_MIN_FREE_GIB=295
unset FLASH_ATTENTION_DETERMINISTIC CUBLAS_WORKSPACE_CONFIG RAY_ADDRESS
exec bash "$REPO/examples/multiturn_ppo/run_reliable_full_python_ppo.sh" \
  trainer.total_training_steps=784 trainer.total_epochs=4 trainer.save_freq=5 \
  trainer.resume_mode=resume_path "trainer.resume_from_path=${spec[4]}" \
  "trainer.default_local_dir=${spec[5]}" \
  trainer.del_local_ckpt_after_load=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 critic.ppo_mini_batch_size=128 \
  "agentlightning.multi_turn_ppo.critic_head_init=${spec[0]}" \
  agentlightning.reliability.retention_across_resume=true \
  agentlightning.reliability.checkpoint_lock_path=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/.swe-ab-checkpoint.lock \
  agentlightning.reliability.resume_expected_step=21 \
  "agentlightning.reliability.resume_expected_data_ids=${spec[6]}"
