#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
PREPARED=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/b-resume75-preflight-20260920-01
[[ -f "$PREPARED/completed.json" ]]
mapfile -t spec < <(CUDA_VISIBLE_DEVICES= python - "$PREPARED/B.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
for key in ('gpus', 'port', 'tag', 'checkpoint', 'checkpoint_root', 'expected_first_step'):
    print(v[key])
print(json.dumps(v['expected_data_ids'], separators=(',', ':')))
PY
)
[[ "${#spec[@]}" == 7 ]]
export AGL_GPUS="${spec[0]}" AGL_TRAIN_PORT="${spec[1]}" AGL_TRAIN_TAG="${spec[2]}"
export AGL_RAY_TMPDIR=/media/ubuntu/D1/zsj/ray-b-resume75-0920-01
export AGL_FULL_ENV_AUDIT=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-reliable-v2-20260919-01
export AGL_GPU_MONITOR=1 AGL_EXPORT_FINAL_ACTOR=0 AGL_MIN_FREE_GIB=295
unset FLASH_ATTENTION_DETERMINISTIC CUBLAS_WORKSPACE_CONFIG RAY_ADDRESS
exec bash "$REPO/examples/multiturn_ppo/run_reliable_full_python_ppo.sh" \
  trainer.total_training_steps=784 trainer.total_epochs=4 trainer.save_freq=5 \
  trainer.resume_mode=resume_path "trainer.resume_from_path=${spec[3]}" \
  "trainer.default_local_dir=${spec[4]}" trainer.del_local_ckpt_after_load=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 critic.ppo_mini_batch_size=128 \
  agentlightning.multi_turn_ppo.critic_head_init=zero \
  agentlightning.reliability.retention_across_resume=true \
  agentlightning.reliability.checkpoint_lock_path=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/.swe-ab-checkpoint.lock \
  "agentlightning.reliability.resume_expected_step=${spec[5]}" \
  "agentlightning.reliability.resume_expected_data_ids=${spec[6]}"
