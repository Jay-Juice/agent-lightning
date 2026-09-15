#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGL_GPUS="${AGL_GPUS:-0,1,2,3}"
[[ "$AGL_GPUS" == "0,1,2,3" ]] || { echo 'PI-v2 is reserved for physical GPUs 0-3'; exit 2; }
export AGL_TRAIN_PORT="${AGL_TRAIN_PORT:-18701}"
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:-capo-swe-pythonfull-v7-p2-4b-gpu03-$(date +%Y%m%d-%H%M%S)}"
export AGL_FULL_ENV_AUDIT="${AGL_FULL_ENV_AUDIT:-/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-p2-20260915-01}"
export AGL_TRAIN_MODEL="${AGL_TRAIN_MODEL:-/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507}"
export AGL_CAPO_PROGRESS=1 AGL_MAX_LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-32}" AGL_MIN_FREE_GIB="${AGL_MIN_FREE_GIB:-360}"
export SMITH_PRIVILEGED_STATE=critic SMITH_PRIVILEGED_ENCODING=semantic_hunks_v2
export SMITH_PRIVILEGED_MAX_TOKENS=4096 SMITH_PRIVILEGED_SAFETY_MARGIN=32
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
DATA="${AGL_FULL_DATA:-$RUNTIME/data/swe-smith-training/python-full-v7}"
source "$RUNTIME/admin/activate-agent-lightning-d1.sh"
export PYTHONPATH="$TOOLS:$(cd "$TOOLS/../.." && pwd):${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES= python "$TOOLS/check_full_python_ready.py" --data "$DATA" --audit "$AGL_FULL_ENV_AUDIT"
SCHEDULE=$(CUDA_VISIBLE_DEVICES= python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads((Path(sys.argv[1]) / 'manifest.json').read_text())
batches = (manifest['counts']['train'] + 31) // 32
assert batches * 4 == manifest['steps_for_four_complete_epochs']
print(batches, batches * 4)
PY
)
read -r EPOCH_STEPS TOTAL_STEPS <<< "$SCHEDULE"

# Direct full run. The smoke wrappers are intentionally excluded here.
bash "$TOOLS/run_training.sh" \
  --train-file "$DATA/train.jsonl" --val-file "$DATA/val.jsonl" \
  trainer.n_gpus_per_node=4 ray_kwargs.ray_init.num_gpus=4 ray_kwargs.ray_init.num_cpus=16 \
  trainer.total_epochs=4 trainer.total_training_steps=null \
  data.train_batch_size=32 data.val_batch_size="${AGL_VAL_BATCH_SIZE:-470}" \
  agentlightning.full_dataset=true agentlightning.audit_every_n_steps=32 \
  agentlightning.local.agent_class=full_python_agent.FullPythonAgent agentlightning.rollout_timeout_seconds=3600 \
  agentlightning.multi_turn_ppo.backend=capo agentlightning.multi_turn_ppo.capo_strict_padding=true \
  agentlightning.multi_turn_ppo.distributed_padding=false agentlightning.multi_turn_ppo.whiten_advantages=true \
  algorithm.adv_estimator=token_gae algorithm.gamma=1.0 algorithm.lam=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 critic.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  critic.ppo_micro_batch_size_per_gpu=2 critic.forward_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 actor_rollout_ref.actor.optim.weight_decay=0.01 critic.optim.weight_decay=0.01 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.actor.clip_ratio_c=3 actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.calculate_entropy=false actor_rollout_ref.actor.ppo_epochs=1 critic.ppo_epochs=1 \
  trainer.use_legacy_worker_impl=enable trainer.critic_warmup=0 \
  actor_rollout_ref.model.use_fused_kernels=true actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 actor_rollout_ref.rollout.val_kwargs.top_p=0.8 actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  actor_rollout_ref.rollout.max_num_seqs=8 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' trainer.save_freq=40 trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=3 trainer.max_critic_ckpt_to_keep=3 trainer.val_before_train=true trainer.resume_mode=disable \
  agentlightning.privileged_critic.enabled=true agentlightning.privileged_critic.state_schema_version=1 \
  agentlightning.privileged_critic.encoding=semantic_hunks_v2 agentlightning.privileged_critic.max_tokens=4096 \
  agentlightning.privileged_critic.safety_margin=32 "$@"

RUN="$RUNTIME/logs/training-$AGL_TRAIN_TAG"
CUDA_VISIBLE_DEVICES= python -m verl.model_merger merge --backend fsdp \
  --local_dir "$RUN/checkpoints/global_step_$TOTAL_STEPS/actor" --target_dir "$RUN/final-actor" \
  >"$RUN/final-export.log" 2>&1
