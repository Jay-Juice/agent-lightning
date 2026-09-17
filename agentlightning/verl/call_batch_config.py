# Copyright (c) Microsoft. All rights reserved.

"""Adapt veRL's one-row-per-task batch check to CAPO's variable call rows."""

import copy


def validate_worker_config(config, *, use_reference_policy, use_critic):
    from verl.trainer.main_ppo import validate_config

    options = config.agentlightning.get("multi_turn_ppo", {})
    call_rows = options.get("enabled", False) and options.get("backend") == "capo"
    if not call_rows:
        return validate_config(config, use_reference_policy, use_critic)

    from .capo_ppo import validate_config as validate_capo

    validate_capo(config)
    tasks = config.data.train_batch_size
    world = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if tasks <= 0 or tasks % world:
        raise ValueError("CAPO task batch must be positive and divisible by world size")
    actor_mini = config.actor_rollout_ref.actor.ppo_mini_batch_size
    critic_mini = config.critic.ppo_mini_batch_size if use_critic else 0
    if max(actor_mini, critic_mini) <= tasks:
        return validate_config(config, use_reference_policy, use_critic)
    if not options.get("capo_strict_padding", False):
        raise ValueError("Variable/short call minibatches require strict padding normalization")
    # Upstream ActorConfig/CriticConfig.validate assume one response row per
    # sampled task. CAPO expands tasks to variable call rows and already handles
    # a short final minibatch. Only the validation copy uses a row capacity;
    # the actual dataloader, rollout count, worker config, and saved config stay
    # unchanged. Preserve all other upstream validation and task divisibility.
    validation_copy = copy.deepcopy(config)
    validation_copy.data.train_batch_size = max(tasks, actor_mini, critic_mini)
    result = validate_config(validation_copy, use_reference_policy, use_critic)
    print(f"CAPO_CALL_BATCH_CONFIG task_batch={tasks} actor_call_minibatch={actor_mini} "
          f"critic_call_minibatch={critic_mini}; upstream row-capacity check uses a validation-only copy", flush=True)
    return result
