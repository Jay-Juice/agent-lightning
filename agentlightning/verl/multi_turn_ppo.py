# Copyright (c) Microsoft. All rights reserved.

"""Token GAE across complete, ordered model calls of one environment episode.

Each critic value is V(history before its response token). Tool observations
are part of the next prompt, never actions or discounting steps. The last
action of a completed episode has a zero bootstrap. Sampling cutoffs with an
unfinished episode are deliberately unsupported.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.utils.torch_functional import masked_whiten

from .agl_rollout_manager import CompletedRollout
from .rollout_adapter import RolloutAdapter


def validate_config(config: Any) -> None:
    """Reject combinations that silently change temporal PPO semantics."""
    if not config.agentlightning.multi_turn_ppo.enabled:
        if config.agentlightning.multi_turn_ppo.get("backend", "agl") == "capo":
            raise ValueError("CAPO backend requires multi_turn_ppo.enabled=true")
        if config.agentlightning.multi_turn_ppo.get("distributed_padding", False):
            raise ValueError("distributed_padding requires multi_turn_ppo.enabled=true")
        return
    backend = config.agentlightning.multi_turn_ppo.get("backend", "agl")
    if backend not in {"agl", "capo"}:
        raise ValueError(f"Unknown multi-turn PPO backend: {backend}")
    estimator = "token_gae" if backend == "capo" else "gae"
    if config.algorithm.adv_estimator != estimator or not config.critic.enable:
        raise ValueError(f"multi_turn_ppo backend={backend} requires adv_estimator={estimator} and critic.enable=true")
    if config.algorithm.enable_rollout_level_advantage:
        raise ValueError("multi_turn_ppo requires enable_rollout_level_advantage=false")
    if config.agentlightning.trace_aggregator.level != "transition":
        raise ValueError("multi_turn_ppo requires transition traces")
    if config.agentlightning.async_rollout.enabled:
        raise ValueError("multi_turn_ppo currently requires synchronous episode collection")
    if config.agentlightning.max_ppo_update_times is not None:
        raise ValueError("multi_turn_ppo retains every turn; max_ppo_update_times must be null")
    if config.actor_rollout_ref.actor.policy_loss.loss_mode != "vanilla":
        raise ValueError("The initial multi_turn_ppo baseline uses vanilla clipped PPO loss")
    if backend == "capo":
        from .capo_ppo import validate_config as validate_capo_config

        validate_capo_config(config)
    if config.agentlightning.multi_turn_ppo.get("distributed_padding", False):
        world = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.rollout.n != 1:
            raise ValueError("Padding-aware PPO currently requires rollout.n=1")
        if config.trainer.use_legacy_worker_impl not in {"auto", "enable"}:
            raise ValueError("Padding-aware PPO requires the legacy FSDP actor/critic workers")
        for worker in (config.actor_rollout_ref.actor, config.critic):
            if (
                worker.strategy != "fsdp"
                or worker.use_dynamic_bsz
                or worker.ppo_mini_batch_size != world
                or worker.ppo_micro_batch_size_per_gpu != 1
                or worker.get("ulysses_sequence_parallel_size", 1) != 1
                or worker.loss_agg_mode != "seq-mean-token-mean"
            ):
                raise ValueError(
                    "Padding-aware PPO requires FSDP, mini_batch_size=world_size, micro_batch=1, "
                    "sequence_parallel=1, dynamic_bsz=false, loss_agg_mode=seq-mean-token-mean"
                )


def build_batch(adapter: RolloutAdapter, rollouts: list[CompletedRollout], *, global_steps: int = 0):
    """Validate full episodes before adaptation; reward only their last token.

    Fail the step on invalid/incomplete traces instead of labelling infrastructure
    failures as reward zero, dropping turns, or training truncated responses.
    """
    if adapter.trace_aggregator_level != "transition":
        raise ValueError("multi-turn PPO needs one row per model call")
    by_id = {}
    for rollout in rollouts:
        if rollout.rollout_id in by_id:
            raise ValueError(f"Duplicate rollout {rollout.rollout_id}")
        by_id[rollout.rollout_id] = rollout
        if rollout.rollout_state != "succeeded" or rollout.error_message:
            raise ValueError(f"Incomplete episode {rollout.rollout_id}: {rollout.rollout_state}")
        if rollout.final_reward is None or not math.isfinite(rollout.final_reward):
            raise ValueError(f"Missing/nonfinite terminal reward: {rollout.rollout_id}")
        if not rollout.triplets:
            raise ValueError(f"Episode has no model actions: {rollout.rollout_id}")
        requests = [e for e in rollout.triplet_events if e["event_type"] == "model_request"]
        if requests and len(requests) != len(rollout.triplets):
            raise ValueError(f"Episode has missing/failed model calls: {rollout.rollout_id}")
        raw_requests = [e for e in rollout.events if e["event_type"] == "model_request"]
        if raw_requests and len(raw_requests) != len(rollout.triplets):
            raise ValueError("Model calls were deduplicated; deploy a server supporting triplet-preserve")
        for triplet in rollout.triplets:
            prompt, response = (
                triplet.prompt["token_ids"],
                triplet.response["token_ids"],
            )
            if not prompt or not response:
                raise ValueError(f"Empty prompt/action: {rollout.rollout_id}")
            if len(prompt) > adapter.max_prompt_length or len(response) > adapter.max_response_length:
                raise ValueError(f"Episode exceeds training token limits: {rollout.rollout_id}; increase limits")
    if not by_id:
        raise ValueError("No complete episodes collected")
    batch, metrics = adapter.get_train_data_batch(rollouts, global_steps=global_steps)
    if batch.batch["is_drop_mask"].any() or len(batch) != sum(len(r.triplets or []) for r in rollouts):
        raise RuntimeError("Adapter changed a complete episode")
    width = batch.batch["responses"].shape[-1]
    batch.batch["response_mask"] = batch.batch["attention_mask"][:, -width:].clone()
    scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    counts, terminals = [], []
    for row, (rid, turn) in enumerate(
        zip(
            batch.non_tensor_batch["rollout_id_list"],
            batch.non_tensor_batch["turn_index_list"],
            strict=True,
        )
    ):
        episode = by_id[rid]
        count = len(episode.triplets)
        terminal = int(turn) == count - 1
        counts.append(count)
        terminals.append(terminal)
        if terminal:
            final_token = batch.batch["response_mask"][row].nonzero(as_tuple=True)[0][-1]
            scores[row, final_token] = episode.final_reward
    batch.batch["token_level_scores"] = scores
    batch.non_tensor_batch["episode_turn_count"] = np.array(counts)
    batch.non_tensor_batch["episode_terminal"] = np.array(terminals)
    metrics.update(
        {
            "ppo/episodes": len(by_id),
            "ppo/model_calls": len(batch),
            "ppo/action_tokens": int(batch.batch["response_mask"].sum()),
        }
    )
    return batch, metrics


def validate_batch_size(batch: DataProto, config: Any) -> None:
    """No row flooring, biased reward-based dropping, or dummy-token padding."""
    if (
        config.agentlightning.multi_turn_ppo.get("distributed_padding", False)
        or config.agentlightning.multi_turn_ppo.get("backend", "agl") == "capo"
    ):
        return
    world = config.trainer.n_gpus_per_node * config.trainer.nnodes
    sizes = [
        world,
        config.actor_rollout_ref.actor.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n,
        config.critic.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n,
    ]
    if any(len(batch) % size for size in sizes):
        raise ValueError(
            f"Collected {len(batch)} turns, incompatible with DP/minibatch sizes {sizes}. "
            "Use one GPU, rollout.n=1 and actor/critic.ppo_mini_batch_size=1 for variable turns. "
            "No turns have been dropped or updated."
        )


@torch.no_grad()
def compute_advantage(batch: DataProto, *, gamma: float, lam: float, whiten: bool = True) -> DataProto:
    """GAE on generated tokens, ordered by (rollout_id, turn_index, token).

    Returns use unwhitened advantages + old values. Whitening affects only the
    actor targets and is over all real action tokens in the collected batch.
    Row order may change during load balancing; episode order may not.
    """
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0, 1]")
    mask = batch.batch["response_mask"].bool()
    values = batch.batch["values"].float()
    rewards = batch.batch["token_level_rewards"].float()
    if values.shape != mask.shape or rewards.shape != mask.shape:
        raise ValueError("Values, rewards and action mask must have identical shapes")
    if not torch.isfinite(values[mask]).all() or not torch.isfinite(rewards[mask]).all():
        raise ValueError("Nonfinite values/rewards on action tokens")
    groups = defaultdict(list)
    for row, rid in enumerate(batch.non_tensor_batch["rollout_id_list"]):
        groups[rid].append(row)
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    for rid, rows in groups.items():
        rows.sort(key=lambda row: int(batch.non_tensor_batch["turn_index_list"][row]))
        turns = [int(batch.non_tensor_batch["turn_index_list"][row]) for row in rows]
        counts = [int(batch.non_tensor_batch["episode_turn_count"][row]) for row in rows]
        terminals = [bool(batch.non_tensor_batch["episode_terminal"][row]) for row in rows]
        if turns != list(range(len(rows))) or counts != [len(rows)] * len(rows):
            raise ValueError(f"Missing/duplicate turns in episode {rid}")
        if terminals != [False] * (len(rows) - 1) + [True]:
            raise ValueError(f"Episode {rid} has no unique final terminal")
        next_value = values.new_zeros(())
        carry = values.new_zeros(())
        for row in reversed(rows):
            positions = mask[row].nonzero(as_tuple=True)[0].tolist()
            if not positions:
                raise ValueError(f"Empty action in episode {rid}")
            for pos in reversed(positions):
                delta = rewards[row, pos] + gamma * next_value - values[row, pos]
                carry = delta + gamma * lam * carry
                advantages[row, pos] = carry
                returns[row, pos] = carry + values[row, pos]
                next_value = values[row, pos]
    batch.batch["raw_advantages"] = advantages.clone()
    if whiten and int(mask.sum()) > 1:
        advantages = masked_whiten(advantages, mask)
    batch.batch["advantages"] = advantages.masked_fill(~mask, 0)
    batch.batch["returns"] = returns
    return batch


def save_audit(batch: DataProto, directory: str | None, step: int) -> None:
    """Optional exact pre-update tensors for reproducible PPO/PI comparisons."""
    if directory is None:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    keys = (
        "input_ids",
        "attention_mask",
        "responses",
        "response_mask",
        "token_level_scores",
        "token_level_rewards",
        "old_log_probs",
        "rollout_log_probs",
        "ref_log_prob",
        "values",
        "raw_advantages",
        "advantages",
        "returns",
        "position_ids",
    )
    torch.save(
        {k: batch.batch[k].detach().cpu() for k in keys if k in batch.batch},
        path / f"step-{step:06d}.pt",
    )
    metadata = {
        k: batch.non_tensor_batch[k].tolist()
        for k in (
            "rollout_id_list",
            "data_id_list",
            "turn_index_list",
            "episode_turn_count",
            "episode_terminal",
        )
    }
    if "is_pad" in batch.non_tensor_batch:
        metadata["is_pad"] = batch.non_tensor_batch["is_pad"].tolist()
    (path / f"step-{step:06d}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
