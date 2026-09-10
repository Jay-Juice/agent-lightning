# Copyright (c) Microsoft. All rights reserved.

"""Thin Agent Lightning bindings for the vendored CAPO token PPO baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .vendor.capo import trajectory


def verify_vendor():
    root = Path(__file__).with_name("vendor") / "capo"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        actual = hashlib.sha256((root / name).read_text().encode()).hexdigest()
        if actual != entry["copied_sha256_lf"]:
            raise RuntimeError(f"CAPO vendor source changed: {name}")
    return manifest


def validate_config(config):
    if config.algorithm.adv_estimator != "token_gae" or not config.agentlightning.multi_turn_ppo.whiten_advantages:
        raise ValueError("CAPO PPO requires its original token_gae and advantage whitening")
    if config.agentlightning.multi_turn_ppo.distributed_padding:
        raise ValueError("CAPO PPO uses its copied padding behavior, not AGL distributed_padding")
    if config.trainer.use_legacy_worker_impl not in {"auto", "enable"}:
        raise ValueError("CAPO PPO requires legacy FSDP workers")
    world = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if config.actor_rollout_ref.rollout.n != 1:
        raise ValueError("The initial CAPO SWE port uses rollout.n=1")
    for worker in (config.actor_rollout_ref.actor, config.critic):
        if (
            worker.strategy != "fsdp"
            or worker.use_dynamic_bsz
            or worker.ppo_mini_batch_size % world
            or worker.ppo_mini_batch_size < world
            or worker.ppo_micro_batch_size_per_gpu != 1
            or worker.get("ulysses_sequence_parallel_size", 1) != 1
            or worker.loss_agg_mode != "seq-mean-token-mean"
        ):
            raise ValueError(
                "CAPO port requires FSDP, microbatch=1, fixed batches divisible by world, and call-mean loss"
            )


def prepare_batch(trainer, batch):
    batch.non_tensor_batch["trajectory_uids"] = batch.non_tensor_batch["rollout_id_list"].copy()
    batch.non_tensor_batch["step_indices"] = batch.non_tensor_batch["turn_index_list"].copy()
    # This is the copied CAPO method, including its is_pad-only marking.
    result = trajectory._pad_dataproto_to_world_size(trainer, batch)
    result.meta_info["global_token_num"] = result.batch["attention_mask"].sum(-1).tolist()
    return result


def compute_advantage(batch, gamma, lam):
    batch = trajectory.compute_advantage(batch, adv_estimator="token_gae", gamma=gamma, lam=lam)
    # Diagnostic only: the copied targets above are passed to PPO without modification.
    batch.batch["raw_advantages"] = (batch.batch["returns"] - batch.batch["values"]) * batch.batch["response_mask"]
    batch.batch["raw_advantages"][batch.non_tensor_batch["is_pad"]] = 0
    return batch


def register_in_worker():
    """Bind copied update classes; adapt only the newer FSDP log-prob return API."""
    import verl.workers.actor as actor_api
    import verl.workers.critic as critic_api

    from .vendor.capo.dp_actor import DataParallelPPOActor
    from .vendor.capo.dp_critic import DataParallelPPOCritic

    verify_vendor()

    class CompatibleActor(DataParallelPPOActor):
        # CAPO returns a tuple; the installed FSDP wrapper requires a dictionary.
        def compute_log_prob(self, data, calculate_entropy=False):  # pyright: ignore[reportIncompatibleMethodOverride]
            log_probs, entropys = super().compute_log_prob(data, calculate_entropy=calculate_entropy)
            result = {"log_probs": log_probs}
            if calculate_entropy:
                result["entropys"] = entropys
            return result

    actor_api.DataParallelPPOActor = CompatibleActor
    critic_api.DataParallelPPOCritic = DataParallelPPOCritic
    print(
        "CAPO_PPO_WORKERS "
        + json.dumps(
            {
                "actor_class": DataParallelPPOActor.__module__,
                "critic_class": DataParallelPPOCritic.__module__,
                "actor_update_inherited": CompatibleActor.update_policy is DataParallelPPOActor.update_policy,
            }
        ),
        flush=True,
    )
