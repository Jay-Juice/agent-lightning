# Copyright (c) Microsoft. All rights reserved.

"""AgentLightningRayPPOTrainer drives VERL rollouts through Agent Lightning."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any, TypeVar

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,  # pyright: ignore[reportPrivateImportUsage]
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import marked_timer
from verl.utils.ray_utils import auto_await
from verl.utils.tracking import Tracking

from agentlightning.client import AgentLightningSyncClient
from agentlightning.hooks import RolloutHooks, load_hooks

from . import multi_turn_ppo
from .agl_rollout_manager import (
    AglAsyncRolloutManager,
    AglRolloutManager,
    AglRolloutManagerBase,
    CompletedRollout,
    EnqueuedRollout,
)
from .distributed_ppo import pad_inference, pad_update, unpad_inference
from .per_rollout_loss import (
    PER_ROLLOUT_MEAN_LOSS_MODE,
    normalize_advantages_by_rollout,
)
from .rollout_adapter import RolloutAdapter
from .rollout_level_advantage import compute_rollout_level_advantage

log = logging.getLogger(__name__)

RolloutManagerT = TypeVar("RolloutManagerT", bound=AglRolloutManagerBase)


def _batch_dict_len(batch: dict[str, Any] | None) -> int:
    """Leading-dim length of a dataloader batch dict (0 if absent/empty)."""
    if batch is None or not batch:
        return 0
    return len(next(iter(batch.values())))


def _grpo_group_metrics(batch: Any) -> dict[str, int]:
    """Count GRPO groups and how many have zero intra-group reward variance."""
    uids = batch.non_tensor_batch.get("uid")
    scores = batch.batch.get("token_level_scores")
    if uids is None or scores is None:
        return {}
    sequence_score = scores.sum(-1).detach().float().cpu()
    groups: dict[Any, list[float]] = defaultdict(list)
    for uid, score in zip(uids, sequence_score.tolist(), strict=False):
        groups[uid].append(score)
    n_zero_adv = sum(1 for vals in groups.values() if max(vals) - min(vals) == 0.0)
    return {
        "training/n_groups": len(groups),
        "training/n_zero_adv_groups": n_zero_adv,
    }


def _same_reward_uid_indices(batch: DataProto) -> list[int]:
    if "uid" not in batch.non_tensor_batch:
        return []

    rewards = batch.batch["token_level_scores"].sum(dim=-1)
    uid_to_indices: dict[Any, list[int]] = {}
    for sample_idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid_to_indices.setdefault(uid, []).append(sample_idx)

    same_reward_indices: list[int] = []
    for indices in uid_to_indices.values():
        group_rewards = rewards[indices]
        if torch.allclose(group_rewards, group_rewards[0].expand_as(group_rewards)):
            same_reward_indices.extend(indices)

    return same_reward_indices


class AgentLightningRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that drives train and validation rollouts via Agent Lightning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_turn_ppo = self.config.agentlightning.get("multi_turn_ppo", {}).get("enabled", False)
        self.capo_ppo = (
            self.multi_turn_ppo and self.config.agentlightning.multi_turn_ppo.get("backend", "agl") == "capo"
        )
        self.distributed_ppo = self.config.agentlightning.get("multi_turn_ppo", {}).get("distributed_padding", False)
        if self.distributed_ppo and not self.multi_turn_ppo:
            raise ValueError("distributed_padding requires multi_turn_ppo.enabled=true")
        if self.multi_turn_ppo:
            multi_turn_ppo.validate_config(self.config)
        self.is_async = self.config.agentlightning.async_rollout.enabled
        self.epoch = 0
        self._rollout_weight_version = 0
        async_train_batch_size = self.config.agentlightning.async_rollout.async_train_batch_size
        train_batch_size = self.config.data.train_batch_size
        if self.is_async and async_train_batch_size <= train_batch_size:
            raise ValueError(
                f"async_train_batch_size ({async_train_batch_size}) must be > "
                f"data.train_batch_size ({train_batch_size})."
            )
        self._hooks: RolloutHooks | None = None
        self._agl_client: AgentLightningSyncClient | None = None
        self._carry_over_rollouts: list[EnqueuedRollout] = []
        self._train_dataloader_iter: Any | None = None
        self._behavior_initial_tokens: float | None = None
        self._update_phase = "complete"

    def _reliability_enabled(self):
        return bool(self.config.agentlightning.get("reliability", {}).get("enabled", False))

    def _save_checkpoint(self):
        lock_path = self.config.agentlightning.get("reliability", {}).get("checkpoint_lock_path")
        if self._reliability_enabled() and lock_path:
            # Concurrent four-GPU runs share the disk. Serialize large writes,
            # then check free space inside the lock, before removing anything.
            import fcntl

            with Path(lock_path).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    return self._save_checkpoint_unlocked()
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        return self._save_checkpoint_unlocked()

    def _save_checkpoint_unlocked(self):
        if self._reliability_enabled():
            if getattr(self, "_update_phase", "complete") != "complete":
                raise RuntimeError("Cannot checkpoint a partially updated PPO batch")
            root = Path(self.config.trainer.default_local_dir)
            root.mkdir(parents=True, exist_ok=True)
            reserve = float(self.config.agentlightning.reliability.checkpoint_reserve_gib)
            if reserve <= 0 or shutil.disk_usage(root).free < reserve * 2**30:
                raise RuntimeError(f"Checkpoint requires {reserve:g} GiB free before writing; no checkpoint deleted")
        result = super()._save_checkpoint()
        if self._reliability_enabled():
            folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            path = folder / "reliability-state.json"
            state = {"version": 1, "step": self.global_steps, "epoch": self.epoch,
                     "initial_output_tokens": self._behavior_initial_tokens}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state))
            temporary.replace(path)
            if self.config.agentlightning.reliability.get("retention_across_resume", False):
                from .checkpoint_retention import retain_complete_checkpoints

                report = retain_complete_checkpoints(
                    root, self.global_steps, self.resource_pool_manager.get_n_gpus(),
                    self.config.trainer.max_actor_ckpt_to_keep,
                )
                print(f"RELIABILITY_CHECKPOINT_RETENTION {json.dumps(report)}", flush=True)
        return result

    def _load_checkpoint(self):
        resolved = None
        original_mode = self.config.trainer.resume_mode
        original_path = self.config.trainer.get("resume_from_path")
        if self._reliability_enabled() and original_mode == "auto":
            # The upstream latest pointer can be written before our completion
            # marker. Ignore an interrupted write and recover a complete pair.
            root = Path(self.config.trainer.default_local_dir)
            folders = [p for p in root.glob("global_step_*") if p.name.removeprefix("global_step_").isdigit()]
            for folder in sorted(folders, key=lambda p: int(p.name.removeprefix("global_step_")), reverse=True):
                if ((folder / "reliability-state.json").is_file() and (folder / "data.pt").is_file()
                        and (folder / "actor").is_dir() and (not self.use_critic or (folder / "critic").is_dir())):
                    resolved = str(folder)
                    break
            if folders and resolved is None:
                raise RuntimeError("No complete reliable checkpoint; refusing an implicit restart")
            if resolved:
                self.config.trainer.resume_mode = "resume_path"
                self.config.trainer.resume_from_path = resolved
        try:
            result = super()._load_checkpoint()
        finally:
            if resolved:
                self.config.trainer.resume_mode = original_mode
                self.config.trainer.resume_from_path = original_path
        if self._reliability_enabled() and self.global_steps:
            root = resolved or self.config.trainer.resume_from_path or str(
                Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}")
            state_path = Path(root) / "reliability-state.json"
            if not state_path.exists():
                raise RuntimeError("Reliable runs must resume a checkpoint with matching reliability state")
            state = json.loads(state_path.read_text())
            if state["step"] != self.global_steps:
                raise RuntimeError("Reliability state/checkpoint step mismatch")
            if not (Path(root) / "data.pt").is_file():
                raise RuntimeError("Reliable checkpoint is missing dataloader state")
            if state.get("version") != 1:
                raise RuntimeError("Unsupported reliability checkpoint state")
            self._behavior_initial_tokens = state["initial_output_tokens"]
            self.epoch = int(state["epoch"])
        return result

    def _reliability_failure(self, reason, details=None):
        if not self._reliability_enabled():
            return
        root = Path(self.config.trainer.default_local_dir)
        root.mkdir(parents=True, exist_ok=True)
        payload = {"step": self.global_steps, "phase": getattr(self, "_update_phase", "complete"),
                   "reason": reason, "details": details}
        (root / f"reliability-failure-step-{self.global_steps}.json").write_text(json.dumps(payload, indent=2))

    def _persist_failed_rollout(self, rollout):
        # The legacy on_failed hook saves status only; retain raw events before
        # a failed infrastructure attempt is replaced by a new rollout.
        if rollout.rollout_state != "succeeded":
            root = Path(self.config.trainer.default_local_dir) / "episode-audit"
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{rollout.rollout_id}.failed-events.json").write_text(
                json.dumps({"rollout_id": rollout.rollout_id, "events": rollout.events})
            )

    def _check_behavior(self, metrics, *, initial=False):
        if not self._reliability_enabled():
            return
        from .reliability_control import behavior_violations

        if initial and self._behavior_initial_tokens is None:
            self._behavior_initial_tokens = metrics["val/behavior/output_tokens_per_episode/mean"]
        violations = behavior_violations(metrics, self._behavior_initial_tokens)
        if violations:
            self._reliability_failure("behavior_gate", violations)
            # Initial model has no completed training update to checkpoint.
            if not initial:
                self._save_checkpoint()
            raise RuntimeError(f"Validation behavior gate triggered: {violations}")

    def _ensure_hooks(self) -> RolloutHooks | None:
        if self._hooks is not None:
            return self._hooks
        hooks_path = self.config.agentlightning.hooks
        if not hooks_path:
            return None
        self._hooks = load_hooks(hooks_path)
        self._hooks.on_startup()
        return self._hooks

    def _ensure_agl_client(self) -> AgentLightningSyncClient:
        if self._agl_client is not None:
            return self._agl_client
        self._agl_client = AgentLightningSyncClient(
            base_url=self.config.agentlightning.agl_base_url,
            key=self.config.agentlightning.agl_key,
            timeout=300,
        )
        return self._agl_client

    def _make_rollout_manager(self, manager_cls: type[RolloutManagerT]) -> RolloutManagerT:
        al = self.config.agentlightning
        return manager_cls(
            agl_base_url=al.agl_base_url,
            agl_key=al.agl_key,
            model=self.config.actor_rollout_ref.model.path,
            step=self.global_steps,
            train_rollout_n=self.config.actor_rollout_ref.rollout.n,
            rollout_timeout_seconds=al.rollout_timeout_seconds,
            hooks=self._ensure_hooks(),
            local_agent_class=al.local.agent_class,
            local_env_map=al.local.env_map,
            k8s_job_template_path=al.k8s.job_template_path,
            preserve_model_calls=self.multi_turn_ppo,
        )

    def _rollout_replicas(self) -> list[Any]:
        if hasattr(self, "llm_server_manager"):
            return list(self.llm_server_manager.get_replicas())  # pyright: ignore[reportAttributeAccessIssue]
        return list(self.async_rollout_manager.rollout_replicas)  # pyright: ignore[reportAttributeAccessIssue]

    @auto_await
    async def _abort_all_rollout_requests(self) -> None:
        await asyncio.gather(*[replica.abort_all_requests() for replica in self._rollout_replicas()])

    @auto_await
    async def _resume_all_rollout_generation(self) -> None:
        await asyncio.gather(*[replica.resume_generation() for replica in self._rollout_replicas()])

    def _resume_gateway(self) -> None:
        # Resuming is idempotent and safe to retry.
        self._ensure_agl_client().post_with_retry("/proxy/resume")

    def _pause_and_drain_gateway(self, *, reason: str) -> dict[str, Any]:
        client = self._ensure_agl_client()

        # Pausing is idempotent; allow five minutes when in-flight requests saturate the server.
        response = client.post_with_retry("/proxy/pause", json={"reason": reason}, timeout=300.0)
        paused_payload = response.json()
        inflight_on_pause = int(paused_payload.get("inflight", 0))

        drain_started_at = time.perf_counter()
        residual = inflight_on_pause
        while True:
            response = client.get("/proxy/state")
            response.raise_for_status()
            residual = int(response.json().get("inflight", 0))
            if residual <= 0:
                break
            time.sleep(0.25)
        drain_seconds = time.perf_counter() - drain_started_at

        return {
            "training/async/proxy_inflight_at_pause": inflight_on_pause,
            "training/async/proxy_drain_seconds": drain_seconds,
        }

    def _compute_async_rollout_metrics(
        self,
        *,
        previous_carry_over_rollouts: list[EnqueuedRollout],
        completed_rollouts: list[CompletedRollout],
        new_carry_over_rollouts: list[EnqueuedRollout],
    ) -> dict[str, Any]:
        max_carry_over_age = max(
            (self.global_steps - rollout.step for rollout in new_carry_over_rollouts),
            default=0,
        )
        return {
            "training/async/n_prev_carry_over_rollouts": len(previous_carry_over_rollouts),
            "training/async/n_completed_rollouts": len(completed_rollouts),
            "training/async/n_new_carry_over_rollouts": len(new_carry_over_rollouts),
            "training/async/new_carry_over_age_max_steps": max_carry_over_age,
        }

    def _rollout_lifecycle_metrics(
        self,
        completed_rollouts: list[CompletedRollout],
    ) -> dict[str, Any]:
        """Per-rollout pod queue/run timing for the current step.

        Emits scalar aggregates of the queue wait / run duration / total so the
        "pods launched in batches" effect shows up as a trend curve. Pod startup
        is approximated by running_at - submitted (queue + init), which is
        exactly the wait we want to watch under CPU-limited batched launch.

        The per-rollout wandb Table (rollout_lifecycle/step_N) is disabled: its
        one-row-per-rollout payload was large and slow to log.
        """
        if not completed_rollouts:
            return {}

        queue_waits: list[float] = []
        run_durations: list[float] = []
        totals: list[float] = []
        n_missing_running = 0

        for rollout in completed_rollouts:
            submitted = rollout.enqueue_time
            running_at = rollout.running_at
            finished_at = rollout.finished_at
            queue_wait = (running_at - submitted) if running_at is not None else None
            run_duration = finished_at - running_at if (running_at is not None and finished_at is not None) else None
            total = (finished_at - submitted) if finished_at is not None else None
            if queue_wait is not None:
                queue_waits.append(queue_wait)
            else:
                n_missing_running += 1
            if run_duration is not None:
                run_durations.append(run_duration)
            if total is not None:
                totals.append(total)

        metrics: dict[str, Any] = {}
        # Keep scalar timings; per-rollout W&B tables are too large and slow.

        def _agg(prefix: str, values: list[float]) -> None:
            if not values:
                return
            arr = np.asarray(values, dtype=float)
            metrics[f"{prefix}/mean"] = float(arr.mean())
            metrics[f"{prefix}/p50"] = float(np.percentile(arr, 50))
            metrics[f"{prefix}/p90"] = float(np.percentile(arr, 90))
            metrics[f"{prefix}/max"] = float(arr.max())

        _agg("timing/rollout_queue_wait_s", queue_waits)
        _agg("timing/rollout_run_duration_s", run_durations)
        _agg("timing/rollout_total_s", totals)
        metrics["timing/rollout_n_missing_running_ts"] = n_missing_running
        return metrics

    def _next_train_batch_dict_for_rollout(self) -> dict[str, Any]:
        if not self.is_async:
            return self._next_train_batch_dict()

        async_cfg = self.config.agentlightning.async_rollout
        async_train_batch_size = async_cfg.async_train_batch_size
        n_carry_over = len({rollout.data_id for rollout in self._carry_over_rollouts})
        n_new = int(async_train_batch_size) - n_carry_over
        return self._next_train_batch_dict_with_size(n_new)

    def _next_train_batch_dict(self) -> dict[str, Any]:
        while True:
            if self._train_dataloader_iter is None:
                self._train_dataloader_iter = iter(self.train_dataloader)

            for batch_dict in self._train_dataloader_iter:
                return batch_dict

            self.epoch += 1
            self._train_dataloader_iter = None

    def _next_train_batch_dict_with_size(self, size: int) -> dict[str, Any]:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")

        def split_batch(batch: dict[str, Any], head_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
            head: dict[str, Any] = {}
            tail: dict[str, Any] = {}
            for key, value in batch.items():
                head[key] = value[:head_size]
                tail[key] = value[head_size:]
            return head, tail

        def concat_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
            if len(batches) == 1:
                return batches[0]

            output: dict[str, Any] = {}
            for key in batches[0]:
                values = [batch[key] for batch in batches]
                sample = values[0]
                if isinstance(sample, torch.Tensor):
                    output[key] = torch.cat(values, dim=0)
                elif isinstance(sample, np.ndarray):
                    output[key] = np.concatenate(values, axis=0)
                elif isinstance(sample, list):
                    merged: list[Any] = []
                    for value in values:
                        merged.extend(value)
                    output[key] = merged
                else:
                    raise TypeError(
                        f"unsupported batch value type for key {key!r}: {type(sample).__name__}. "
                        "Expected torch.Tensor, numpy.ndarray, or list."
                    )
            return output

        collected: list[dict[str, Any]] = []
        remaining = size
        buffered_batch = getattr(self, "_train_dataloader_buf", None)

        if buffered_batch is not None:
            buffered_size = _batch_dict_len(buffered_batch)
            if buffered_size <= remaining:
                collected.append(buffered_batch)
                remaining -= buffered_size
                self._train_dataloader_buf = None
            else:
                head, tail = split_batch(buffered_batch, remaining)
                collected.append(head)
                self._train_dataloader_buf = tail
                remaining = 0

        while remaining > 0:
            if self._train_dataloader_iter is None:
                self._train_dataloader_iter = iter(self.train_dataloader)

            try:
                batch_dict = next(self._train_dataloader_iter)
            except StopIteration:
                self.epoch += 1
                self._train_dataloader_iter = None
                continue

            batch_size = _batch_dict_len(batch_dict)
            if batch_size <= remaining:
                collected.append(batch_dict)
                remaining -= batch_size
            else:
                head, tail = split_batch(batch_dict, remaining)
                collected.append(head)
                self._train_dataloader_buf = tail
                remaining = 0

        return concat_batches(collected)

    def _rollout(self, gen_batch: DataProto, is_train: bool) -> tuple[DataProto, dict[str, Any]]:
        """Run Agent Lightning rollouts and return the resulting DataProto plus metrics."""
        # verl 0.8.0 moved rollout server state behind llm_server_manager.
        has_llm_server_manager = hasattr(self, "llm_server_manager")
        if has_llm_server_manager:
            server_addresses = list(self.llm_server_manager.get_addresses())  # pyright: ignore[reportAttributeAccessIssue]
        else:
            server_addresses = list(self.async_rollout_manager.server_addresses)  # pyright: ignore[reportAttributeAccessIssue]
        self._resume_all_rollout_generation()  # pyright: ignore[reportUnusedCoroutine]
        if self.is_async:
            self._resume_gateway()
        data_dict = dict(gen_batch.non_tensor_batch)

        async_rollout_metrics: dict[str, Any] = {}
        if self.is_async and is_train:
            rollout_manager = self._make_rollout_manager(AglAsyncRolloutManager)
            rollout_manager.delete_model()
            rollout_manager.register_model(server_addresses, version=self._rollout_weight_version)
            previous_carry_over_rollouts = list(self._carry_over_rollouts)
            completed_rollouts, new_carry_over_rollouts = rollout_manager.enqueue_and_wait_until_group_completed(
                data_dict,
                previous_carry_over_rollouts,
                is_train=True,
                target_finished_group_num=self.config.data.train_batch_size,
            )
            self._carry_over_rollouts = new_carry_over_rollouts
            async_rollout_metrics = self._compute_async_rollout_metrics(
                previous_carry_over_rollouts=previous_carry_over_rollouts,
                completed_rollouts=completed_rollouts,
                new_carry_over_rollouts=new_carry_over_rollouts,
            )
        else:
            rollout_manager = self._make_rollout_manager(AglRolloutManager)
            rollout_manager.delete_model()
            rollout_manager.register_model(server_addresses, version=self._rollout_weight_version)
            completed_rollouts = rollout_manager.enqueue_and_wait_until_completed(data_dict, is_train=is_train)

        reliability_metrics = {}
        if self._reliability_enabled():
            from .episode_contract import retryable_infrastructure, validate_completed
            from .reliability_metrics import rollout_diagnostics

            retries = int(self.config.agentlightning.reliability.infrastructure_retries)
            if (self.is_async or self.config.actor_rollout_ref.rollout.n != 1
                    or self.config.actor_rollout_ref.rollout.val_kwargs.n != 1 or retries not in (0, 1)):
                raise ValueError("SWE reliability requires synchronous n=1 and at most one infrastructure retry")
            retry_count = 0
            retried_rollouts = []
            for index, rollout in enumerate(completed_rollouts):
                self._persist_failed_rollout(rollout)
                if retries and retryable_infrastructure(rollout):
                    # Re-run the same task at the same registered policy version.
                    # Never retry an ordinary score=0 or unresolved grader outcome.
                    retried_rollouts.append(rollout.rollout_id)
                    sample = rollout.sample_idx_in_step
                    retry_data = {key: value[sample : sample + 1] for key, value in data_dict.items()}
                    replacements = rollout_manager.enqueue_and_wait_until_completed(retry_data, is_train=is_train)
                    if len(replacements) != 1 or replacements[0].input != rollout.input:
                        raise RuntimeError("Infrastructure retry returned a different task")
                    self._persist_failed_rollout(replacements[0])
                    completed_rollouts[index] = replacements[0].model_copy(update={
                        "data_id": rollout.data_id, "sample_idx_in_step": rollout.sample_idx_in_step,
                    })
                    retry_count += 1
            prefix = "training" if is_train else "val"
            reliability_metrics.update(rollout_diagnostics(completed_rollouts, prefix=f"{prefix}/behavior"))
            reliability_metrics[f"{prefix}/infrastructure_retries"] = retry_count
            expected = len(next(iter(data_dict.values())))
            if len(completed_rollouts) != expected:
                raise RuntimeError("Incomplete rollout coverage")
            originals = completed_rollouts
            from .episode_contract import outcome

            journal = Path(self.config.trainer.default_local_dir) / "episode-audit"
            journal.mkdir(parents=True, exist_ok=True)
            records = [{"rollout_id": r.rollout_id, "sample_idx": r.sample_idx_in_step,
                        "reward": r.final_reward, "state": r.rollout_state, "outcome": outcome(r)} for r in originals]
            (journal / f"{prefix}-step-{self.global_steps}-{uuid.uuid4().hex}.json").write_text(json.dumps({
                "policy_version": self._rollout_weight_version, "retried_rollout_ids": retried_rollouts,
                "episodes": records,
            }, indent=2))
            try:
                completed_rollouts = [validate_completed(r, self._rollout_weight_version) for r in originals]
            except Exception as exc:
                self._reliability_failure("episode_contract", {
                    "error": str(exc), "metrics": reliability_metrics,
                    "rollout_ids": [r.rollout_id for r in originals],
                })
                raise

        trace_aggregator = self.config.agentlightning.trace_aggregator
        level = trace_aggregator.get("level", "transition")
        max_prompt_length = (
            trace_aggregator.get("trajectory_max_prompt_length", self.config.data.max_prompt_length)
            if str(level).startswith("trajectory")
            else self.config.data.max_prompt_length
        )
        max_response_length = (
            trace_aggregator.get("trajectory_max_response_length", self.config.data.max_response_length)
            if str(level).startswith("trajectory")
            else self.config.data.max_response_length
        )
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        rollout_adapter = RolloutAdapter(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            device=torch.device("cpu"),
            pad_token_id=pad_token_id,
            reward_fillna_value=self.config.agentlightning.reward_fillna_value,
            trace_aggregator_level=level,
            tokenizer=self.tokenizer,
            # [multimodal-patch] RayPPOTrainer stores the processor from entrypoint; forwarding it
            # enables pixel_values + mrope position ids for image-bearing training rows.
            processor=getattr(self, "processor", None),
        )

        if is_train:
            if self.multi_turn_ppo:
                out, metrics = multi_turn_ppo.build_batch(
                    rollout_adapter, completed_rollouts, global_steps=self.global_steps
                )
            else:
                out, metrics = rollout_adapter.get_train_data_batch(completed_rollouts, global_steps=self.global_steps)
            metrics.update(async_rollout_metrics)
            metrics.update(self._rollout_lifecycle_metrics(completed_rollouts))
        else:
            metrics = rollout_adapter.get_test_metrics(completed_rollouts, global_steps=self.global_steps)
            out = DataProto(batch=None)

        if self.is_async:
            print("AgentLightningRayPPOTrainer: pausing and draining agl gateway.")
            metrics.update(self._pause_and_drain_gateway(reason=f"rollout_done step={self.global_steps}"))
            print("AgentLightningRayPPOTrainer: agl gateway paused and drained.")
        else:
            print("AgentLightningRayPPOTrainer: aborting residual vLLM requests.")
            self._abort_all_rollout_requests()  # pyright: ignore[reportUnusedCoroutine]
            print("AgentLightningRayPPOTrainer: residual vLLM requests aborted.")
        metrics.update(reliability_metrics)
        return out, metrics

    def _train_step(
        self,
        timing_raw: dict[str, float],
        curr_step_profile: bool,
    ) -> tuple[dict[str, Any], DataProto] | None:
        metrics: dict[str, Any] = {}
        self._step_start_wall = time.time()
        metrics["timing/step_start_wall"] = self._step_start_wall
        rollout_n = self.config.actor_rollout_ref.rollout.n

        batch_dict = self._next_train_batch_dict_for_rollout()

        expected = self.config.agentlightning.get("reliability", {}).get("resume_expected_data_ids")
        if expected is not None and self.global_steps == self.config.agentlightning.reliability.resume_expected_step:
            actual = [str(value) for value in batch_dict["data_id"]]
            if actual != list(expected):
                raise RuntimeError("Resumed dataloader next batch does not match saved sampler state")
            if self._rollout_weight_version != self.global_steps - 1:
                raise RuntimeError("Resumed rollout weights do not match the checkpoint step")
            print(f"RELIABILITY_RESUME_NEXT_BATCH_OK step={self.global_steps} "
                  f"rollout_weight_version={self._rollout_weight_version} data_ids={json.dumps(actual)}", flush=True)

        batch: DataProto = DataProto.from_single_dict(batch_dict)

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        # verl 0.8.0 moved rollout profiling behind llm_server_manager.
        has_llm_server_manager = hasattr(self, "llm_server_manager")

        with marked_timer("gen", timing_raw, color="red"):
            if curr_step_profile:
                if has_llm_server_manager:
                    self.llm_server_manager.start_profile()  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    self.async_rollout_manager.start_profile()  # pyright: ignore[reportAttributeAccessIssue]

            gen_batch_output, agent_metrics = self._rollout(gen_batch, is_train=True)
            if curr_step_profile:
                if has_llm_server_manager:
                    self.llm_server_manager.stop_profile()  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    self.async_rollout_manager.stop_profile()  # pyright: ignore[reportAttributeAccessIssue]
            metrics.update(agent_metrics)
        metrics["timing/rollout_phase_end_wall"] = time.time()

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX baseline not yet supported in AgentLightningRayPPOTrainer")

        batch = gen_batch_output
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "data_id_list" in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]
        else:
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        assert "token_level_scores" in batch.batch, "rollout bridge must populate token_level_scores"
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        metrics["training/n_sample_collected"] = len(batch)
        if self.multi_turn_ppo:
            multi_turn_ppo.validate_batch_size(batch, self.config)
            metrics["training/n_sample_trained"] = len(batch)
        else:
            if "is_drop_mask" in batch.batch:
                keep = (~batch.batch["is_drop_mask"].bool()).nonzero(as_tuple=True)[0].tolist()
                metrics["training/n_sample_dropped/marked"] = len(batch) - len(keep)
                batch = batch[keep]  # pyright: ignore[reportAssignmentType]

            mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            n_transition = len(batch)
            max_ppo_update_times = self.config.agentlightning.get("max_ppo_update_times", None)
            n_remained_transition = n_transition // mini_bs * mini_bs
            if max_ppo_update_times is not None:
                n_remained_transition = min(n_remained_transition, mini_bs * max_ppo_update_times)

            n_to_drop = n_transition - n_remained_transition
            n_dropped_same_reward = 0
            n_dropped_random = 0
            if n_to_drop > 0:
                same_reward_indices = _same_reward_uid_indices(batch)
                random.shuffle(same_reward_indices)
                same_reward_drop_indices = same_reward_indices[:n_to_drop]
                same_reward_drop_set = set(same_reward_drop_indices)
                n_dropped_same_reward = len(same_reward_drop_indices)

                n_random_to_drop = n_to_drop - n_dropped_same_reward
                random_drop_indices: list[int] = []
                if n_random_to_drop > 0:
                    random_candidates = [
                        sample_idx for sample_idx in range(n_transition) if sample_idx not in same_reward_drop_set
                    ]
                    random.shuffle(random_candidates)
                    random_drop_indices = random_candidates[:n_random_to_drop]
                    n_dropped_random = len(random_drop_indices)

                drop_indices = same_reward_drop_set | set(random_drop_indices)
                keep_indices = [sample_idx for sample_idx in range(n_transition) if sample_idx not in drop_indices]
                batch = batch[keep_indices]  # pyright: ignore[reportAssignmentType]
            metrics["training/n_sample_dropped/same_reward"] = n_dropped_same_reward
            metrics["training/n_sample_dropped/random"] = n_dropped_random
            metrics["training/n_sample_trained"] = len(batch)
            if len(batch) == 0:
                print("WARNING: no trainable batch after drop+floor; skipping this training step.")
                return None

        print("AgentLightningRayPPOTrainer: sleeping rollout replicas.")
        self.checkpoint_manager.sleep_replicas()  # pyright: ignore[reportOptionalMemberAccess, reportUnusedCoroutine]
        print("AgentLightningRayPPOTrainer: rollout replicas slept.")

        if self.capo_ppo:
            from .capo_ppo import prepare_batch

            batch = prepare_batch(self, batch)
            metrics["capo/padding_rows"] = int(batch.non_tensor_batch["is_pad"].sum())

        if self.config.trainer.balance_batch:
            if self.distributed_ppo:
                batch = pad_inference(batch, self.resource_pool_manager.get_n_gpus())
            self._balance_batch(batch, metrics=metrics)
        elif self.distributed_ppo:
            batch = pad_inference(batch, self.resource_pool_manager.get_n_gpus())

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_mode = bool(rollout_corr_config and rollout_corr_config.get("bypass_mode", False))

        if bypass_mode:
            if "rollout_log_probs" not in batch.batch:
                raise RuntimeError("bypass_mode requires rollout_log_probs in batch")
            if not torch.isfinite(batch.batch["rollout_log_probs"]).all():
                raise RuntimeError("bypass_mode requires finite rollout_log_probs everywhere")
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                apply_bypass_mode(
                    batch,
                    rollout_corr_config,
                    self.config.actor_rollout_ref.actor.policy_loss,
                )
        else:
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                metrics["perf/mfu/actor_infer"] = old_log_prob_mfu
                if "entropys" in old_log_prob.batch:
                    old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self._compute_values(batch)
                batch = batch.union(values)

        if self.distributed_ppo:
            batch = unpad_inference(batch)

        with marked_timer("adv", timing_raw, color="brown"):
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,  # pyright: ignore[reportArgumentType]
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            if rollout_corr_config is not None and not bypass_mode and "rollout_log_probs" in batch.batch:
                from verl.trainer.ppo.rollout_corr_helper import (
                    compute_rollout_correction_and_add_to_batch,
                )

                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                metrics.update(is_metrics)

            adv_kwargs = {
                "adv_estimator": self.config.algorithm.adv_estimator,
                "gamma": self.config.algorithm.gamma,
                "lam": self.config.algorithm.lam,
                "num_repeat": rollout_n,
                "norm_adv_by_std_in_grpo": self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                "config": self.config.algorithm,
            }
            if self.multi_turn_ppo:
                if self.capo_ppo:
                    from .capo_ppo import compute_advantage as compute_capo_advantage

                    batch = compute_capo_advantage(batch, self.config.algorithm.gamma, self.config.algorithm.lam)
                else:
                    batch = multi_turn_ppo.compute_advantage(
                        batch,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        whiten=self.config.agentlightning.multi_turn_ppo.whiten_advantages,
                    )
                action_mask = batch.batch["response_mask"].bool()
                if self.capo_ppo:
                    action_mask[batch.non_tensor_batch["is_pad"]] = False
                raw_std = batch.batch["raw_advantages"][action_mask].float().std(correction=0)
                actor_std = batch.batch["advantages"][action_mask].float().std(correction=0)
                metrics["ppo/raw_advantage_std"] = raw_std.item()
                metrics["ppo/actor_advantage_std"] = actor_std.item()
                metrics["ppo/advantage_scale"] = (actor_std / raw_std.clamp_min(1e-8)).item()
                multi_turn_ppo.save_audit(
                    batch,
                    self.config.agentlightning.multi_turn_ppo.audit_dir,
                    self.global_steps,
                )
            elif self.config.algorithm.get("enable_rollout_level_advantage", False):
                batch, rollout_adv_metrics = compute_rollout_level_advantage(batch, **adv_kwargs)
                metrics.update(rollout_adv_metrics)
            else:
                batch = compute_advantage(batch, **adv_kwargs)

        # Count GRPO groups and zero-advantage groups in the trained batch.
        if not self.multi_turn_ppo:
            metrics.update(_grpo_group_metrics(batch))
        metrics["critic/n_transition_after_dropping"] = len(batch)
        if self._reliability_enabled():
            from .reliability_metrics import batch_diagnostics

            valid = ~batch.non_tensor_batch.get("is_pad", np.zeros(len(batch), dtype=bool))
            real_batch = batch.select_idxs(np.flatnonzero(valid))
            metrics.update(compute_data_metrics(batch=real_batch, use_critic=self.use_critic))
            metrics.update(batch_diagnostics(batch))
        else:
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == PER_ROLLOUT_MEAN_LOSS_MODE:
            rollout_ids = batch.non_tensor_batch.get("rollout_id_list")
            if rollout_ids is None:
                raise RuntimeError("per_rollout_mean loss requires rollout_id_list")
            batch.batch["advantages"] = normalize_advantages_by_rollout(
                batch.batch["advantages"],
                batch.batch["response_mask"],
                rollout_ids,
                num_trained_rows=len(batch),
            )

        update_batch = batch
        if self.distributed_ppo:
            update_batch, padding_count = pad_update(batch, self.resource_pool_manager.get_n_gpus())
            metrics["ppo/update_padding_rows"] = padding_count
            metrics["ppo/update_physical_rows"] = len(update_batch)

        if self.use_critic:
            self._update_phase = "critic_update"
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self._update_critic(update_batch)
            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

        if self.multi_turn_ppo:
            metrics["ppo/actor_updated"] = int(self.config.trainer.critic_warmup <= self.global_steps)
        if self.config.trainer.critic_warmup <= self.global_steps:
            self._update_phase = "actor_update"
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(update_batch)
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        self._update_phase = "weight_sync"
        with marked_timer("update_weights", timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)  # pyright: ignore[reportOptionalMemberAccess, reportUnusedCoroutine]
            self._rollout_weight_version = self.global_steps

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        self._update_phase = "complete"
        # Return the batch so fit() can compute throughput after the step timer closes.
        return metrics, batch

    def fit(self):
        """Training loop driven by AglRolloutManager for rollouts."""

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.epoch = 0
        self._train_dataloader_iter = None
        self._carry_over_rollouts = []
        self._load_checkpoint()
        # Push loaded weights before the first rollout or validation.
        self.checkpoint_manager.update_weights(self.global_steps)  # pyright: ignore[reportOptionalMemberAccess, reportUnusedCoroutine]
        self._rollout_weight_version = self.global_steps

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            self._check_behavior(val_metrics, initial=True)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        self.global_steps += 1
        if not self._reliability_enabled() or self.epoch == 0:
            self.epoch += 1
        last_val_metrics = None

        while True:
            if self.global_steps > self.total_training_steps or self.epoch > self.config.trainer.total_epochs:
                progress_bar.close()
                return

            timing_raw: dict[str, float] = {}

            curr_step_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )

            self._update_phase = "sampling"
            try:
                with marked_timer("step", timing_raw):
                    result = self._train_step(timing_raw, curr_step_profile)
            except Exception as exc:
                # Critic or actor may already have taken optimizer steps. Never
                # publish that partial state as a completed recovery point.
                self._reliability_failure("train_step_failed", {"error_type": type(exc).__name__})
                raise
            if result is None:
                if self._reliability_enabled():
                    raise RuntimeError("Reliable training cannot silently skip a task batch")
                print("AgentLightningRayPPOTrainer: train step returned no batch; advancing step.")
                self.global_steps += 1
                continue
            metrics, step_batch = result

            # Compute timing and throughput after the step timer records its total duration.
            metrics.update(compute_timing_metrics(batch=step_batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            if n_gpus > 0 and "step" in timing_raw:
                metrics.update(compute_throughout_metrics(batch=step_batch, timing_raw=timing_raw, n_gpus=n_gpus))

            is_last_step = self.global_steps >= self.total_training_steps

            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("validate", timing_raw, color="green"):
                    try:
                        val_metrics = self._validate()
                    except Exception:
                        # A complete validation score must never contain a
                        # missing reward. Preserve the just-updated actor,
                        # critic, optimizer, RNG and dataloader state before
                        # surfacing the failure so recovery does not roll back
                        # to an older periodic checkpoint.
                        print(
                            f"Validation failed at step {self.global_steps}; saving a recovery checkpoint before exit.",
                            flush=True,
                        )
                        self._save_checkpoint()
                        raise
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)
                logger.log(data=metrics, step=self.global_steps)
                self._check_behavior(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw):
                    self._save_checkpoint()

            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1

    def _validate(self, merged: bool = False):
        """Run validation through the Agent Lightning rollout manager."""
        # verl 0.8.0 moved rollout server state behind llm_server_manager.
        has_llm_server_manager = hasattr(self, "llm_server_manager")
        if has_llm_server_manager:
            server_addresses = list(self.llm_server_manager.get_addresses())  # pyright: ignore[reportAttributeAccessIssue]
        else:
            server_addresses = list(self.async_rollout_manager.server_addresses)  # pyright: ignore[reportAttributeAccessIssue]
        assert server_addresses, "_validate called before rollout server addresses are available"

        merged_metrics: dict[str, Any] = {}
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            if val_n > 1:
                test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                "validate": True,
                "global_steps": self.global_steps,
            }

            _, val_metrics_step = self._rollout(test_gen_batch, is_train=False)
            for k, v in val_metrics_step.items():
                merged_metrics[k] = v

        return merged_metrics
