# Copyright (c) Microsoft. All rights reserved.

"""Opt-in full-dataset batching, validation accounting and bounded tensor audits."""

from collections.abc import Sized

from omegaconf import open_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto

from .trainer import AgentLightningRayPPOTrainer


def merge_validation_metrics(batches):
    if not batches:
        raise ValueError("Validation produced no batches")
    counts = ("val/n_rollouts", "val/n_rollouts_w_trace", "val/n_rollouts_w_reward")
    means = {
        "val/reward": "val/n_rollouts",
        "val/mean_response_length_per_turn": "val/n_rollouts_w_trace",
        "val/mean_total_response_length_per_rollout": "val/n_rollouts_w_trace",
        "val/turn_count": "val/n_rollouts_w_trace",
    }
    merged = dict(batches[-1])
    for key in counts:
        merged[key] = sum(batch[key] for batch in batches)
    for key, weight in means.items():
        if merged[weight] <= 0:
            raise ValueError(f"Validation has no samples for {key}")
        merged[key] = sum(batch[key] * batch[weight] for batch in batches) / merged[weight]
    from .reliability_control import merge_behavior_metrics

    behavior = merge_behavior_metrics(batches)
    if behavior:
        merged = {key: value for key, value in merged.items() if not key.startswith("val/behavior/")}
        merged.update(behavior)
        merged["val/infrastructure_retries"] = sum(b.get("val/infrastructure_retries", 0) for b in batches)
    return merged


class FullDatasetRayPPOTrainer(AgentLightningRayPPOTrainer):
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        previous = self.train_dataloader
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.dataloader_num_workers,
            drop_last=False,
            collate_fn=previous.collate_fn,
            sampler=previous.sampler,
        )
        steps = self.config.trainer.total_training_steps
        if steps is None:
            steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.total_training_steps = steps
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = steps
            self.config.critic.optim.total_training_steps = steps
        print(f"Full dataset: {len(self.train_dataloader)} batches/epoch, retain tail, total steps={steps}")

    def _train_step(self, *args, **kwargs):
        interval = int(self.config.agentlightning.get("audit_every_n_steps", 32))
        if interval < 1:
            raise ValueError("audit_every_n_steps must be positive")
        options = self.config.agentlightning.multi_turn_ppo
        directory = options.audit_dir
        capture = self.global_steps in {1, self.total_training_steps} or self.global_steps % interval == 0
        try:
            if not capture:
                options.audit_dir = None
            return super()._train_step(*args, **kwargs)
        finally:
            options.audit_dir = directory

    def _validate(self, merged=False):
        if not isinstance(self.val_dataset, Sized):
            raise TypeError("Full validation requires a finite dataset with a known length")
        # Same validation rollout path as the parent. Aggregate every batch;
        # the pilot parent's overwrite only reported the last validation batch.
        metrics = []
        for test_data in self.val_dataloader:
            batch = DataProto.from_single_dict(test_data)
            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            if val_n > 1:
                batch = batch.repeat(repeat_times=val_n, interleave=True)
            generated = self._get_gen_batch(batch)
            generated.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                "validate": True,
                "global_steps": self.global_steps,
            }
            _, values = self._rollout(generated, is_train=False)
            if values["val/n_rollouts_w_reward"] != values["val/n_rollouts"]:
                raise RuntimeError("Validation has missing rewards; inspect failed rollouts instead of reporting zeros")
            metrics.append(values)
        result = merge_validation_metrics(metrics)
        expected = len(self.val_dataset) * self.config.actor_rollout_ref.rollout.val_kwargs.n
        if result["val/n_rollouts"] != expected:
            raise RuntimeError(f"Validation coverage {result['val/n_rollouts']} differs from {expected}")
        return result
