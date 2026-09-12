# Copyright (c) Microsoft. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from agentlightning.verl.full_dataset import (
    FullDatasetRayPPOTrainer,
    merge_validation_metrics,
)
from agentlightning.verl.trainer import AgentLightningRayPPOTrainer


def batch(n, reward, trace, length):
    return {
        "val/n_rollouts": n,
        "val/n_rollouts_w_trace": trace,
        "val/n_rollouts_w_reward": n,
        "val/reward": reward,
        "val/mean_response_length_per_turn": length,
        "val/mean_total_response_length_per_rollout": length * 3,
        "val/turn_count": 3,
    }


def test_validation_uses_all_samples_and_correct_denominators():
    result = merge_validation_metrics([batch(32, 0.5, 30, 10), batch(6, 0, 6, 40)])
    assert result["val/n_rollouts"] == 38
    assert result["val/reward"] == pytest.approx(16 / 38)
    assert result["val/mean_response_length_per_turn"] == pytest.approx(15)
    assert result["val/mean_total_response_length_per_rollout"] == pytest.approx(45)


def test_full_dataset_retains_tail_and_recomputes_optimizer_schedule(monkeypatch):
    train = list(range(67))
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 32, "dataloader_num_workers": 0},
            "trainer": {"total_epochs": 4, "total_training_steps": None},
            "actor_rollout_ref": {"actor": {"optim": {"total_training_steps": 8}}},
            "critic": {"optim": {"total_training_steps": 8}},
        }
    )
    worker = object.__new__(FullDatasetRayPPOTrainer)
    worker.config = config
    worker.train_dataset = train

    def original(self, *_):
        self.train_dataloader = StatefulDataLoader(train, batch_size=32, drop_last=True)

    monkeypatch.setattr(AgentLightningRayPPOTrainer, "_create_dataloader", original)
    FullDatasetRayPPOTrainer._create_dataloader(worker, train, [], None, None)
    groups = list(worker.train_dataloader)
    assert [len(group) for group in groups] == [32, 32, 3]
    assert [int(x) for group in groups for x in group] == train
    assert worker.total_training_steps == 12
    assert config.actor_rollout_ref.actor.optim.total_training_steps == 12
    assert config.critic.optim.total_training_steps == 12


def test_audit_sampling_preserves_optimizer_call_and_restores_config(monkeypatch):
    worker = object.__new__(FullDatasetRayPPOTrainer)
    worker.config = OmegaConf.create(
        {
            "agentlightning": {
                "audit_every_n_steps": 32,
                "multi_turn_ppo": {"audit_dir": "/audit"},
            }
        }
    )
    worker.total_training_steps = 792
    observed = []

    def train(self, *args, **kwargs):
        observed.append(self.config.agentlightning.multi_turn_ppo.audit_dir)
        return "updated"

    monkeypatch.setattr(AgentLightningRayPPOTrainer, "_train_step", train)
    for step in [1, 2, 32, 791, 792]:
        worker.global_steps = step
        assert worker._train_step() == "updated"
        assert worker.config.agentlightning.multi_turn_ppo.audit_dir == "/audit"
    assert observed == ["/audit", None, "/audit", None, "/audit"]


@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("missing_reward", [False, True])
def test_validation_visits_every_batch_including_tail(repeats, missing_reward):
    worker = object.__new__(FullDatasetRayPPOTrainer)
    worker.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "val_kwargs": {
                        "n": repeats,
                        "do_sample": True,
                        "temperature": 0.7,
                    }
                }
            }
        }
    )
    worker.val_dataset = list(range(38))
    worker.val_dataloader = [{"input_ids": torch.zeros(n, 2, dtype=torch.long)} for n in (32, 6)]
    worker.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    worker.global_steps = 0
    worker._get_gen_batch = lambda data: data
    visited = []

    def rollout(data, is_train):
        assert not is_train and data.meta_info["validate"]
        n = len(data)
        visited.append(n)
        metrics = batch(n, 0.5 if len(visited) == 1 else 0, n, 10)
        if missing_reward and len(visited) == 2:
            metrics["val/n_rollouts_w_reward"] -= 1
        return None, metrics

    worker._rollout = rollout
    if missing_reward:
        with pytest.raises(RuntimeError, match="missing rewards"):
            worker._validate()
        return
    result = worker._validate()
    assert visited == [32 * repeats, 6 * repeats]
    assert result["val/n_rollouts"] == 38 * repeats
    assert result["val/reward"] == pytest.approx(16 / 38)
