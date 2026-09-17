"""Task sampling counts must not change when validating expanded call batches."""

from copy import deepcopy

import pytest
from omegaconf import OmegaConf

pytest.importorskip("verl")

from agentlightning.verl.call_batch_config import validate_worker_config


def config():
    worker = {"strategy": "fsdp", "use_dynamic_bsz": False, "ppo_mini_batch_size": 128,
              "ppo_micro_batch_size_per_gpu": 2, "loss_agg_mode": "seq-mean-token-mean"}
    return OmegaConf.create({
        "agentlightning": {"multi_turn_ppo": {"enabled": True, "backend": "capo",
            "capo_strict_padding": True, "whiten_advantages": True, "distributed_padding": False}},
        "algorithm": {"adv_estimator": "token_gae"},
        "data": {"train_batch_size": 32},
        "trainer": {"n_gpus_per_node": 4, "nnodes": 1, "use_legacy_worker_impl": "enable"},
        "actor_rollout_ref": {"actor": worker, "rollout": {"n": 1}}, "critic": deepcopy(worker),
    })


def test_expanded_call_validation_does_not_mutate_task_or_worker_config(monkeypatch):
    import verl.trainer.main_ppo as upstream

    cfg = config()
    before = OmegaConf.to_container(cfg, resolve=True)
    observed = []

    def validate(candidate, use_reference_policy, use_critic):
        assert use_reference_policy and use_critic
        assert candidate.data.train_batch_size >= candidate.actor_rollout_ref.actor.ppo_mini_batch_size
        observed.append(candidate)

    monkeypatch.setattr(upstream, "validate_config", validate)
    validate_worker_config(cfg, use_reference_policy=True, use_critic=True)
    assert len(observed) == 1 and observed[0] is not cfg
    assert observed[0].data.train_batch_size == 128
    assert OmegaConf.to_container(cfg, resolve=True) == before


@pytest.mark.parametrize("field,value", [("task_batch", 31), ("padding", False), ("minibatch", 127)])
def test_invalid_task_divisibility_padding_and_worker_shapes_still_fail(field, value):
    cfg = config()
    if field == "task_batch":
        cfg.data.train_batch_size = value
    elif field == "padding":
        cfg.agentlightning.multi_turn_ppo.capo_strict_padding = value
    else:
        cfg.critic.ppo_mini_batch_size = value
    with pytest.raises(ValueError):
        validate_worker_config(cfg, use_reference_policy=True, use_critic=True)


def test_ordinary_batch_preserves_original_validation(monkeypatch):
    import verl.trainer.main_ppo as upstream

    cfg = config()
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 32
    cfg.critic.ppo_mini_batch_size = 32
    observed = []
    monkeypatch.setattr(upstream, "validate_config", lambda candidate, *args: observed.append(candidate))
    validate_worker_config(cfg, use_reference_policy=True, use_critic=True)
    assert observed == [cfg] and observed[0] is cfg
