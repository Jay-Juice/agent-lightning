# Copyright (c) Microsoft. All rights reserved.

"""Check copied-source integrity, temporal binding, padding and worker APIs."""
# ruff: noqa: E402

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")
from verl import DataProto

from agentlightning.verl import capo_ppo
from agentlightning.verl.vendor.capo import dp_actor, dp_critic, trajectory, verl_core_algos


def test_vendor_sources_match_manifest():
    manifest = capo_ppo.verify_vendor()
    assert manifest["source_commit"] == "e8407baea32fb36191f029f0a4666bf681bdf1ca"
    for name in ("arft_core_algos.py", "verl_core_algos.py", "LICENSE"):
        entry = manifest["files"][name]
        assert entry["source_sha256_lf"] == entry["copied_sha256_lf"]
    assert dp_actor.get_policy_loss_fn("vanilla") is verl_core_algos.compute_policy_loss_vanilla
    assert dp_critic.core_algos is verl_core_algos


def test_capo_padding_preserves_its_original_training_semantics():
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(3, 2, dtype=torch.long),
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
            "response_mask": torch.ones(3, 2, dtype=torch.long),
            "values": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "token_level_rewards": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
        },
        non_tensors={"rollout_id_list": np.array(["a", "a", "b"]), "turn_index_list": np.array([0, 1, 0])},
    )
    trainer = SimpleNamespace(
        use_critic=True,
        critic_wg=SimpleNamespace(world_size=4),
        use_reference_policy=False,
        hybrid_engine=True,
        actor_rollout_wg=SimpleNamespace(world_size=4),
    )
    padded = capo_ppo.prepare_batch(trainer, batch)
    result = capo_ppo.compute_advantage(padded, gamma=0.99, lam=1.0)
    assert result.non_tensor_batch["is_pad"].tolist() == [False, False, False, True]
    # CAPO marks duplicates but DOES NOT clear their response masks. In its original
    # update loop the duplicate still contributes critic loss and reference KL.
    assert result.batch["response_mask"][3].tolist() == [1, 1]
    assert result.batch["advantages"][3].count_nonzero() == 0
    assert result.batch["returns"][3].count_nonzero() == 0
    torch.testing.assert_close(result.batch["returns"][:3], torch.tensor([[0.99**3, 0.99**2], [0.99, 1], [1.98, 2]]))
    valid, _ = trajectory.get_valid_data(result)
    assert len(valid) == 3


def test_worker_binding_preserves_copied_updates_and_adapts_only_return_type(monkeypatch):
    import verl.workers.actor as actor_api
    import verl.workers.critic as critic_api
    from verl.trainer.ppo import core_algos as installed_algos

    monkeypatch.setattr(actor_api, "DataParallelPPOActor", actor_api.DataParallelPPOActor)
    monkeypatch.setattr(critic_api, "DataParallelPPOCritic", critic_api.DataParallelPPOCritic)
    original_aggregate = installed_algos.agg_loss
    capo_ppo.register_in_worker()
    assert actor_api.DataParallelPPOActor.update_policy is dp_actor.DataParallelPPOActor.update_policy
    assert critic_api.DataParallelPPOCritic.update_critic is dp_critic.DataParallelPPOCritic.update_critic
    assert installed_algos.agg_loss is original_aggregate
    monkeypatch.setattr(
        dp_actor.DataParallelPPOActor, "compute_log_prob", lambda *a, **kw: (torch.ones(2), torch.zeros(2))
    )
    actor = object.__new__(actor_api.DataParallelPPOActor)
    assert actor.compute_log_prob(None, calculate_entropy=True).keys() == {"log_probs", "entropys"}
    assert actor.compute_log_prob(None, calculate_entropy=False).keys() == {"log_probs"}
