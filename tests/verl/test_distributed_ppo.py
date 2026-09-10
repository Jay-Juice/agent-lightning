# Copyright (c) Microsoft. All rights reserved.

"""Padding invariance and gradient equivalence of multi-rank PPO aggregation."""
# ruff: noqa: E402

from typing import cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")
from verl import DataProto

from agentlightning.verl import distributed_ppo as ppo


def batch(size=5):
    return DataProto.from_dict(
        tensors={
            "responses": torch.arange(size * 3).reshape(size, 3),
            "attention_mask": torch.ones(size, 6, dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0]] * size),
            "advantages": torch.ones(size, 3),
            "rollout_log_probs": torch.zeros(size, 3),
        },
        non_tensors={"row_id": np.arange(size)},
    )


def test_inference_padding_survives_balancing_without_entering_gae():
    original = batch()
    padded = ppo.pad_inference(original, 4)
    assert len(padded) == 8
    assert padded.batch["response_mask"].sum() == 16  # real inputs for inference
    restored = ppo.unpad_inference(cast(DataProto, padded[[7, 0, 6, 1, 5, 4, 2, 3]]))
    assert sorted(restored.non_tensor_batch["row_id"].tolist()) == list(range(5))
    assert len(restored) == 5 and "ppo_padding" not in restored.batch


@pytest.mark.parametrize("size", [1, 3, 4, 5, 9])
def test_update_padding_has_zero_weight_and_preserves_original(size):
    original = batch(size)
    padded, count = ppo.pad_update(original, 4)
    assert len(padded) % 4 == 0
    assert count == (-size) % 4
    assert torch.equal(padded.batch["response_mask"][:size], original.batch["response_mask"])
    assert padded.batch["response_mask"][size:].count_nonzero() == 0
    assert original.batch["response_mask"].sum() == size * 2
    assert "rollout_log_probs" not in padded.batch
    assert "rollout_log_probs" in original.batch


def test_four_rank_gradient_matches_mean_of_real_call_losses(monkeypatch):
    # Three real calls of different lengths and one dummy-only rank.
    masks = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.zeros(1, 3),
    ]
    coefficients = [
        torch.tensor([[1.0, 90.0, 90.0]]),
        torch.tensor([[2.0, 4.0, 90.0]]),
        torch.tensor([[3.0, 6.0, 9.0]]),
        torch.full((1, 3), 999.0),
    ]
    monkeypatch.setattr(ppo.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ppo.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(ppo.dist, "all_reduce", lambda count, **kwargs: count.fill_(3))
    losses, grads = [], []
    for mask, coeff in zip(masks, coefficients, strict=True):
        parameter = torch.tensor(2.0, requires_grad=True)
        loss = ppo.global_call_mean(parameter * coeff, mask)
        loss.backward()
        losses.append(loss.item())
        grads.append(parameter.grad.item())
    assert sum(losses) / 4 == pytest.approx(2 * (1 + 3 + 6) / 3)
    assert sum(grads) / 4 == pytest.approx((1 + 3 + 6) / 3)
    assert losses[-1] == grads[-1] == 0


def test_native_actor_and_critic_losses_are_finite_on_dummy_rank(monkeypatch):
    from verl.trainer.ppo import core_algos
    from verl.workers.actor import dp_actor

    monkeypatch.setattr(core_algos, "agg_loss", core_algos.agg_loss)
    monkeypatch.setattr(dp_actor, "agg_loss", dp_actor.agg_loss)  # pyright: ignore[reportPrivateImportUsage]
    monkeypatch.setattr(ppo.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ppo.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(ppo.dist, "all_reduce", lambda count, **kwargs: count.fill_(3))
    ppo.register_in_worker()
    prediction = torch.ones(1, 3, requires_grad=True)
    mask = torch.zeros_like(prediction)
    loss, _ = core_algos.compute_value_loss(
        prediction, torch.zeros_like(prediction), torch.zeros_like(prediction), mask, 0.5, "seq-mean-token-mean"
    )
    loss.backward()
    assert loss.item() == 0 and prediction.grad.count_nonzero() == 0
    assert dp_actor.agg_loss(prediction, mask, "seq-mean-token-mean").item() == 0  # pyright: ignore[reportPrivateImportUsage]
