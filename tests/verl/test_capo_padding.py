# Copyright (c) Microsoft. All rights reserved.

from typing import Any, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")

from agentlightning.verl.capo_padding import row_weights  # noqa: E402


@pytest.mark.parametrize("counts", [[4, 3], [3, 1]])
def test_padding_and_short_tail_match_unpadded_optimizer_gradient(counts):
    # Two ranks, nominal two calls/rank/minibatch. The final minibatch may be short.
    rank_flags = (
        [[False, False, False, False], [False, False, False, True]]
        if counts == [4, 3]
        else [[False, False, False], [False, True, True]]
    )
    parameter = torch.tensor(0.3, requires_grad=True)
    scales = [row_weights(np.array(flags), 2, counts, 2) for flags in rank_flags]
    for mini, real_count in enumerate(counts):
        losses, real_losses = [], []
        for rank, flags in enumerate(rank_flags):
            for row in range(mini * 2, min((mini + 1) * 2, len(flags))):
                target = 1000.0 if flags[row] else float(rank * 10 + row)
                loss = (parameter - target) ** 2
                losses.append(loss * scales[rank][row] / 2 / 2)  # accumulation, then FSDP averaging
                if not flags[row]:
                    real_losses.append(loss)
        actual = torch.autograd.grad(sum(losses), parameter, retain_graph=True)[0]
        expected = torch.autograd.grad(sum(real_losses) / real_count, parameter, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)


def test_entirely_dummy_minibatch_is_rejected():
    with pytest.raises(ValueError, match="no real calls"):
        row_weights(np.array([True]), 1, [0], 1)


@pytest.mark.parametrize("kind", ["actor", "critic"])
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_copied_optimizer_loops_ignore_dummy_and_normalize_short_tail(monkeypatch, kind, microbatch):
    import verl.workers.actor as actor_api
    import verl.workers.critic as critic_api
    from omegaconf import OmegaConf
    from verl import DataProto

    from agentlightning.verl import capo_padding, capo_ppo
    from agentlightning.verl.vendor.capo import dp_actor, dp_critic, verl_core_algos

    # Run the actual copied loops on a differentiable scalar model. The dummy
    # has an extreme target, and the final minibatch has only one real call.
    monkeypatch.setenv("AGL_CAPO_STRICT_PADDING", "1")
    for module in (capo_padding, dp_actor, dp_critic):
        monkeypatch.setattr(module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(actor_api, "DataParallelPPOActor", actor_api.DataParallelPPOActor)
    monkeypatch.setattr(critic_api, "DataParallelPPOCritic", critic_api.DataParallelPPOCritic)
    monkeypatch.setattr(verl_core_algos, "agg_loss", verl_core_algos.agg_loss)
    monkeypatch.setattr(dp_actor, "agg_loss", dp_actor.agg_loss)
    capo_ppo.register_in_worker()
    cls = actor_api.DataParallelPPOActor if kind == "actor" else critic_api.DataParallelPPOCritic
    worker = cast(Any, object.__new__(cls))
    mini = max(2, microbatch)
    rows = mini + 1
    worker.config = OmegaConf.create(
        dict(
            ppo_mini_batch_size=mini,
            ppo_micro_batch_size_per_gpu=microbatch,
            ppo_epochs=1,
            use_dynamic_bsz=False,
            cliprange_value=1e9,
            loss_agg_mode="seq-mean-token-mean",
            use_kl_loss=True,
            kl_loss_type="low_var_kl",
            kl_loss_coef=0.1,
            entropy_coeff=0,
            calculate_entropy=False,
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3,
            policy_loss={"loss_mode": "vanilla"},
            global_batch_info={},
        )
    )
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    setattr(worker, kind + "_module", model)
    setattr(worker, kind + "_optimizer", optimizer)
    worker.scaler = None

    def forward(self, inputs, **kwargs):
        value = model.weight.expand_as(inputs["responses"])
        return (None, value) if kind == "actor" else value

    base = dp_actor.DataParallelPPOActor if kind == "actor" else dp_critic.DataParallelPPOCritic
    monkeypatch.setattr(base, "_forward_micro_batch", forward)

    def step():
        norm = model.weight.grad.norm().detach()
        optimizer.step()
        return norm

    worker._optimizer_step = step
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(rows, 2, dtype=torch.long),
            "responses": torch.ones(rows, 2, dtype=torch.long),
            "response_mask": torch.ones(rows, 2),
            "attention_mask": torch.ones(rows, 2),
            "position_ids": torch.zeros(rows, 2, dtype=torch.long),
            "values": torch.full((rows, 2), 0.3),
            "returns": torch.tensor([[1.0, 1.0]] + [[1000.0, 1000.0]] * (mini - 1) + [[3.0, 3.0]]),
            "old_log_probs": torch.full((rows, 2), 0.3),
            "ref_log_prob": torch.full((rows, 2), 0.1),
            "advantages": torch.tensor([[1.0, 1.0]] + [[1000.0, 1000.0]] * (mini - 1) + [[2.0, 2.0]]),
        },
        non_tensors={"is_pad": np.array([False] + [True] * (mini - 1) + [False])},
        meta_info={"temperature": 1.0},
    )

    expected = torch.tensor(0.3, requires_grad=True)
    for index in (0, mini):
        if kind == "critic":
            loss = 0.5 * (expected - batch.batch["returns"][index, 0]) ** 2
        else:
            ratio = (expected - 0.3).exp()
            advantage = batch.batch["advantages"][index, 0]
            pg = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2))
            kl = (0.1 - expected).exp() - (0.1 - expected) - 1
            loss = pg + 0.1 * kl
        (grad,) = torch.autograd.grad(loss, expected)
        expected = (expected - 0.05 * grad).detach().requires_grad_(True)
    result = worker.update_policy(batch) if kind == "actor" else worker.update_critic(batch)
    torch.testing.assert_close(model.weight.squeeze(), expected)
    assert result["padding/ignored_calls"] == mini - 1
    assert capo_padding._loss_weight.get() == 1
