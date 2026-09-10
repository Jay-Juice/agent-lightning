# Copyright (c) Microsoft. All rights reserved.

"""Temporal correctness and strict full-episode PPO regression tests."""
# ruff: noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")
from verl import DataProto

from agentlightning.schemas import RolloutState
from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet
from agentlightning.verl.multi_turn_ppo import build_batch, compute_advantage
from agentlightning.verl.rollout_adapter import RolloutAdapter


def episode(rid="a", reward=1.0):
    return CompletedRollout(
        rollout_id=rid,
        data_id="same-task",
        step=1,
        sample_idx_in_step=0,
        enqueue_time=0,
        rollout_state=RolloutState.SUCCEEDED,
        final_reward=reward,
        triplets=[
            Triplet(
                prompt={"token_ids": [1, 2]},
                response={"token_ids": [3, 4], "log_probs": None},
            ),
            # Repeated prompts must still represent two different sampled actions.
            Triplet(
                prompt={"token_ids": [1, 2]},
                response={"token_ids": [5], "log_probs": None},
            ),
        ],
    )


def adapter():
    return RolloutAdapter(
        max_prompt_length=8,
        max_response_length=4,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level="transition",
    )


def test_only_terminal_call_receives_reward():
    batch, _ = build_batch(adapter(), [episode()])
    assert batch.batch["token_level_scores"].tolist() == [[0, 0, 0, 0], [1, 0, 0, 0]]
    assert batch.non_tensor_batch["episode_terminal"].tolist() == [False, True]
    assert batch.batch["response_mask"].tolist() == [[1, 1, 0, 0], [1, 0, 0, 0]]
    # The legacy adapter's reward semantics must not change for existing GRPO users.
    legacy, _ = adapter().get_train_data_batch([episode()])
    assert legacy.batch["token_level_scores"].sum(-1).tolist() == [1, 1]


@pytest.mark.parametrize("mutation", ["failed", "missing_reward", "oversize", "empty", "lost_call"])
def test_refuse_incomplete_episode(mutation):
    rollout = episode()
    assert rollout.triplets is not None
    if mutation == "failed":
        rollout.rollout_state = RolloutState.FAILED
    elif mutation == "missing_reward":
        rollout.final_reward = None
    elif mutation == "oversize":
        rollout.triplets[1].prompt["token_ids"] = list(range(9))
    elif mutation == "empty":
        rollout.triplets[0].response["token_ids"] = []
    else:
        rollout.events = [{"event_type": "model_request"}] * 3
    with pytest.raises(ValueError):
        build_batch(adapter(), [rollout])


def temporal_batch():
    # Shuffled rows: episode a turn 1, episode b turn 0, episode a turn 0.
    # Non-actions have arbitrary values/rewards and must not affect the answer.
    return DataProto.from_dict(
        tensors={
            "values": torch.tensor([[0.4, 99, 0.5], [0.6, 0.7, 99], [0.1, 99, 0.2]]),
            "token_level_rewards": torch.tensor([[0.0, 99, 1.0], [0.0, 2.0, 99], [0.0, 99, 0.0]]),
            "response_mask": torch.tensor([[1, 0, 1], [1, 1, 0], [1, 0, 1]]),
        },
        non_tensors={
            "rollout_id_list": np.array(["a", "b", "a"], dtype=object),
            "turn_index_list": np.array([1, 0, 0]),
            "episode_turn_count": np.array([2, 1, 2]),
            "episode_terminal": np.array([True, True, False]),
        },
    )


@pytest.mark.parametrize("gamma,lam", [(1.0, 1.0), (0.9, 0.8), (0.0, 0.0), (1.0, 0.0)])
def test_gae_matches_independent_flattened_oracle(gamma, lam):
    batch = temporal_batch()
    out = compute_advantage(batch, gamma=gamma, lam=lam, whiten=False)
    for coordinates in [[(2, 0), (2, 2), (0, 0), (0, 2)], [(1, 0), (1, 1)]]:
        vals = [float(batch.batch["values"][r, p]) for r, p in coordinates] + [0.0]
        rewards = [float(batch.batch["token_level_rewards"][r, p]) for r, p in coordinates]
        deltas = [reward + gamma * vals[i + 1] - vals[i] for i, reward in enumerate(rewards)]
        for i, (r, p) in enumerate(coordinates):
            expected = sum((gamma * lam) ** (j - i) * deltas[j] for j in range(i, len(deltas)))
            assert float(out.batch["advantages"][r, p]) == pytest.approx(expected, abs=1e-6)
            assert float(out.batch["returns"][r, p]) == pytest.approx(expected + vals[i], abs=1e-6)
    invalid = ~out.batch["response_mask"].bool()
    assert (out.batch["advantages"][invalid] == 0).all()
    assert (out.batch["returns"][invalid] == 0).all()


def test_whitening_does_not_change_critic_returns():
    raw = compute_advantage(temporal_batch(), gamma=1, lam=1, whiten=False)
    norm = compute_advantage(temporal_batch(), gamma=1, lam=1, whiten=True)
    assert torch.equal(raw.batch["returns"], norm.batch["returns"])
    valid = norm.batch["response_mask"].bool()
    assert norm.batch["advantages"][valid].mean().item() == pytest.approx(0, abs=1e-6)
    assert norm.batch["returns"][0, 2].item() == pytest.approx(1)
    assert norm.batch["returns"][1, 1].item() == pytest.approx(2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("turn_index_list", [1, 0, 1]),
        ("episode_turn_count", [3, 1, 3]),
        ("episode_terminal", [False, True, False]),
    ],
)
def test_refuse_broken_temporal_metadata(key, value):
    batch = temporal_batch()
    batch.non_tensor_batch[key] = np.array(value)
    with pytest.raises(ValueError):
        compute_advantage(batch, gamma=1, lam=1)
