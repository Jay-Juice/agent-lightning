"""CPU checks for diagnostic denominators, padding exclusion and immutability."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from agentlightning.verl.reliability_metrics import batch_diagnostics, rollout_diagnostics  # noqa: E402


def _batch():
    # Real calls have 2, 1, 1 response tokens. NaNs in masked and pad
    # positions must not pollute any real-only statistic.
    ignored = float("nan")
    mask = torch.tensor([[1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
    return SimpleNamespace(
        batch={
            "response_mask": mask,
            "values": torch.tensor([[0., ignored, 2.], [1., ignored, ignored],
                                    [ignored, 0., ignored], [ignored, ignored, ignored]]),
            "returns": torch.tensor([[0., ignored, 1.], [1., ignored, ignored],
                                     [ignored, 0., ignored], [ignored, ignored, ignored]]),
            "advantages": torch.tensor([[1., ignored, 3.], [10., ignored, ignored],
                                        [ignored, -2., ignored], [ignored, ignored, ignored]]),
            "raw_advantages": torch.tensor([[2., ignored, 6.], [20., ignored, ignored],
                                            [ignored, -4., ignored], [ignored, ignored, ignored]]),
            "token_level_scores": torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),
            "old_log_probs": torch.full((4, 3), -2.),
            "ref_log_prob": torch.full((4, 3), -3.),
            "rollout_log_probs": torch.full((4, 3), -2.),
            # A prompt-shaped unrelated tensor must not be examined or copied.
            "input_ids": torch.zeros(4, 65536, dtype=torch.int32),
        },
        non_tensor_batch={
            "is_pad": [False, False, False, True],
            "rollout_id_list": ["episode-a", "episode-a", "episode-b", "episode-c"],
            "uid": ["same-task"] * 4,
        },
    )


def test_batch_real_response_value_advantage_and_kl_denominators():
    batch = _batch()
    original = {key: tensor.clone() for key, tensor in batch.batch.items()}
    metadata = copy.deepcopy(batch.non_tensor_batch)
    result = batch_diagnostics(batch)
    assert result["reliability/calls"] == 3
    assert result["reliability/action_tokens"] == 4
    assert result["reliability/episodes"] == 2
    assert result["reliability/calls_per_episode/mean"] == 1.5
    assert result["reliability/calls_per_episode/p50"] == 1.5
    assert result["reliability/value/mse"] == .25
    assert result["reliability/value/return_variance"] == .25
    assert result["reliability/value/constant_mean_mse"] == .25
    assert result["reliability/value/constant_zero_mse"] == .5
    assert result["reliability/value/explained_variance"] == .25
    assert result["reliability/value/mean"] == .75
    assert result["reliability/advantages/all/token_mean"] == 3
    assert result["reliability/advantages/all/call_mean"] == pytest.approx(10 / 3)
    assert result["reliability/advantages/success/token_mean"] == pytest.approx(14 / 3)
    assert result["reliability/advantages/success/call_mean"] == 6
    assert result["reliability/advantages/failure/token_count"] == 1
    assert result["reliability/advantages/failure/call_mean"] == -2
    assert result["reliability/raw_advantages/all/token_mean"] == 6
    assert result["reliability/sampled_k3/old_vs_ref/mean"] == pytest.approx(math.exp(-1))
    assert result["reliability/sampled_k3/old_vs_ref/p99"] == pytest.approx(math.exp(-1))
    assert result["reliability/sampled_k3/old_vs_rollout/max"] == 0
    assert all(math.isfinite(value) for value in result.values())
    for key, tensor in batch.batch.items():
        torch.testing.assert_close(tensor, original[key], rtol=0, atol=0, equal_nan=True)
    assert batch.non_tensor_batch == metadata


def test_empty_success_and_constant_returns_omit_undefined_metrics():
    batch = _batch()
    batch.batch["returns"].zero_()
    batch.batch["token_level_scores"].zero_()
    result = batch_diagnostics(batch)
    assert result["reliability/success/calls"] == 0
    assert result["reliability/advantages/success/token_count"] == 0
    assert "reliability/advantages/success/token_mean" not in result
    assert result["reliability/value/ev_available"] == 0
    assert "reliability/value/explained_variance" not in result
    assert result["reliability/value/constant_mean_mse"] == 0
    assert all(math.isfinite(value) for value in result.values())


def test_all_padding_and_missing_optional_inputs_are_explicit():
    batch = _batch()
    batch.non_tensor_batch["is_pad"] = [True] * 4
    result = batch_diagnostics(batch)
    assert result["reliability/calls"] == 0
    assert result["reliability/episodes"] == 0
    assert result["reliability/value/count"] == 0
    assert "reliability/value/mse" not in result
    assert result["reliability/advantages/all/token_count"] == 0
    assert all(math.isfinite(value) for value in result.values())
    batch = SimpleNamespace(batch={"response_mask": torch.ones(1, 2)}, non_tensor_batch={})
    result = batch_diagnostics(batch)
    assert result["reliability/episodes_available"] == 0
    assert "reliability/episodes" not in result
    assert result["reliability/reward_groups_available"] == 0
    assert result["reliability/sampled_k3/old_vs_ref/available"] == 0


@pytest.mark.parametrize("key", ["values", "returns", "advantages", "old_log_probs", "ref_log_prob"])
def test_real_nonfinite_tensors_are_rejected(key):
    batch = _batch()
    batch.batch[key][0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        batch_diagnostics(batch)


def test_k3_overflow_is_not_silently_clipped_or_logged():
    batch = _batch()
    batch.batch["ref_log_prob"].fill_(1000)
    with pytest.raises(ValueError, match="Nonfinite"):
        batch_diagnostics(batch)


def _model(tokens, finish="stop", **overrides):
    return {"event_type": "model_request", "data": {
        "response": {"choices": [{"token_ids": tokens, "finish_reason": finish}]}, **overrides,
    }}


def _reward(reason):
    return {"event_type": "reward", "data": {"value": 0., "reason": reason}}


def test_rollout_counts_exclude_failed_requests_and_use_episode_denominators():
    episodes = [
        SimpleNamespace(events=[_model([1, 2]), _model([3], "length"), _reward("submitted")], final_reward=1),
        SimpleNamespace(events=[_model([5] * 100, status="error"), _model([9], http_status=500),
                                _model([]), _model([4, 5, 6]), _reward("format_errors")], final_reward=0),
        SimpleNamespace(events=[_model([], status="error"), _reward("context_budget")], final_reward=0),
    ]
    original = copy.deepcopy(episodes)
    result = rollout_diagnostics(episodes, "training/behavior/")
    key = "training/behavior/"
    assert result[key + "episodes"] == 3
    assert result[key + "model_requests"] == 7
    assert result[key + "failed_requests"] == 3
    assert result[key + "empty_response_requests"] == 1
    assert result[key + "calls"] == 3
    assert result[key + "output_tokens"] == 6
    assert result[key + "calls_per_episode/mean"] == 1
    assert result[key + "calls_per_episode/min"] == 0
    for name in ("submitted", "format_stop", "length"):
        assert result[key + name + "_episode_ratio"] == pytest.approx(1 / 3)
    assert result[key + "finish_reason/length/count"] == 1
    assert result[key + "stop_reason/context_budget/count"] == 1
    assert episodes == original


def test_rollout_streaming_trimmed_and_triplet_fallback():
    episodes = [
        {"events": [{"event_type": "model_request", "data": {"response": [
            {"choices": [{"token_ids": [1, 2], "finish_reason": None}]},
            {"choices": [{"token_ids": [3], "finish_reason": "length"}]},
        ]}}, _reward("turn_budget")]},
        {"triplet_events": [{"event_type": "model_request", "data": {"response_token_ids": [4, 5]}},
                            _reward("submitted")]},
        {"triplets": [{"response": {"token_ids": [6]}}]},
    ]
    result = rollout_diagnostics(episodes)
    assert result["val/behavior/calls"] == 3
    assert result["val/behavior/output_tokens"] == 6
    assert result["val/behavior/length_observed_episode_ratio"] == pytest.approx(1 / 3)
    assert result["val/behavior/length_episode_ratio_available"] == 0
    assert "val/behavior/length_episode_ratio" not in result
    assert result["val/behavior/finish_reason/unknown/count"] == 2
    assert result["val/behavior/triplet_fallback_episodes"] == 1
    assert result["val/behavior/stop_reason/unknown/count"] == 1


def test_empty_rollouts_and_nonfinite_reward():
    result = rollout_diagnostics([])
    assert result["val/behavior/episodes"] == 0
    assert "val/behavior/submitted_episode_ratio" not in result
    assert all(math.isfinite(value) for value in result.values())
    with pytest.raises(ValueError, match="Nonfinite"):
        rollout_diagnostics([{"final_reward": float("nan")}])
