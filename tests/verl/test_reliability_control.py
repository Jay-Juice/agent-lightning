# Copyright (c) Microsoft. All rights reserved.
import pytest

from agentlightning.verl.reliability_control import behavior_violations, merge_behavior_metrics


def metrics(n=100, tokens=1000):
    values = {"episodes": n, "submitted_episodes": n, "format_stop_episodes": 0,
              "length_episodes": 0, "length_unknown_episodes": 0, "length_episode_ratio_available": 1,
              "submitted_episode_ratio": 1, "format_stop_episode_ratio": 0, "length_episode_ratio": 0,
              "output_tokens": n * tokens, "calls": n}
    for name, value in (("calls_per_episode", 1), ("output_tokens_per_episode", tokens)):
        values.update({f"{name}/count": n, f"{name}/mean": value, f"{name}/std": 0,
                       f"{name}/min": value, f"{name}/max": value, f"{name}/p50": value})
    return {"val/behavior/" + key: value for key, value in values.items()}


def test_behavior_gate_detects_observed_collapse():
    healthy = metrics()
    assert behavior_violations(healthy, 1000) == []
    healthy["val/behavior/submitted_episode_ratio"] = 0
    healthy["val/behavior/format_stop_episode_ratio"] = 437 / 470
    healthy["val/behavior/length_episode_ratio"] = 447 / 470
    assert len(behavior_violations(healthy, 1000)) == 3
    healthy["val/behavior/length_episode_ratio_available"] = 0
    with pytest.raises(ValueError, match="coverage"):
        behavior_violations(healthy, 1000)


def test_split_validation_aggregates_counts_and_moments_without_fake_quantiles():
    result = merge_behavior_metrics([metrics(3, 1000), metrics(1, 2000)])
    assert result["val/behavior/episodes"] == 4
    assert result["val/behavior/output_tokens_per_episode/mean"] == 1250
    assert result["val/behavior/output_tokens_per_episode/std"] == pytest.approx(433.0127019)
    assert "val/behavior/output_tokens_per_episode/p50" not in result
    assert behavior_violations(result, 1000) == []
