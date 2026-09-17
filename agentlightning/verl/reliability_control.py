# Copyright (c) Microsoft. All rights reserved.
"""Pure control rules for opt-in SWE reliability; no PPO objective changes."""

import math


def behavior_violations(metrics, initial_tokens):
    prefix = "val/behavior/"
    required = ("episodes", "submitted_episode_ratio", "format_stop_episode_ratio",
                "length_episode_ratio_available", "output_tokens_per_episode/mean")
    if any(prefix + key not in metrics or not math.isfinite(metrics[prefix + key]) for key in required):
        raise ValueError("Incomplete/nonfinite validation behavior metrics")
    if metrics[prefix + "episodes"] <= 0 or not metrics[prefix + "length_episode_ratio_available"]:
        raise ValueError("Validation lacks complete episode/finish-reason coverage")
    if not math.isfinite(initial_tokens) or initial_tokens <= 0:
        raise ValueError("Missing initial validation token baseline")
    violations = []
    for key, limit, lower in (
        ("format_stop_episode_ratio", .05, False),
        ("submitted_episode_ratio", .8, True),
        ("length_episode_ratio", .1, False),
        ("output_tokens_per_episode/mean", 2 * initial_tokens, False),
    ):
        value = metrics[prefix + key]
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite behavior metric: {key}")
        if (value < limit) if lower else (value > limit):
            violations.append({"metric": prefix + key, "value": value, "limit": limit})
    return violations


def merge_behavior_metrics(batches, prefix="val/behavior/"):
    """Aggregate counts/moments exactly; do not average batch quantiles."""
    if not any(prefix + "episodes" in batch for batch in batches):
        return {}
    if any(prefix + "episodes" not in batch for batch in batches):
        raise ValueError("Missing behavior batch")
    if len(batches) == 1:
        return {key: value for key, value in batches[0].items() if key.startswith(prefix)}
    keys = {key for batch in batches for key in batch if key.startswith(prefix)}
    result = {}
    for key in keys:
        name = key.removeprefix(prefix)
        if ("/" not in name and "ratio" not in name) or name.endswith("/count"):
            result[key] = sum(batch.get(key, 0) for batch in batches)
    count = result[prefix + "episodes"]
    if count <= 0:
        raise ValueError("No validation episodes")
    for name in ("submitted", "format_stop"):
        result[prefix + name + "_episode_ratio"] = result[prefix + name + "_episodes"] / count
    available = not result[prefix + "length_unknown_episodes"]
    result[prefix + "length_episode_ratio_available"] = int(available)
    result[prefix + "length_observed_episode_ratio"] = result[prefix + "length_episodes"] / count
    if available:
        result[prefix + "length_episode_ratio"] = result[prefix + "length_episodes"] / count
    for key in list(result):
        if key.startswith(prefix + "stop_reason/") and key.endswith("/count"):
            result[key.removesuffix("count") + "episode_ratio"] = result[key] / count
    for name in ("calls_per_episode", "output_tokens_per_episode"):
        base = prefix + name + "/"
        mean = sum(b[base + "mean"] * b[base + "count"] for b in batches) / count
        second = sum((b[base + "std"] ** 2 + b[base + "mean"] ** 2) * b[base + "count"]
                     for b in batches) / count
        result[base + "mean"] = mean
        result[base + "std"] = math.sqrt(max(0, second - mean ** 2))
        result[base + "min"] = min(b[base + "min"] for b in batches)
        result[base + "max"] = max(b[base + "max"] for b in batches)
    return result
