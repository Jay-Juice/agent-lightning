"""Read-only, real-sample diagnostics; no training objective or tensor mutation.

Batch statistics use response tokens only. Token means weight tokens equally;
call means first average within each nonempty action response. Success means a
positive terminal score. Episode identity is rollout_id_list, never task UID.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _finite(values: torch.Tensor, name: str) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"Nonfinite real response values in {name}")


def _distribution(values: torch.Tensor, name: str, metrics: dict[str, float]) -> None:
    metrics[f"{name}/count"] = int(values.numel())
    if not values.numel():
        return
    _finite(values, name)
    metrics[f"{name}/mean"] = float(values.mean())
    metrics[f"{name}/std"] = float(values.std(unbiased=False))
    quantiles = torch.quantile(values, torch.tensor([0., .5, .9, .95, .99, 1.]))
    for label, value in zip(("min", "p50", "p90", "p95", "p99", "max"), quantiles, strict=True):
        metrics[f"{name}/{label}"] = float(value)


def batch_diagnostics(batch: Any) -> dict[str, float]:
    """Diagnose a post-GAE DataProto without retaining or editing its tensors.

    Absent inputs and empty groups have explicit counts/availability and omit
    undefined statistics. All reductions operate in CPU float32 on masked
    response data; full prompt tensors are never materialized or copied.
    """
    tensors = batch.batch
    metadata = batch.non_tensor_batch
    source_mask = tensors["response_mask"]
    if source_mask.ndim != 2:
        raise ValueError("response_mask must be a [calls, response_tokens] tensor")
    if not torch.isfinite(source_mask).all() or not ((source_mask == 0) | (source_mask == 1)).all():
        raise ValueError("response_mask must be finite and binary")
    mask = source_mask.detach().bool().clone()
    n_rows, width = mask.shape
    if "is_pad" in metadata:
        padding = torch.as_tensor(metadata["is_pad"], device=mask.device, dtype=torch.bool)
        if padding.shape != (n_rows,):
            raise ValueError("is_pad must have exactly one flag per batch row")
        mask[padding] = False
    counts = mask.sum(dim=-1).cpu()
    live = counts > 0
    # CPU row indices track each selected token, enabling equal-call summaries.
    row_ids = torch.repeat_interleave(torch.arange(n_rows), counts)
    metrics: dict[str, float] = {
        "reliability/calls": int(live.sum()),
        "reliability/action_tokens": int(counts.sum()),
    }
    _distribution(counts[live].float(), "reliability/action_tokens_per_call", metrics)

    def selected(key: str) -> torch.Tensor | None:
        if key not in tensors:
            return None
        value = tensors[key]
        if value.ndim != 2 or value.shape[0] != n_rows or value.shape[1] < width:
            raise ValueError(f"{key} is not aligned to the response mask: {tuple(value.shape)}")
        result = value[:, -width:][mask].detach().to(device="cpu", dtype=torch.float32)
        _finite(result, key)
        return result

    def call_means(values: torch.Tensor) -> torch.Tensor:
        sums = torch.zeros(n_rows, dtype=torch.float32)
        sums.scatter_add_(0, row_ids, values)
        return sums[live] / counts[live]

    rollout_ids = metadata.get("rollout_id_list")
    metrics["reliability/episodes_available"] = int(rollout_ids is not None)
    if rollout_ids is not None:
        if len(rollout_ids) != n_rows:
            raise ValueError("rollout_id_list must have exactly one ID per batch row")
        calls_per_episode = Counter(str(rollout_ids[i]) for i in live.nonzero().flatten().tolist())
        metrics["reliability/episodes"] = len(calls_per_episode)
        _distribution(
            torch.tensor(list(calls_per_episode.values()), dtype=torch.float32),
            "reliability/calls_per_episode",
            metrics,
        )

    predictions, returns = selected("values"), selected("returns")
    metrics["reliability/value/available"] = int(predictions is not None and returns is not None)
    metrics["reliability/value/count"] = int(predictions.numel()) if predictions is not None else 0
    metrics["reliability/value/ev_available"] = 0
    if predictions is not None and returns is not None and predictions.numel():
        error = predictions - returns
        return_var = returns.var(unbiased=False)
        metrics.update({
            "reliability/value/mse": float(error.square().mean()),
            "reliability/value/mae": float(error.abs().mean()),
            "reliability/value/bias": float(error.mean()),
            "reliability/value/mean": float(predictions.mean()),
            "reliability/value/std": float(predictions.std(unbiased=False)),
            "reliability/value/return_mean": float(returns.mean()),
            "reliability/value/return_variance": float(return_var),
            "reliability/value/constant_mean_mse": float(return_var),
            "reliability/value/constant_zero_mse": float(returns.square().mean()),
        })
        if return_var > 0:
            metrics["reliability/value/ev_available"] = 1
            metrics["reliability/value/explained_variance"] = float(1 - error.var(unbiased=False) / return_var)

    scores = selected("token_level_scores")
    metrics["reliability/reward_groups_available"] = int(scores is not None)
    success = None
    if scores is not None:
        per_call_scores = torch.zeros(n_rows, dtype=torch.float32)
        per_call_scores.scatter_add_(0, row_ids, scores)
        success = per_call_scores > 0
        for group, rows in (("success", live & success), ("failure", live & ~success)):
            metrics[f"reliability/{group}/calls"] = int(rows.sum())
            metrics[f"reliability/{group}/action_tokens"] = int(counts[rows].sum())

    for key in ("raw_advantages", "advantages"):
        values = selected(key)
        name = f"reliability/{key}"
        metrics[f"{name}/available"] = int(values is not None)
        if values is None:
            continue
        groups = {"all": torch.ones(n_rows, dtype=torch.bool)}
        if success is not None:
            groups.update(success=success, failure=~success)
        per_call = call_means(values)
        for group, rows in groups.items():
            tokens = values[rows[row_ids]]
            calls = per_call[rows[live]]
            metrics[f"{name}/{group}/token_count"] = int(tokens.numel())
            metrics[f"{name}/{group}/call_count"] = int(calls.numel())
            if tokens.numel():
                metrics[f"{name}/{group}/token_mean"] = float(tokens.mean())
                metrics[f"{name}/{group}/call_mean"] = float(calls.mean())

    old = selected("old_log_probs")
    for key, label in (("ref_log_prob", "old_vs_ref"), ("rollout_log_probs", "old_vs_rollout")):
        other = selected(key)
        name = f"reliability/sampled_k3/{label}"
        metrics[f"{name}/available"] = int(old is not None and other is not None)
        if old is None or other is None:
            metrics[f"{name}/count"] = 0
            continue
        # k3 = q/p - log(q/p) - 1, with p=old and q=reference/rollout.
        # expm1 avoids cancellation near zero; no clipping changes diagnostics.
        delta = other - old
        k3 = torch.expm1(delta) - delta
        _distribution(k3, name, metrics)
        if k3.numel():
            metrics[f"{name}/call_mean"] = float(call_means(k3).mean())

    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite batch reliability diagnostic reduction")
    return metrics


def _field(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _completion(data: Mapping[str, Any]) -> tuple[list[int], str | None]:
    response = data.get("response")
    if isinstance(response, Mapping):
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("token_ids") or [], choices[0].get("finish_reason")
    if isinstance(response, list):
        tokens, reason = [], None
        for chunk in response:
            choices = chunk.get("choices", [])
            if choices:
                tokens.extend(choices[0].get("token_ids") or [])
                reason = choices[0].get("finish_reason") or reason
        return tokens, reason
    return data.get("response_token_ids") or [], data.get("finish_reason")


def rollout_diagnostics(completed_rollouts: Sequence[Any], prefix: str = "val/behavior") -> dict[str, float]:
    """Measure terminal reasons and successful model responses from event data.

    A call/action is a non-error request with nonempty output token IDs. HTTP
    retries and empty/error responses do not count. Format-stop ratio only
    means the *terminal* reason was a format error; intermediate parse errors
    are not recoverable from reward/model_request events and are not inferred.
    All episode ratios divide by all completed episodes, including zero-call
    failures. Length means at least one accepted completion ended in length.
    """
    prefix = prefix.rstrip("/")
    metrics: dict[str, float] = {f"{prefix}/episodes": len(completed_rollouts)}
    stops: Counter[str] = Counter()
    finishes: Counter[str] = Counter()
    counts, lengths = [], []
    requests = failed_requests = empty_requests = fallback_episodes = 0
    length_episodes = unknown_length_episodes = 0
    for rollout in completed_rollouts:
        reward = _field(rollout, "final_reward")
        if reward is not None and not math.isfinite(float(reward)):
            raise ValueError("Nonfinite rollout reward")
        events = _field(rollout, "events", []) or []
        trimmed = _field(rollout, "triplet_events", []) or []
        reward_events = [e for e in events if _field(e, "event_type") == "reward"]
        if not reward_events:
            reward_events = [e for e in trimmed if _field(e, "event_type") == "reward"]
        for event in reward_events:
            value = _field(event, "data", {}).get("value")
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("Nonfinite reward event value")
        reason = (_field(reward_events[-1], "data", {}).get("reason") if reward_events else None) or "unknown"
        stops[re.sub(r"[^a-zA-Z0-9_-]", "_", str(reason))] += 1
        model_events = [e for e in events if _field(e, "event_type") == "model_request"]
        if not model_events:
            model_events = [e for e in trimmed if _field(e, "event_type") == "model_request"]
        call_count = token_count = 0
        saw_length = unknown_finish = False
        for event in model_events:
            data = _field(event, "data", {})
            requests += 1
            http_status = data.get("http_status")
            response = data.get("response")
            if (data.get("status") == "error" or (isinstance(http_status, int) and http_status >= 400)
                    or (isinstance(response, Mapping) and "error" in response)):
                failed_requests += 1
                continue
            tokens, finish = _completion(data)
            if not tokens:
                empty_requests += 1
                continue
            call_count += 1
            token_count += len(tokens)
            finishes[re.sub(r"[^a-zA-Z0-9_-]", "_", str(finish or "unknown"))] += 1
            saw_length |= finish == "length"
            unknown_finish |= finish is None
        if not model_events:
            # Triplets still establish calls/tokens, but finish reasons are unknown.
            fallback_episodes += 1
            for triplet in _field(rollout, "triplets", []) or []:
                tokens = _field(triplet, "response", {}).get("token_ids") or []
                if tokens:
                    call_count += 1
                    token_count += len(tokens)
                    finishes["unknown"] += 1
                    unknown_finish = True
        counts.append(call_count)
        lengths.append(token_count)
        length_episodes += int(saw_length)
        unknown_length_episodes += int(unknown_finish and not saw_length)
    metrics.update({
        f"{prefix}/model_requests": requests,
        f"{prefix}/failed_requests": failed_requests,
        f"{prefix}/empty_response_requests": empty_requests,
        f"{prefix}/triplet_fallback_episodes": fallback_episodes,
        f"{prefix}/calls": sum(counts),
        f"{prefix}/output_tokens": sum(lengths),
        f"{prefix}/length_episodes": length_episodes,
        f"{prefix}/length_unknown_episodes": unknown_length_episodes,
        f"{prefix}/length_episode_ratio_available": int(unknown_length_episodes == 0 and bool(completed_rollouts)),
        f"{prefix}/stop_reason_known_episodes": len(completed_rollouts) - stops["unknown"],
        f"{prefix}/submitted_episodes": stops["submitted"],
        f"{prefix}/format_stop_episodes": stops["format_errors"] + stops["format_error"],
    })
    for reason, count in stops.items():
        metrics[f"{prefix}/stop_reason/{reason}/count"] = count
    for reason, count in finishes.items():
        metrics[f"{prefix}/finish_reason/{reason}/count"] = count
    if completed_rollouts:
        for name in ("submitted", "format_stop"):
            metrics[f"{prefix}/{name}_episode_ratio"] = metrics[f"{prefix}/{name}_episodes"] / len(completed_rollouts)
        metrics[f"{prefix}/length_observed_episode_ratio"] = length_episodes / len(completed_rollouts)
        if not unknown_length_episodes:
            metrics[f"{prefix}/length_episode_ratio"] = length_episodes / len(completed_rollouts)
        for reason, count in stops.items():
            metrics[f"{prefix}/stop_reason/{reason}/episode_ratio"] = count / len(completed_rollouts)
    _distribution(torch.tensor(counts, dtype=torch.float32), f"{prefix}/calls_per_episode", metrics)
    _distribution(torch.tensor(lengths, dtype=torch.float32), f"{prefix}/output_tokens_per_episode", metrics)
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite rollout reliability diagnostic reduction")
    return metrics
