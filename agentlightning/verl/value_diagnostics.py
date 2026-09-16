"""Read-only value quality diagnostics computed before optimizer updates."""

from __future__ import annotations

import torch
from verl import DataProto


def _summary(pred: torch.Tensor, target: torch.Tensor, prefix: str) -> dict[str, float]:
    if not pred.numel():
        return {
            f"value/{name}_{prefix}": float("nan")
            for name in (
                "mse",
                "mae",
                "bias",
                "return_mean",
                "return_var",
                "pred_mean",
                "pred_var",
                "explained_variance",
            )
        }
    error = pred.float() - target.float()
    var = target.float().var(unbiased=False)
    ev = float("nan") if var.item() < 1e-12 else float(1.0 - error.var(unbiased=False).item() / var.item())
    return {
        f"value/mse_{prefix}": float(error.square().mean().item()),
        f"value/mae_{prefix}": float(error.abs().mean().item()),
        f"value/bias_{prefix}": float(error.mean().item()),
        f"value/return_mean_{prefix}": float(target.float().mean().item()),
        f"value/return_var_{prefix}": float(var.item()),
        f"value/pred_mean_{prefix}": float(pred.float().mean().item()),
        f"value/pred_var_{prefix}": float(pred.float().var(unbiased=False).item()),
        f"value/explained_variance_{prefix}": ev,
    }


def compute_value_diagnostics(batch: DataProto) -> dict[str, float]:
    mask = batch.batch["response_mask"].bool().clone()
    if "is_pad" in batch.non_tensor_batch:
        pad = torch.as_tensor(batch.non_tensor_batch["is_pad"], device=mask.device, dtype=torch.bool)
        mask[pad] = False
    pred = batch.batch["values"].float()
    target = batch.batch["returns"].float()
    if pred.shape != mask.shape or target.shape != mask.shape:
        raise ValueError(
            f"value diagnostics shape mismatch: values={pred.shape}, returns={target.shape}, mask={mask.shape}"
        )
    metrics = {"value/n_tokens": float(mask.sum().item())}
    metrics.update(_summary(pred[mask], target[mask], "all"))
    first_pred, first_target, last_pred, last_target = [], [], [], []
    for row in range(mask.shape[0]):
        indexes = mask[row].nonzero(as_tuple=True)[0]
        if len(indexes):
            first, last = indexes[0], indexes[-1]
            first_pred.append(pred[row, first])
            first_target.append(target[row, first])
            last_pred.append(pred[row, last])
            last_target.append(target[row, last])
    empty = pred.new_empty(0)
    for name, predictions, targets in (("first", first_pred, first_target), ("last", last_pred, last_target)):
        metrics.update(
            _summary(
                torch.stack(predictions) if predictions else empty,
                torch.stack(targets) if targets else empty,
                name,
            )
        )

    # Row metadata follows DataProto filtering/reordering/padding. Never infer
    # semantic content from prompt length: a manifest is not a semantic patch.
    keys = ("pi_semantic_content", "pi_truncated", "pi_changed_lines_total")
    metadata_available = all(key in batch.non_tensor_batch for key in keys)
    metrics["value/pi_groups_available"] = float(metadata_available)
    row_flags = {}
    if metadata_available:
        for key in keys:
            flag = torch.as_tensor(batch.non_tensor_batch[key], device=mask.device)
            if flag.shape != mask.shape[:1]:
                raise ValueError(f"value diagnostics row metadata shape mismatch for {key}: {flag.shape}")
            row_flags[key] = flag
        semantic = row_flags["pi_semantic_content"].bool()
        truncated = row_flags["pi_truncated"].bool()
        changed = row_flags["pi_changed_lines_total"] > 0
        groups = {
            "with_semantic_pi": semantic,
            "no_semantic_pi": ~semantic,
            "truncated_pi": truncated,
            # Unchanged initial state does not count as fully represented PI.
            "full_pi": changed & semantic & ~truncated,
        }
    else:
        groups = {
            name: torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
            for name in ("with_semantic_pi", "no_semantic_pi", "truncated_pi", "full_pi")
        }
    for name, rows in groups.items():
        group_mask = mask & rows[:, None]
        metrics[f"value/n_tokens_{name}"] = float(group_mask.sum().item())
        metrics[f"value/n_rows_{name}"] = float(group_mask.any(dim=-1).sum().item())
        metrics[f"value/available_{name}"] = float(group_mask.any().item())
        metrics.update(_summary(pred[group_mask], target[group_mask], name))
    return metrics
