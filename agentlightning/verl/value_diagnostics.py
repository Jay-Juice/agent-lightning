"""Read-only value quality diagnostics computed before optimizer updates."""

from __future__ import annotations

import math
import torch
from verl import DataProto


def _summary(pred: torch.Tensor, target: torch.Tensor, prefix: str) -> dict[str, float]:
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
        raise ValueError(f"value diagnostics shape mismatch: values={pred.shape}, returns={target.shape}, mask={mask.shape}")
    if not mask.any():
        return {"value/n_tokens": 0.0}
    metrics = {"value/n_tokens": float(mask.sum().item())}
    metrics.update(_summary(pred[mask], target[mask], "all"))
    first_pred, first_target, last_pred, last_target = [], [], [], []
    for row in range(mask.shape[0]):
        indexes = mask[row].nonzero(as_tuple=True)[0]
        if len(indexes):
            first, last = indexes[0], indexes[-1]
            first_pred.append(pred[row, first]); first_target.append(target[row, first])
            last_pred.append(pred[row, last]); last_target.append(target[row, last])
    metrics.update(_summary(torch.stack(first_pred), torch.stack(first_target), "first"))
    metrics.update(_summary(torch.stack(last_pred), torch.stack(last_target), "last"))
    return metrics
