# Copyright (c) Microsoft. All rights reserved.

"""Opt-in FSDP PPO padding with a global mean over real model calls.

Supported layout: one call per rank per optimizer step, no sequence parallelism.
Padding only equalizes worker invocation counts; it never enters GAE/whitening.
The loss is an average of per-call token means, matching the single-card call
weighting while increasing the optimizer batch to the data-parallel world size.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.distributed as dist
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor


def pad_inference(batch: DataProto, world_size: int) -> DataProto:
    padded, count = pad_dataproto_to_divisor(batch, world_size)
    marker = torch.zeros(len(padded), dtype=torch.bool)
    if count:
        marker[-count:] = True
    padded.batch["ppo_padding"] = marker
    padded.meta_info["global_token_num"] = padded.batch["attention_mask"].sum(-1).tolist()
    return padded


def unpad_inference(batch: DataProto) -> DataProto:
    """A marker survives load balancing; appended-row slicing would not."""
    keep = (~batch.batch["ppo_padding"]).nonzero(as_tuple=True)[0].tolist()
    result = cast(DataProto, batch[keep])
    result.batch.pop("ppo_padding")
    result.meta_info["global_token_num"] = result.batch["attention_mask"].sum(-1).tolist()
    return result


def pad_update(batch: DataProto, world_size: int) -> tuple[DataProto, int]:
    padded, count = pad_dataproto_to_divisor(batch, world_size)
    # Rollout-vs-current-policy diagnostics on empty dummy rows are undefined.
    # Those diagnostics have already been computed over real rows by the trainer.
    # TensorDict iteration is row-wise; explicitly iterate its keys.
    result = padded.select(batch_keys=[k for k in padded.batch.keys() if k != "rollout_log_probs"])  # noqa: SIM118
    result.batch["response_mask"] = result.batch["response_mask"].clone()
    if count:
        result.batch["response_mask"][-count:] = 0
    result.meta_info["global_token_num"] = result.batch["attention_mask"].sum(-1).tolist()
    return result, count


def local_loss_sum(loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable zero for a dummy-only rank, with no zero denominator."""
    mask = loss_mask.to(loss_mat.dtype)
    lengths = mask.sum(-1)
    per_call = (loss_mat * mask).sum(-1) / lengths.clamp_min(1)
    return per_call.sum(), (lengths > 0).sum().to(dtype=torch.float32)


def global_call_mean(loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    local_sum, global_count = local_loss_sum(loss_mat, loss_mask)
    world = 1
    if dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if global_count.item() == 0:
        raise RuntimeError("An optimizer microbatch contains no real model calls on any rank")
    # FSDP averages gradients across ranks; undo that factor before normalization.
    return local_sum * world / global_count


def register_in_worker() -> None:
    """Patch only this Ray job's selected loss aggregation, including critic.

    veRL 0.7.1's stock aggregation divides by a local valid-row count, which is
    zero for a dummy-only rank. Its critic API also lacks global normalization
    arguments. Keeping their clipped objectives and replacing this shared
    aggregation avoids copying the actor/critic optimizer loops.
    """
    from verl.trainer.ppo import core_algos
    from verl.workers.actor import dp_actor

    from . import per_rollout_loss  # noqa: F401

    original = core_algos.agg_loss
    if getattr(original, "_agl_padding_aware", False):
        return

    def aggregate(loss_mat, loss_mask, loss_agg_mode, *args, **kwargs):
        if loss_agg_mode == "seq-mean-token-mean":
            return global_call_mean(loss_mat, loss_mask)
        return original(loss_mat, loss_mask, loss_agg_mode, *args, **kwargs)

    cast(Any, aggregate)._agl_padding_aware = True
    core_algos.agg_loss = aggregate
    dp_actor.agg_loss = aggregate  # pyright: ignore[reportPrivateImportUsage]
