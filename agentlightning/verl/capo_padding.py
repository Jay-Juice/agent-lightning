# Copyright (c) Microsoft. All rights reserved.

"""Keep CAPO's copied update loops, but give duplicate padding zero loss weight.

Loss weights normalize over real calls
in the entire optimizer minibatch, accounting for FSDP gradient averaging and
the copied loop's fixed gradient-accumulation divisor, including a short tail.
"""

import os
import time
from collections import deque
from contextvars import ContextVar
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from verl.utils.device import get_device_id

_loss_weight: ContextVar[float | list[float]] = ContextVar("capo_padding_loss_weight", default=1.0)


def row_weights(is_pad, mini_size, global_counts, world):
    weights = []
    for index, count in enumerate(global_counts):
        if count <= 0:
            raise ValueError("An optimizer minibatch has no real calls")
        local = is_pad[index * mini_size : (index + 1) * mini_size]
        weights.extend(0.0 if pad else world * mini_size / count for pad in local)
    return weights


def update_with_real_call_mean(worker, data, copied_update):
    if "is_pad" not in data.non_tensor_batch:
        raise ValueError("CAPO padding markers were lost before the optimizer update")
    flags = np.asarray(data.non_tensor_batch["is_pad"], dtype=bool)
    if len(flags) != len(data):
        raise ValueError("CAPO padding markers do not match the optimizer batch")
    mini = worker.config.ppo_mini_batch_size
    counts = torch.tensor(
        [sum(~flags[i : i + mini]) for i in range(0, len(flags), mini)], dtype=torch.float32, device=get_device_id()
    )
    world = 1
    if dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    weights = row_weights(flags, mini, counts.tolist(), world) * worker.config.ppo_epochs
    worker._capo_loss_weights = deque(weights)
    worker._capo_progress = (time.monotonic(), len(weights)) if os.environ.get("AGL_CAPO_PROGRESS") == "1" else None
    token = _loss_weight.set(1.0)
    try:
        result = copied_update(data)
        if worker._capo_loss_weights:
            raise RuntimeError("CAPO forward order differs from the supported fixed microbatch layout")
        result["padding/real_calls"] = int((~flags).sum())
        result["padding/ignored_calls"] = int(flags.sum())
        if worker._capo_progress and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"CAPO_UPDATE_FINISHED role={type(worker).__name__} forwards={len(weights)} "
                f"seconds={time.monotonic() - worker._capo_progress[0]:.1f}",
                flush=True,
            )
        return result
    finally:
        worker._capo_loss_weights = None
        worker._capo_progress = None
        _loss_weight.reset(token)


def before_forward(worker, micro_batch):
    weights = getattr(worker, "_capo_loss_weights", None)
    if weights is not None:
        rows = micro_batch["responses"].shape[0]
        if len(weights) < rows:
            raise RuntimeError("Unexpected extra CAPO update forward")
        progress = getattr(worker, "_capo_progress", None)
        if progress and (progress[1] - len(weights)) % 16 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"CAPO_UPDATE_PROGRESS role={type(worker).__name__} "
                f"forwards={progress[1] - len(weights)}/{progress[1]} "
                f"seconds={time.monotonic() - progress[0]:.1f}",
                flush=True,
            )
        if rows == worker.config.ppo_micro_batch_size_per_gpu == 1:
            _loss_weight.set(weights.popleft())
        else:
            # The copied loop divides by nominal accumulation, while agg_loss
            # averages the actual microbatch rows. Compensate a short microbatch
            # and assign zero to dummy rows without changing either copied loop.
            scale = rows / worker.config.ppo_micro_batch_size_per_gpu
            _loss_weight.set([weights.popleft() * scale for _ in range(rows)])


def register_aggregation():
    from .vendor.capo import dp_actor, verl_core_algos

    original = verl_core_algos.agg_loss
    if getattr(original, "_capo_padding_adapter", False):
        return

    def aggregate(loss_mat, loss_mask, loss_agg_mode, *args, **kwargs):
        weight = _loss_weight.get()
        if isinstance(weight, list):
            if loss_agg_mode != "seq-mean-token-mean" or loss_mat.shape[0] != len(weight):
                raise ValueError("CAPO microbatch weights require the supported call-mean loss layout")
            loss_mat = loss_mat * loss_mat.new_tensor(weight).unsqueeze(-1)
            return original(loss_mat, loss_mask, loss_agg_mode, *args, **kwargs)
        return original(loss_mat, loss_mask, loss_agg_mode, *args, **kwargs) * weight

    cast(Any, aggregate)._capo_padding_adapter = True
    verl_core_algos.agg_loss = aggregate
    dp_actor.agg_loss = aggregate
