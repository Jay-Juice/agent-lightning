# Copyright (c) Microsoft. All rights reserved.

"""Keep CAPO's copied update loops, but give duplicate padding zero loss weight.

Loss weights normalize over real calls
in the entire optimizer minibatch, accounting for FSDP gradient averaging and
the copied loop's fixed gradient-accumulation divisor, including a short tail.
"""

import os
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from verl.utils.device import get_device_id

_loss_weight: ContextVar[float | list[float]] = ContextVar("capo_padding_loss_weight", default=1.0)


def _all_rank_bounds(value: int) -> tuple[int, int]:
    """Every rank participates, including ranks about to abort or skip."""
    if not dist.is_initialized():
        return value, value
    # Gloo CPU tests and NCCL production both need a supported collective device.
    collective_device = get_device_id() if dist.get_backend() == "nccl" else "cpu"
    low = torch.tensor(value, dtype=torch.int64, device=collective_device)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    return int(low.item()), int(high.item())


@contextmanager
def _guard_original_clipping(worker, original_step, module):
    """Guard the exact clip implementation called by CAPO, without replacing it.

    FSDP1 clips using its distributed norm; FSDP2 uses the copied worker's
    imported helper; unsharded CPU tests use torch's normal clip function.
    The consensus happens *after* clipping collectives and *before* any rank
    reaches optimizer.step, so one bad rank cannot leave peers in the next
    FSDP forward/backward. No norm or clipping coefficient is recomputed.
    """
    namespace = original_step.__func__.__globals__
    if "FSDP" in namespace and isinstance(module, namespace["FSDP"]):
        owner, name = module, "clip_grad_norm_"
    elif "FSDPModule" in namespace and isinstance(module, namespace["FSDPModule"]):
        owner, name = namespace, "fsdp2_clip_grad_norm_"
    else:
        owner, name = torch.nn.utils, "clip_grad_norm_"
    dictionary = isinstance(owner, dict)
    original = owner[name] if dictionary else getattr(owner, name)
    had_instance_attr = dictionary or name in vars(owner)

    def guarded_clip(*args, **kwargs):
        norm = original(*args, **kwargs)
        # DTensor norms require collective materialization on every rank.
        local_norm = norm.full_tensor() if hasattr(norm, "full_tensor") else norm
        norm_tensor = torch.as_tensor(local_norm)
        finite = int(torch.isfinite(norm_tensor).all().item())
        low, _ = _all_rank_bounds(finite)
        if not low:
            raise FloatingPointError("Reliability stop: nonfinite gradient norm on at least one rank before step")
        return norm

    if dictionary:
        owner[name] = guarded_clip
    else:
        setattr(owner, name, guarded_clip)
    try:
        yield
    finally:
        if dictionary:
            owner[name] = original
        elif had_instance_attr:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


@contextmanager
def reliable_optimizer_audit(worker):
    """Opt-in instrumentation around the unchanged copied optimizer routine.

    Post hooks count real optimizer.step calls, not minibatches or scaler.step
    attempts. Totals are per worker process, not checkpoint optimizer counters.
    Workers perform updates serially: the temporary clip hook is not intended
    for concurrent optimizers in the same process. Nonfinite clipping aborts
    collectively before updates; a later silent scaler skip aborts after all
    ranks return from the step routine and its counts are recorded.
    """
    if os.environ.get("AGL_SWE_RELIABILITY", "0") != "1":
        yield {}
        return
    role = "actor" if getattr(worker, "actor_optimizer", None) is not None else "critic"
    optimizer = getattr(worker, f"{role}_optimizer")
    module = getattr(worker, f"{role}_module")
    original_step = worker._optimizer_step
    had_instance_step = "_optimizer_step" in vars(worker)
    totals = getattr(worker, "_reliability_optimizer_totals", None)
    if totals is None:
        totals = {"attempted": 0, "actual": 0, "skipped": 0}
        worker._reliability_optimizer_totals = totals
    before = dict(totals)
    metrics = {}

    def after_step(optimizer, args, kwargs):
        # Fused AMP optimizers may enter step() but skip internally. GradScaler
        # attaches found_inf until after step returns, so it is visible here.
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is not None and bool(torch.as_tensor(found_inf).ne(0).any().item()):
            return
        totals["actual"] += 1

    handle = optimizer.register_step_post_hook(after_step)

    def audited_step():
        totals["attempted"] += 1
        actual_before = totals["actual"]
        try:
            with _guard_original_clipping(worker, original_step, module):
                norm = original_step()
            actual = totals["actual"] - actual_before
            low, high = _all_rank_bounds(actual)
            if low != 1 or high != 1:
                raise FloatingPointError(
                    f"Reliability stop: optimizer step skipped or inconsistent across ranks (min={low}, max={high})"
                )
            return norm
        finally:
            if totals["actual"] == actual_before:
                totals["skipped"] += 1

    worker._optimizer_step = audited_step
    try:
        yield metrics
    finally:
        handle.remove()
        if had_instance_step:
            worker._optimizer_step = original_step
        else:
            del worker._optimizer_step
        for key in totals:
            metrics[f"reliability/{role}/optimizer_steps_{key}"] = totals[key] - before[key]
            metrics[f"reliability/{role}/optimizer_steps_{key}_worker_total"] = totals[key]
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"RELIABILITY_OPTIMIZER_COUNTS role={role} counts={metrics}", flush=True)


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
        with reliable_optimizer_audit(worker) as optimizer_metrics:
            result = copied_update(data)
        result.update(optimizer_metrics)
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
