"""Actual optimizer hooks and a two-rank CPU nonfinite-stop regression."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")

from agentlightning.verl.capo_padding import reliable_optimizer_audit  # noqa: E402


class Worker:
    def __init__(self):
        self.actor_module = torch.nn.Linear(1, 1, bias=False)
        self.actor_module.weight.data.fill_(1.)
        self.actor_optimizer = torch.optim.AdamW(self.actor_module.parameters(), lr=.01)

    def _optimizer_step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=1.)
        if torch.isfinite(norm):
            self.actor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()
        return norm


class SkippingWorker(Worker):
    def _optimizer_step(self):
        # Model a scaler silently skipping the real optimizer despite a finite norm.
        return torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=1.)


class FusedSkipOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})
        self.found_inf = torch.tensor(1.)

    def step(self, closure=None):
        # Same visible convention as fused AMP: step is entered, no update.
        return None


def test_actual_post_hook_counts_preserve_numerics_and_cleanup(monkeypatch):
    monkeypatch.setenv("AGL_SWE_RELIABILITY", "1")
    worker, reference = Worker(), Worker()
    original_clip = torch.nn.utils.clip_grad_norm_
    with reliable_optimizer_audit(worker) as metrics:
        for gradient in (4., -3.):
            worker.actor_module.weight.grad = torch.full_like(worker.actor_module.weight, gradient)
            reference.actor_module.weight.grad = torch.full_like(reference.actor_module.weight, gradient)
            actual_norm = worker._optimizer_step()
            expected_norm = reference._optimizer_step()
            torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
    torch.testing.assert_close(worker.actor_module.weight, reference.actor_module.weight, rtol=0, atol=0)
    assert metrics["reliability/actor/optimizer_steps_attempted"] == 2
    assert metrics["reliability/actor/optimizer_steps_actual"] == 2
    assert metrics["reliability/actor/optimizer_steps_skipped"] == 0
    assert worker.actor_optimizer.state[worker.actor_module.weight]["step"] == 2
    assert "_optimizer_step" not in vars(worker)
    assert torch.nn.utils.clip_grad_norm_ is original_clip
    assert not worker.actor_optimizer._optimizer_step_post_hooks
    with reliable_optimizer_audit(worker) as metrics2:
        worker.actor_module.weight.grad = torch.ones_like(worker.actor_module.weight)
        worker._optimizer_step()
    assert metrics2["reliability/actor/optimizer_steps_actual"] == 1
    assert metrics2["reliability/actor/optimizer_steps_actual_worker_total"] == 3


def test_default_off_leaves_normal_optimizer_behavior(monkeypatch):
    monkeypatch.delenv("AGL_SWE_RELIABILITY", raising=False)
    worker = Worker()
    worker.actor_module.weight.grad = torch.full_like(worker.actor_module.weight, float("nan"))
    with reliable_optimizer_audit(worker) as metrics:
        worker._optimizer_step()
    assert metrics == {}
    assert not hasattr(worker, "_reliability_optimizer_totals")
    assert worker.actor_module.weight.item() == 1.


def test_nonfinite_prevents_step_and_records_skip_on_exception(monkeypatch):
    monkeypatch.setenv("AGL_SWE_RELIABILITY", "1")
    worker = Worker()
    original_clip = torch.nn.utils.clip_grad_norm_
    worker.actor_module.weight.grad = torch.full_like(worker.actor_module.weight, float("nan"))
    with pytest.raises(FloatingPointError, match="before step"), reliable_optimizer_audit(worker) as metrics:
        worker._optimizer_step()
    assert worker.actor_module.weight.item() == 1.
    assert not worker.actor_optimizer.state
    assert metrics["reliability/actor/optimizer_steps_actual"] == 0
    assert metrics["reliability/actor/optimizer_steps_skipped"] == 1
    assert metrics["reliability/actor/optimizer_steps_attempted"] == 1
    assert torch.nn.utils.clip_grad_norm_ is original_clip


def test_silent_skip_is_counted_and_stops(monkeypatch):
    monkeypatch.setenv("AGL_SWE_RELIABILITY", "1")
    worker = SkippingWorker()
    worker.actor_module.weight.grad = torch.ones_like(worker.actor_module.weight)
    with pytest.raises(FloatingPointError, match="skipped"), reliable_optimizer_audit(worker) as metrics:
        worker._optimizer_step()
    assert metrics["reliability/actor/optimizer_steps_attempted"] == 1
    assert metrics["reliability/actor/optimizer_steps_actual"] == 0
    assert metrics["reliability/actor/optimizer_steps_skipped"] == 1


def test_fused_amp_internal_skip_is_not_counted_as_actual(monkeypatch):
    monkeypatch.setenv("AGL_SWE_RELIABILITY", "1")
    worker = Worker()
    worker.actor_optimizer = FusedSkipOptimizer(worker.actor_module.parameters())
    worker.actor_module.weight.grad = torch.ones_like(worker.actor_module.weight)
    with pytest.raises(FloatingPointError, match="skipped"), reliable_optimizer_audit(worker) as metrics:
        worker._optimizer_step()
    assert metrics["reliability/actor/optimizer_steps_actual"] == 0
    assert metrics["reliability/actor/optimizer_steps_skipped"] == 1


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_actual_copied_worker_optimizer_method_is_instrumented(monkeypatch, role):
    from agentlightning.verl.vendor.capo.dp_actor import DataParallelPPOActor
    from agentlightning.verl.vendor.capo.dp_critic import DataParallelPPOCritic

    monkeypatch.setenv("AGL_SWE_RELIABILITY", "1")
    cls = DataParallelPPOActor if role == "actor" else DataParallelPPOCritic
    worker = object.__new__(cls)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.)
    model.weight.grad = torch.full_like(model.weight, 2.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    setattr(worker, f"{role}_module", model)
    setattr(worker, f"{role}_optimizer", optimizer)
    worker.config = SimpleNamespace(grad_clip=1.)
    worker.scaler = None
    with reliable_optimizer_audit(worker) as metrics:
        norm = worker._optimizer_step()
    assert norm.item() == 2.
    assert optimizer.state[model.weight]["step"] == 1
    assert metrics[f"reliability/{role}/optimizer_steps_actual"] == 1
    assert metrics[f"reliability/{role}/optimizer_steps_skipped"] == 0


def _distributed_nonfinite_worker(rank, rendezvous, output):
    torch.distributed.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    os.environ["AGL_SWE_RELIABILITY"] = "1"
    try:
        worker = Worker()
        gradient = float("nan") if rank == 0 else 2.
        worker.actor_module.weight.grad = torch.full_like(worker.actor_module.weight, gradient)
        try:
            with reliable_optimizer_audit(worker) as metrics:
                worker._optimizer_step()
        except FloatingPointError as error:
            assert "before step" in str(error)
        else:
            raise AssertionError("Both ranks must stop, including the locally finite rank")
        assert worker.actor_module.weight.item() == 1.
        assert not worker.actor_optimizer.state
        Path(output, f"rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo CPU process group unavailable")
def test_one_bad_rank_stops_all_ranks_before_any_update(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_nonfinite_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result["reliability/actor/optimizer_steps_actual"] == 0
        assert result["reliability/actor/optimizer_steps_skipped"] == 1
