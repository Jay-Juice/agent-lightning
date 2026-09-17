# Copyright (c) Microsoft. All rights reserved.
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from agentlightning.verl.trainer import AgentLightningRayPPOTrainer


def trainer(tmp_path):
    worker = object.__new__(AgentLightningRayPPOTrainer)
    worker.config = OmegaConf.create({
        "agentlightning": {"reliability": {"enabled": True, "checkpoint_reserve_gib": 112}},
        "trainer": {"default_local_dir": str(tmp_path), "resume_mode": "auto", "resume_from_path": None},
    })
    worker.global_steps, worker.epoch = 20, 2
    worker._behavior_initial_tokens = 1234
    worker._update_phase = "complete"
    worker.use_critic = True
    return worker


def fake_parent_save(worker):
    folder = __import__("pathlib").Path(worker.config.trainer.default_local_dir) / f"global_step_{worker.global_steps}"
    (folder / "actor").mkdir(parents=True)
    (folder / "critic").mkdir()
    (folder / "data.pt").write_bytes(b"parent-dataloader-state")


def test_completion_marker_and_resume_restore_epoch_and_initial_baseline(monkeypatch, tmp_path):
    import agentlightning.verl.trainer as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * 2**30))
    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", fake_parent_save)
    worker = trainer(tmp_path)
    worker._save_checkpoint()
    # Interrupted newer checkpoint has no completion marker. Auto resume must
    # select the previous complete actor/critic/dataloader boundary.
    (tmp_path / "global_step_40/actor").mkdir(parents=True)
    selected = []

    def parent_load(value):
        selected.append(value.config.trainer.resume_from_path)
        value.global_steps = 20

    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", parent_load)
    worker.global_steps, worker.epoch, worker._behavior_initial_tokens = 0, 0, None
    worker._load_checkpoint()
    assert selected == [str(tmp_path / "global_step_20")]
    assert worker.epoch == 2 and worker._behavior_initial_tokens == 1234
    assert worker.config.trainer.resume_mode == "auto" and worker.config.trainer.resume_from_path is None


@pytest.mark.parametrize("phase,free", [("actor_update", 200), ("complete", 50)])
def test_partial_update_and_low_disk_never_invoke_parent_save(monkeypatch, tmp_path, phase, free):
    import agentlightning.verl.trainer as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=free * 2**30))
    calls = []
    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", lambda self: calls.append(1))
    worker = trainer(tmp_path)
    worker._update_phase = phase
    with pytest.raises(RuntimeError):
        worker._save_checkpoint()
    assert not calls


def test_behavior_stop_saves_only_after_completed_update(monkeypatch, tmp_path):
    worker = trainer(tmp_path)
    calls = []
    monkeypatch.setattr(worker, "_save_checkpoint", lambda: calls.append("save"))
    values = {"episodes": 470, "submitted_episode_ratio": 0, "format_stop_episode_ratio": .9,
              "length_episode_ratio_available": 1, "length_episode_ratio": .9,
              "output_tokens_per_episode/mean": 1234}
    metrics = {"val/behavior/" + key: value for key, value in values.items()}
    with pytest.raises(RuntimeError, match="behavior gate"):
        worker._check_behavior(metrics, initial=True)
    assert not calls
    with pytest.raises(RuntimeError, match="behavior gate"):
        worker._check_behavior(metrics)
    assert calls == ["save"]
    report = json.loads((tmp_path / "reliability-failure-step-20.json").read_text())
    assert report["phase"] == "complete" and len(report["details"]) == 3


def test_failed_attempt_events_survive_episode_replacement(tmp_path):
    worker = trainer(tmp_path)
    failed = SimpleNamespace(rollout_state="failed", rollout_id="first-attempt", events=[{
        "event_type": "model_request", "data": {"logical_call_id": "partial", "http_status": 500},
    }])
    worker._persist_failed_rollout(failed)
    saved = json.loads((tmp_path / "episode-audit/first-attempt.failed-events.json").read_text())
    assert saved["events"] == failed.events
