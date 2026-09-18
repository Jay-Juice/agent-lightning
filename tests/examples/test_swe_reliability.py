# Copyright (c) Microsoft. All rights reserved.
import importlib
import signal
import time
from pathlib import Path

import pytest


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples" / "multiturn_ppo"))
    return importlib.import_module("swe_reliability")


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Linux agent deadline")
def test_nested_deadline_cannot_extend_outer(helpers):
    with (pytest.raises(helpers.EpisodeDeadline),
          helpers.deadline_guard(time.monotonic() + .1),
          helpers.deadline_guard(time.monotonic() + 10)):
        time.sleep(.3)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Linux agent deadline")
def test_fixed_patch_retries_only_transport_errors(helpers, tmp_path):
    calls = []

    def grade(row, patch, target):
        calls.append((row, patch, target.name))
        if len(calls) == 1:
            raise ConnectionError("daemon unavailable")
        return {"pytest_exit": 1, "reward": 0}

    result = helpers.grade_fixed_patch(grade, {"task": 1}, "same patch", tmp_path,
                                      retry_errors=(ConnectionError,), deadline=time.monotonic() + 10)
    assert result["grading_attempts"] == 2 and result["reward"] == 0
    assert [c[:2] for c in calls] == [({"task": 1}, "same patch")] * 2
    assert (tmp_path / "grading-attempt-0/infrastructure-error.json").exists()


@pytest.mark.parametrize("exit_code", [-1, 125, 126, 127, 143])
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Linux agent deadline")
def test_unresolved_exit_never_resamples_or_regrades(helpers, tmp_path, exit_code):
    calls = []

    def grade(row, patch, target):
        calls.append(patch)
        return {"pytest_exit": exit_code, "reward": 0}

    result = helpers.grade_fixed_patch(grade, {}, "candidate", tmp_path,
                                      retry_errors=(ConnectionError,), deadline=time.monotonic() + 10)
    assert calls == ["candidate"] and result["grading_status"] == "requires_review"


def report(code=4, reference=False):
    return {"pytest_exit": code, "reward": float(code == 0), "resolved": code == 0,
            "reference_control": reference, "patch_sha256": "ref" if reference else "candidate",
            "task_sha256": "task", "test_spec_sha256": "tests", "baseline": {"image_id": "image"},
            "container_memory_bytes": 4 * 2**30, "container_nano_cpus": 2 * 10**9,
            "eval_timeout_seconds": 600, "grading_protocol": "f2p_file", "test_runner": "pytest",
            "f2p_total": 2, "p2p_total": 3, "f2p_passed": 2 if code == 0 else 0,
            "p2p_passed": 3 if code == 0 else 0, "host_memory_available_before": 64 * 2**30,
            "test_elapsed_seconds": 601 if code == 124 else 2, "container_oom_kill_delta": int(code == 137),
            "test_statuses": {"test_case": "PASSED" if code == 0 else "FAILED"}}


@pytest.mark.parametrize("code", [2, 3, 4, 5, 120, 124, 137])
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Linux agent deadline")
def test_control_and_fixed_replay_classify_failure(helpers, tmp_path, code):
    calls = []

    def grade(row, patch, target, *, reference=False):
        calls.append((patch, reference))
        return report(0 if reference else code, reference)

    result = helpers.grade_fixed_patch(grade, {}, "candidate", tmp_path,
                                      retry_errors=(ConnectionError,), deadline=time.monotonic() + 10)
    assert calls == [("candidate", False), ("candidate", True), ("candidate", False)]
    assert result["reward"] == 0 and result["grading_status"] == "candidate_failed"
    assert (tmp_path / "grading-adjudication.json").exists()


@pytest.mark.parametrize("mode", ["bad_reference", "successful_replay", "changed_failure"])
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Linux agent deadline")
def test_control_disagreement_never_selects_lucky_reward(helpers, tmp_path, mode):
    calls = []

    def grade(row, patch, target, *, reference=False):
        calls.append(reference)
        if reference:
            return report(1 if mode == "bad_reference" else 0, True)
        return report(4 if len(calls) == 1 else 0 if mode == "successful_replay" else 3)

    result = helpers.grade_fixed_patch(grade, {}, "candidate", tmp_path,
                                      retry_errors=(ConnectionError,), deadline=time.monotonic() + 10)
    assert result["reward"] == 0 and result["grading_status"] == "requires_review"
    assert len(calls) == (2 if mode == "bad_reference" else 3)


@pytest.mark.parametrize("change", ["patch", "image", "task", "tests", "limits", "missing", "oom", "original_oom",
                                   "host_memory", "short_timeout", "partial_reference", "fixture"])
def test_control_evidence_must_be_comparable(helpers, change):
    from swe_grading_evidence import classify_controlled_failure

    code = 137 if change in {"oom", "original_oom", "host_memory"} else 124 if change == "short_timeout" else 4
    original, reference, replay = report(code), report(0, True), report(code)
    if change == "patch":
        replay["patch_sha256"] = "other"
    elif change == "image":
        reference["baseline"]["image_id"] = "other"
    elif change == "task":
        replay["task_sha256"] = "other"
    elif change == "tests":
        reference["test_spec_sha256"] = "other"
    elif change == "limits":
        replay["eval_timeout_seconds"] = 1200
    elif change == "missing":
        del original["task_sha256"]
    elif change == "oom":
        replay["container_oom_kill_delta"] = 0
    elif change == "original_oom":
        original["container_oom_kill_delta"] = 0
    elif change == "host_memory":
        reference["host_memory_available_before"] = 0
    elif change == "short_timeout":
        replay["test_elapsed_seconds"] = 2
    elif change == "fixture":
        reference["recorded_pydicom_fixture"] = {"manifest_sha256": "different"}
    else:
        reference["p2p_passed"] = 0
    assert classify_controlled_failure(original, reference, replay) is None


@pytest.mark.parametrize("change", ["missing", "no_failure", "different", "different_nodes", "bad_control"])
def test_exit120_requires_matching_failed_tests(helpers, change):
    from swe_grading_evidence import classify_controlled_failure

    original, reference, replay = report(120), report(0, True), report(120)
    if change == "missing":
        original.pop("test_statuses")
    elif change == "no_failure":
        original["test_statuses"] = replay["test_statuses"] = {"test_case": "PASSED"}
    elif change == "different":
        replay["test_statuses"] = {"test_case": "PASSED"}
    elif change == "different_nodes":
        reference["test_statuses"] = {"other": "PASSED"}
    else:
        reference["test_statuses"] = {"test_case": "FAILED"}
    assert classify_controlled_failure(original, reference, replay) is None
