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


@pytest.mark.parametrize("exit_code", [3, 124, 137, -1])
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Linux agent deadline")
def test_unresolved_exit_never_resamples_or_regrades(helpers, tmp_path, exit_code):
    calls = []

    def grade(row, patch, target):
        calls.append(patch)
        return {"pytest_exit": exit_code, "reward": 0}

    result = helpers.grade_fixed_patch(grade, {}, "candidate", tmp_path,
                                      retry_errors=(ConnectionError,), deadline=time.monotonic() + 10)
    assert calls == ["candidate"] and result["grading_status"] == "requires_review"
