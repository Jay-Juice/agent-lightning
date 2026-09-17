# Copyright (c) Microsoft. All rights reserved.
"""Deadline and outcome helpers for isolated Linux SWE agent subprocesses."""

import json
import signal
import threading
import time
from contextlib import contextmanager


class EpisodeDeadline(TimeoutError):
    pass


@contextmanager
def deadline_guard(deadline):
    """Bound the whole operation, not each individual HTTP read/retry.

    Only used by opt-in isolated Linux agent processes. Nested guards restore
    the outer deadline without extending it.
    """
    if deadline is None:
        yield
        return
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("SWE deadline guard requires an isolated Linux main-thread agent")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise EpisodeDeadline("SWE operation deadline reached")
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def expired(signum, frame):
        raise EpisodeDeadline("SWE operation deadline reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, previous_timer[0]) if previous_timer[0] else remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, previous_timer[0] - (time.monotonic() - started)),
                             previous_timer[1])


def grading_status(report):
    if report.get("patch_rejection"):
        return "candidate_rejected"
    # These can be caused by a bad patch OR infrastructure; require evidence.
    if report.get("pytest_exit") not in {0, 1}:
        return "requires_review"
    return "completed"


def grade_fixed_patch(grader, row, patch, directory, *, retry_errors, deadline):
    """Retry only a transport/daemon failure, once, with identical patch bytes.

    Returned test failures (including ambiguous exit codes) are never retried.
    Separate directories preserve both attempts. A review-required result has
    no reward delivered to PPO until independently classified.
    """
    for attempt in range(2):
        target = directory / f"grading-attempt-{attempt}"
        target.mkdir()
        try:
            with deadline_guard(deadline):
                report = grader(row, patch, target)
            report = {**report, "grading_attempts": attempt + 1, "grading_status": grading_status(report)}
            (directory / "grade.json").write_text(json.dumps(report, indent=2))
            return report
        except retry_errors as exc:
            (target / "infrastructure-error.json").write_text(json.dumps({"error_type": type(exc).__name__}))
            if attempt or time.monotonic() >= deadline:
                raise
