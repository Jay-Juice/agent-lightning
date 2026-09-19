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
    if report.get("controlled_failure"):
        return "candidate_failed"
    # These can be caused by a bad patch OR infrastructure; require evidence.
    if report.get("pytest_exit") not in {0, 1}:
        return "requires_review"
    return "completed"


def grade_fixed_patch(grader, row, patch, directory, *, retry_errors, deadline):
    """Bounded grading with evidence for candidate failures.

    Ordinary 0/1 scores are final. Ambiguous exits get one reference control
    and one identical candidate replay. A disagreement never becomes a lucky
    success. Transport failures alone permit one additional grading attempt.
    Controls run after the agent sandbox closes and never enter its context.
    """
    from swe_grading_evidence import (
        AMBIGUOUS_EXITS,
        classify_controlled_failure,
        classify_replayed_candidate_syntax_error,
    )

    def run(name, *, reference=False):
        for attempt in range(2):
            target = directory / f"{name}-{attempt}"
            target.mkdir()
            try:
                with deadline_guard(deadline):
                    if reference:
                        report = grader(row, patch, target, reference=True)
                    else:
                        report = grader(row, patch, target)
                return report, attempt + 1
            except retry_errors as exc:
                (target / "infrastructure-error.json").write_text(json.dumps({"error_type": type(exc).__name__}))
                if attempt or time.monotonic() >= deadline:
                    raise

    report, attempts = run("grading-attempt")
    report = {**report, "grading_attempts": attempts}
    if report.get("pytest_exit") in AMBIGUOUS_EXITS:
        reference, _ = run("grading-reference", reference=True)
        replay, _ = run("grading-replay")
        evidence = {"reference": reference, "replay": replay}
        failure = None
        # An unhealthy reference cannot establish candidate failure.
        if reference.get("resolved") is True and reference.get("pytest_exit") == 0:
            failure = classify_controlled_failure(report, reference, replay)
        if not failure:
            first_output = (directory / "grading-attempt-0" / "test-output.txt")
            replay_output = (directory / "grading-replay-0" / "test-output.txt")
            if first_output.is_file() and replay_output.is_file():
                failure = classify_replayed_candidate_syntax_error(
                    report, replay, patch, first_output.read_text(errors="replace"),
                    replay_output.read_text(errors="replace"),
                )
        if failure:
            report["controlled_failure"] = failure
        (directory / "grading-adjudication.json").write_text(json.dumps(evidence, indent=2))
    report["grading_status"] = grading_status(report)
    (directory / "grade.json").write_text(json.dumps(report, indent=2))
    return report
