# Copyright (c) Microsoft. All rights reserved.
"""Bounded, fixed-candidate adjudication; never select a lucky retry reward."""

import hashlib
import json
from pathlib import Path

AMBIGUOUS_EXITS = frozenset({2, 3, 4, 5, 124, 137})


def task_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def memory_events(container):
    """Read the grading container's own cgroup, not host OOM heuristics."""
    result = container.exec_run([
        "/bin/sh", "-c",
        "if [ -r /sys/fs/cgroup/memory.events ]; then cat /sys/fs/cgroup/memory.events; "
        "elif [ -r /sys/fs/cgroup/memory/memory.oom_control ]; then "
        "cat /sys/fs/cgroup/memory/memory.oom_control; fi",
    ])
    if result.exit_code:
        return {}
    values = {}
    for line in result.output.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    return values


def host_memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def classify_controlled_failure(original, reference, replay):
    """Return a failure class only for comparable, independently replayed evidence.

    The caller runs all three against the same immutable task. Reports retain
    raw exit codes and scores. A successful replay is a disagreement, never a
    replacement success. Unknown/changed conditions remain unresolved.
    """
    code = original.get("pytest_exit")
    if code not in AMBIGUOUS_EXITS or replay.get("pytest_exit") != code:
        return None
    if any(r.get("reward") != 0 or r.get("resolved") is not False for r in (original, replay)):
        return None
    if reference.get("pytest_exit") != 0 or reference.get("reward") != 1 or reference.get("resolved") is not True:
        return None
    if reference.get("reference_control") is not True:
        return None
    if any(r.get("reference_control") is not False for r in (original, replay)):
        return None
    if not original.get("patch_sha256") or original["patch_sha256"] != replay.get("patch_sha256"):
        return None
    # Every report is tied to this task, image, exact test selection and limits.
    fields = ("task_sha256", "test_spec_sha256", "container_memory_bytes", "container_nano_cpus",
              "eval_timeout_seconds", "grading_protocol", "test_runner", "f2p_total", "p2p_total")
    for key in fields:
        if original.get(key) is None or any(r.get(key) != original[key] for r in (reference, replay)):
            return None
    image = original.get("baseline", {}).get("image_id")
    if not image or any(r.get("baseline", {}).get("image_id") != image for r in (reference, replay)):
        return None
    if (reference.get("f2p_passed") != reference["f2p_total"]
            or reference.get("p2p_passed") != reference["p2p_total"]):
        return None
    if code in {124, 137}:
        # Resource failures need bounded-container evidence as well as a control.
        limit = original["container_memory_bytes"]
        if not isinstance(limit, int) or limit <= 0:
            return None
        for report in (original, reference, replay):
            available = report.get("host_memory_available_before")
            if available is None or available < 2 * limit:
                return None
        if code == 137:
            # Do not infer OOM from an exit status or a global host event.
            if any(r.get("container_oom_kill_delta", 0) <= 0 for r in (original, replay)):
                return None
            return "candidate_container_oom"
        if any(r.get("test_elapsed_seconds", 0) < r["eval_timeout_seconds"] for r in (original, replay)):
            return None
        return "candidate_test_timeout"
    return "candidate_test_failure"
