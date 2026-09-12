# Copyright (c) Microsoft. All rights reserved.

"""Shared isolated Docker sandbox, migrated from the verified A800 pilot helper."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from pathlib import Path, PurePosixPath

SMITH_SHA256 = "72a6b8950578f929700918127e5da15c4b7051ef906a0532efe53546e78fe433"
SMITH_PATH = Path(__file__).resolve().parents[1] / "swe_smith/agents/smith_agent.py"
PRIVATE_GIT = "/root/agl-private-git"
AGENT_UID = "65534:65534"
TASK_KEYS = {"instance_id", "problem_statement", "image"}


def validate_task(task: dict) -> None:
    if set(task) != TASK_KEYS:
        raise ValueError("Agent task must contain only instance_id, problem_statement, image")
    if not all(isinstance(task[k], str) and task[k] for k in TASK_KEYS):
        raise ValueError("Task fields must be nonempty strings")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", task["instance_id"]):
        raise ValueError("Invalid instance identifier")


def forbidden_patch_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or ".." in parts:
        return True
    protected = {
        "tests",
        "test",
        "conftest.py",
        "pytest.ini",
        "tox.ini",
        "pyproject.toml",
        "setup.cfg",
        "sitecustomize.py",
        "usercustomize.py",
    }
    return any(p in protected or p.startswith((".git", "test_")) or p.endswith(("_test.py", ".pth")) for p in parts)


def load_smith():
    if hashlib.sha256(SMITH_PATH.read_bytes()).hexdigest() != SMITH_SHA256:
        raise RuntimeError("Smith agent source changed; review before reusing its prompt helpers")
    spec = importlib.util.spec_from_file_location("agl_smith_prompt_snapshot", SMITH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def split_export_paths(paths, tracked, *, allow_new_repro=False, allowed_protected=()):
    """Omit newly created reproduction scripts without accepting test tampering.

    New scripts are never copied into the fresh grading container. Existing tests
    and every harness/configuration change remain prohibited.
    """
    excluded, blocked = [], []
    harness = {"conftest.py", "sitecustomize.py", "usercustomize.py"}
    for path in paths:
        if path in allowed_protected:
            continue
        if not forbidden_patch_path(path):
            continue
        parts = PurePosixPath(path).parts
        repro = (
            allow_new_repro
            and path not in tracked
            and path.endswith(".py")
            and not path.startswith("/")
            and ".." not in parts
            and any(p in {"tests", "test"} or p.startswith("test_") or p.endswith("_test.py") for p in parts)
            and not any(p in harness or p.startswith(".git") for p in parts)
        )
        (excluded if repro else blocked).append(path)
    return excluded, blocked


class Sandbox:
    def container_options(self, task):
        return {}

    def __init__(self, client, task, run_id):
        self.client = client
        limits = {"mem_limit": "4g", "nano_cpus": 2_000_000_000, "pids_limit": 256}
        limits.update(self.container_options(task))
        self.container = client.containers.run(
            task["image"],
            command=["/bin/bash", "-c", "exec sleep infinity"],
            detach=True,
            network_mode="none",
            **limits,
            security_opt=["no-new-privileges:true"],
            labels={
                "agl.purpose": "swe-agent-pilot",
                "agl.run_id": run_id,
                "agl.instance_id": task["instance_id"],
            },
        )

    def root(self, argv):
        result = self.container.exec_run(argv, user="0:0", workdir="/testbed")
        output = result.output.decode("utf-8", errors="replace")
        if result.exit_code:
            raise RuntimeError(f"Sandbox preparation/export failed: {output[-2000:]}")
        return output

    def git(self, *args):
        return self.root(
            [
                "git",
                "-c",
                "safe.directory=/testbed",
                "--git-dir",
                PRIVATE_GIT,
                "--work-tree",
                "/testbed",
                *args,
            ]
        )

    def prepare(self):
        self.root(
            [
                "/bin/bash",
                "-lc",
                "set -eu; test -d /testbed/.git; "
                "test ! -e /root/agl-private-git; chmod 700 /root; "
                "mv /testbed/.git /root/agl-private-git; "
                "chown -R 65534:65534 /testbed; "
                "mkdir -p /tmp/agl-home; chown 65534:65534 /tmp/agl-home",
            ]
        )
        # Keep the prebuilt image's prepared checkout: do not reset to the original
        # base_commit, which would restore grading-only tests removed by the image.
        baseline = self.git("rev-parse", "HEAD").strip()
        if self.git("status", "--porcelain").strip():
            raise RuntimeError("Image checkout is not clean; do not mix image changes into a model patch")
        rc, _ = self.execute("test $(id -u) = 65534 && test ! -e /testbed/.git && test ! -r /root/agl-private-git/HEAD")
        if rc:
            raise RuntimeError("Agent user can access Git history or has unexpected privileges")
        self.container.reload()
        cfg = self.container.attrs["HostConfig"]
        assert cfg["NetworkMode"] == "none"
        assert not cfg.get("Privileged") and not cfg.get("Binds")
        return {
            "image_id": self.container.image.id,
            "baseline_head": baseline,
            "network": "none",
            "agent_uid": AGENT_UID,
            "git_history_accessible": False,
        }

    def execute(self, action):
        timeout = int(os.environ.get("SMITH_CMD_TIMEOUT", "45"))
        if timeout <= 0:
            raise ValueError("Command timeout must be positive")
        execution = self.client.api.exec_create(
            self.container.id,
            ["/usr/bin/timeout", "--kill-after=3", str(timeout), "/bin/bash", "-c", action],
            user=AGENT_UID,
            workdir="/testbed",
            environment={
                "HOME": "/tmp/agl-home",
                "PYTHONPATH": "/testbed",
                "PATH": "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin",
                "CONDA_PREFIX": "/opt/miniconda3/envs/testbed",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PAGER": "cat",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            },
        )
        head, tail, total = b"", b"", 0
        for chunk in self.client.api.exec_start(execution["Id"], stream=True):
            total += len(chunk)
            if len(head) < 65536:
                head += chunk[: 65536 - len(head)]
            tail = (tail + chunk)[-65536:]
        if total <= 65536:
            output = head
        elif total <= 131072:
            output = head + tail[-(total - 65536) :]
        else:
            output = head + b"\n[output truncated by host]\n" + tail
        rc = self.client.api.exec_inspect(execution["Id"])["ExitCode"]
        if rc is None:
            raise RuntimeError("Command did not terminate after timeout")
        return rc, output.decode("utf-8", errors="replace")

    def allowed_protected_paths(self, paths):
        return set()

    def export_patch(self, *, allow_new_repro=False):
        self.git("add", "-A")
        paths = self.git("diff", "--cached", "--name-only", "-z").split("\0")
        paths = [p for p in paths if p]
        tracked = set(self.git("ls-tree", "-r", "--name-only", "-z", "HEAD").split("\0"))
        self.excluded_patch_paths, blocked = split_export_paths(
            paths, tracked, allow_new_repro=allow_new_repro, allowed_protected=self.allowed_protected_paths(paths)
        )
        if blocked:
            return "", paths, "forbidden_test_or_config_change"
        raw = self.git("diff", "--cached", "--raw")
        if any("120000" in line.split("\t", 1)[0] for line in raw.splitlines()):
            return "", paths, "symlink_change_not_supported_in_pilot"
        included = [p for p in paths if p not in self.excluded_patch_paths]
        return (
            self.git(
                "diff",
                "--cached",
                "--binary",
                "--no-ext-diff",
                "--full-index",
                "--",
                *(":(literal)" + p for p in included),
            )
            if included
            else "",
            paths,
            None,
        )

    def close(self):
        # Only remove the exact container created by this object.
        self.container.remove(force=True)
