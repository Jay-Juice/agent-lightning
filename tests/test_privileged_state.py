from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

from agentlightning.privileged_state.sandbox import SandboxStateSnapshotter
from agentlightning.privileged_state.serialize import MARKER, budgeted_state_text


class FakeContainer:
    def __init__(self, states):
        self.states = iter(states)

    def exec_run(self, command, **kwargs):
        return SimpleNamespace(exit_code=0, output=json.dumps(next(self.states)).encode())

def _state(files, processes=None, sockets=None):
    return {"filesystem": files, "processes": processes or [], "sockets": sockets or []}


def test_cumulative_delta_and_content_addressed_storage(tmp_path):
    original = {
        "path": "src/a.py",
        "type": "file",
        "mode": 420,
        "size": 3,
        "mtime_ns": 1,
        "inode": 7,
        "sha256": "old",
    }
    modified = {
        "path": "src/a.py",
        "type": "file",
        "mode": 420,
        "size": 3,
        "mtime_ns": 2,
        "inode": 7,
        "sha256": "new",
        "content": "new",
    }
    modified_metadata = {k: modified[k] for k in ("path", "type", "mode", "size", "mtime_ns", "inode")}
    restored_metadata = {**modified_metadata, "mtime_ns": 3}
    restored_detail = {**original, "mtime_ns": 3}
    container = FakeContainer(
        [
            _state([original]),
            _state([modified_metadata]),
            _state([modified]),
            _state([restored_metadata]),
            _state([restored_detail]),
        ]
    )
    snapshotter = SandboxStateSnapshotter(container, tmp_path)
    snapshotter.initialize()
    changed = snapshotter.capture()
    assert changed["delta"]["filesystem"][0]["change"] == "modified"
    assert changed["delta"]["filesystem"][0]["after"]["content"] == "new"
    with gzip.open(changed["state_ref"], "rt", encoding="utf-8") as handle:
        assert json.load(handle) == changed["delta"]
    restored = snapshotter.capture()
    assert restored["delta"]["filesystem"] == []
    assert restored["state_hash"] != changed["state_hash"]


def test_deleted_directory_expands_baseline_descendants(tmp_path):
    baseline = [
        {"path": "pkg", "type": "directory", "mode": 493},
        {"path": "pkg/a.py", "type": "file", "mode": 420, "size": 1, "sha256": "a"},
    ]
    container = FakeContainer([_state(baseline), _state([]), _state([])])
    snapshotter = SandboxStateSnapshotter(container, tmp_path)
    snapshotter.initialize()
    delta = snapshotter.capture()["delta"]["filesystem"]
    assert [(row["change"], row["path"]) for row in delta] == [("removed", "pkg"), ("removed", "pkg/a.py")]


def test_budgeted_serializer_is_deterministic_and_bounded():
    delta = {
        "filesystem": [
            {
                "change": "modified",
                "before": {"path": "a.py", "type": "file", "size": 1},
                "after": {"path": "a.py", "type": "file", "size": 100, "content": "x" * 100},
            }
        ],
        "processes": {"added": [], "removed": []},
        "sockets": {"added": [], "removed": []},
    }
    def length(text):
        return len(text)

    first, stats = budgeted_state_text(delta, token_length=length, max_tokens=180)
    second, _ = budgeted_state_text(delta, token_length=length, max_tokens=180)
    assert first == second
    assert first.startswith(MARKER)
    assert len(first) <= 180
    assert stats["truncated"] is True


def test_budgeted_serializer_keeps_partial_manifest_before_content():
    delta = {
        "filesystem": [
            {"change": "added", "path": f"file-{index}.py", "type": "file", "size": 100, "content": "x" * 100}
            for index in range(20)
        ],
        "processes": {"added": [], "removed": []},
        "sockets": {"added": [], "removed": []},
    }
    text, stats = budgeted_state_text(delta, token_length=len, max_tokens=400)
    assert "[FILE MANIFEST]" in text
    assert "A file-0.py" in text
    assert "A file-19.py" not in text
    assert "[CURRENT FILE:" not in text
    assert stats["truncated"] is True


def test_budget_cannot_depend_on_post_action_fit():
    delta = {"filesystem": [], "processes": {"added": [], "removed": []}, "sockets": {"added": [], "removed": []}}
    text, stats = budgeted_state_text(delta, token_length=len, max_tokens=4096, fits=lambda value: len(value) < 10)
    assert text == ""
    assert stats["serialized_tokens"] == 0
