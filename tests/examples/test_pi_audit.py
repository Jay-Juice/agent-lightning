"""Controlled-state audit regressions that do not require Docker or a model."""

import io
import json
import tarfile
from types import SimpleNamespace
from typing import ClassVar

from examples.multiturn_ppo.audit_pi_v2 import _scenario_edits, _write_files


def test_large_audit_edits_do_not_pass_source_contents_in_exec_arguments():
    class Container:
        def put_archive(self, path, content):
            assert path == "/tmp"
            with tarfile.open(fileobj=io.BytesIO(content)) as archive:
                self.payload = json.load(archive.extractfile(archive.getmembers()[0]))
            return True

        def exec_run(self, command, **kwargs):
            assert max(map(len, command)) < 4096
            assert kwargs["workdir"] == "/testbed"
            return SimpleNamespace(exit_code=0)

    container = Container()
    edits = {"large.py": "x = 1\n" * 100_000, "new.py": None}
    _write_files(container, edits)
    assert container.payload == edits


def test_audit_covers_one_line_multiple_files_and_new_helper_without_mutating_s0():
    sources = {"a.py": "line1\nline2\nline3\n", "b.py": "b\n", "c.py": "c\n"}
    scenarios = list(_scenario_edits("a.py", sources, "CONTROL"))
    assert [row[0] for row in scenarios] == ["large_file_one_line", "multiple_files", "new_helper_file"]
    large = scenarios[0][1]["a.py"].splitlines()
    assert sum(a != b for a, b in zip(large, sources["a.py"].splitlines(), strict=True)) == 1
    assert set(scenarios[1][1]) == set(sources)
    assert set(scenarios[2][1]).isdisjoint(sources)
    assert sources["a.py"] == "line1\nline2\nline3\n"
    for _, edits, markers in scenarios:
        assert all(any(marker in text for text in edits.values()) for marker in markers)


def test_source_selection_respects_snapshot_scope():
    from examples.multiturn_ppo.audit_pi_v2 import _select_sources

    class Store:
        max_text_bytes = 2 << 20
        entries: ClassVar[dict] = {
            "excluded.py": {"mode": "100644", "size": 10000},
            **{path: {"mode": "100644", "size": 1000} for path in ("a.py", "b.py", "c.py")},
        }

        def read_text(self, path):
            assert path != "excluded.py"
            return {"kind": "text", "text": "x\n" * 500}

    initial = {path: {} for path in ("a.py", "b.py", "c.py")}
    _, sources = _select_sources(Store(), len, 100, initial)
    assert set(sources) == set(initial)
