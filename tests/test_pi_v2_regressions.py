import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentlightning.privileged_state import PreparedBaselineBlobStore, budgeted_semantic_state_text


def test_partial_hunk_is_truncated_and_advertises_missing_lines():
    delta = {
        "files": [
            {
                "path": "new.py",
                "change": "added",
                "kind": "text",
                "size": 1000,
                "hunks": [{"header": "@@ -0,0 +1,100 @@", "lines": [f"+line_{i}" for i in range(100)]}],
            }
        ]
    }
    text, stats = budgeted_semantic_state_text(delta, token_length=len, max_tokens=700)
    assert 0 < stats["changed_lines_included"] < stats["changed_lines_total"]
    assert stats["truncated"] and "[OMITTED]" in text
    assert stats["hunks_fully_included"] == 0


def test_single_long_line_keeps_fragments_without_claiming_complete_line():
    delta = {
        "files": [
            {
                "path": "new.py",
                "change": "added",
                "kind": "text",
                "size": 10000,
                "hunks": [{"header": "@@ -0,0 +1 @@", "lines": ["+" + "x" * 10000]}],
            }
        ]
    }
    text, stats = budgeted_semantic_state_text(delta, token_length=len, max_tokens=1000)
    assert "[PATCH new.py" in text and "fragment=" in text
    assert stats["hunk_chunks_included"] > 0
    assert stats["changed_lines_included"] == 0 and stats["truncated"]


def test_partial_manifest_is_preserved_and_counted_as_truncated():
    delta = {"files": [{"path": f"f{i}.bin", "change": "added", "kind": "metadata", "hunks": []} for i in range(100)]}
    text, stats = budgeted_semantic_state_text(delta, token_length=len, max_tokens=500)
    assert "[FILE MANIFEST]" in text
    assert 0 < stats["manifest_records_included"] < 100
    assert stats["truncated"]


def test_frozen_blob_map_preserves_whitespace_and_rejects_ref_access():
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        payload = b"100644 blob " + b"b" * 40 + b" 4\tspace and\nnewline.py\0" if "ls-tree" in args else b"x=1\n"
        return SimpleNamespace(exit_code=0, output=payload)

    store = PreparedBaselineBlobStore(SimpleNamespace(exec_run=execute), exact_commit="a" * 40)
    store.initialize()
    assert store.read_text("space and\nnewline.py")["text"] == "x=1\n"
    assert calls[-1][-3:] == ["cat-file", "blob", "b" * 40]
    assert store.read_text("missing.py")["kind"] == "missing"
    assert len(calls) == 2
    for path in ("../x", "/x", "a/../x", "a//x", "./x"):
        with pytest.raises(ValueError):
            store.read_text(path)


def test_recipe_guard_rejects_missing_agent_exports(tmp_path):
    import json

    path = Path(__file__).parents[1] / "examples/multiturn_ppo/matched_recipe.py"
    spec = importlib.util.spec_from_file_location("matched_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "trainer": {"resume_mode": "disable", "total_training_steps": 784},
        "agentlightning": {"privileged_critic": {"enabled": True}},
    }
    (tmp_path / "resolved-config.json").write_text(json.dumps(config))
    (tmp_path / "provenance.json").write_text(
        json.dumps({"runtime_options": {"smith_budgets": {"SMITH_MAX_TURNS": "32"}, "local_runner_maximum_size": 32}})
    )
    (tmp_path / "datasets.json").write_text(json.dumps({"train": [], "validation": []}))
    rows = tmp_path / "rows.jsonl"
    rows.write_text("")
    env = {
        "AGL_MATCH_BASELINE_RUN": str(tmp_path),
        "AGL_PI_TRAINING_STEPS": "784",
        "AGL_MAX_LOCAL_AGENTS": "32",
        "AGL_GPU_MONITOR": "1",
    }
    with pytest.raises(ValueError, match="SMITH_MAX_TURNS"):
        module.verify_matched_recipe(config, env, rows, rows)
    env["SMITH_MAX_TURNS"] = "32"
    assert module.verify_matched_recipe(config, env, rows, rows)["matched"]
