from agentlightning.privileged_state.semantic_diff import build_semantic_delta
from agentlightning.privileged_state.serialize import budgeted_semantic_state_text


class Store:
    def __init__(self, values):
        self.values = values

    def read_text(self, path):
        value = self.values.get(path)
        return {"kind": "text", "text": value, "blob_sha": "a" * 40} if value is not None else {"kind": "missing"}


def _raw(files):
    return {"filesystem": files, "processes": {}, "sockets": {}}


def test_large_file_single_changed_line_is_semantic_content():
    before = "\n".join([f"line-{i}" for i in range(5000)])
    after = before.replace("line-2500", "line-2500-fixed")
    row = {"change": "modified", "after": {"path": "src/a.py", "type": "file", "size": len(after), "content": after, "sha256": "b" * 64}}
    semantic = build_semantic_delta(_raw([row]), Store({"src/a.py": before}))
    text, stats = budgeted_semantic_state_text(semantic, token_length=lambda s: len(s.split()), max_tokens=4096)
    assert "line-2500-fixed" in text
    assert stats["changed_lines_included"] >= 1


def test_round_robin_keeps_short_hunks_in_multiple_files():
    rows = []
    for path, old, new in [("a.py", "a\n", "a\n" + "x\n" * 1000), ("b.py", "b\n", "b-fixed\n"), ("c.py", "c\n", "c-fixed\n")]:
        rows.append({"change": "modified", "after": {"path": path, "type": "file", "size": len(new), "content": new, "sha256": "b" * 64}})
    semantic = build_semantic_delta(_raw(rows), Store({"a.py": "a\n", "b.py": "b\n", "c.py": "c\n"}))
    text, _ = budgeted_semantic_state_text(semantic, token_length=lambda s: len(s), max_tokens=700)
    assert "b-fixed" in text and "c-fixed" in text


def test_added_and_removed_files_keep_patch_or_manifest():
    semantic = build_semantic_delta(
        _raw([
            {"change": "added", "after": {"path": "new.py", "type": "file", "size": 8, "content": "x=1\n", "sha256": "b" * 64}},
            {"change": "removed", "after": {"path": "old.py", "type": "file", "size": 4, "sha256": "c" * 64}},
        ]),
        Store({"old.py": "x=0\n"}),
    )
    text, _ = budgeted_semantic_state_text(semantic, token_length=len, max_tokens=4096)
    assert "A new.py" in text and "D old.py" in text


def test_frozen_commit_validation():
    from agentlightning.privileged_state.baseline import PreparedBaselineBlobStore

    class Container:
        pass

    for value in ("HEAD", "HEAD~1", "main", "f" * 39, "g" * 40):
        try:
            PreparedBaselineBlobStore(Container(), exact_commit=value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
    PreparedBaselineBlobStore(Container(), exact_commit="a" * 40)
