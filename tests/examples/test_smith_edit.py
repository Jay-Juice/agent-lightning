# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path

import pytest

from examples.multiturn_ppo.smith_edit import parse_edit, replace


def test_edit_uses_literal_text_and_shows_actual_diff(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("def f(x: list[str]):\n    return x[0]\n")
    old, new = parse_edit("<<<<<<< SEARCH\n    return x[0]\n=======\n    return x[-1]\n>>>>>>> REPLACE\n")
    diff = replace(Path("code.py"), old, new, root=tmp_path)
    assert "-    return x[0]" in diff and "+    return x[-1]" in diff
    assert path.read_text() == "def f(x: list[str]):\n    return x[-1]\n"


@pytest.mark.parametrize("old", ["missing", "x", ""])
def test_failed_edit_does_not_write(tmp_path, old):
    path = tmp_path / "code.py"
    path.write_text("x x\n")
    with pytest.raises(ValueError):
        replace(Path("code.py"), old, "y", root=tmp_path)
    assert path.read_text() == "x x\n"


def test_cannot_edit_outside_task_checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (tmp_path / "outside.py").write_text("old\n")
    with pytest.raises(ValueError):
        replace(Path("../outside.py"), "old\n", "new\n", root=root)


@pytest.mark.parametrize("text", ["", "<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n", "<<<<<<< SEARCH\nx\n=======\ny\n"])
def test_malformed_edit_rejected(text):
    with pytest.raises(ValueError):
        parse_edit(text)
