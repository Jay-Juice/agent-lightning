# Copyright (c) Microsoft. All rights reserved.
"""Regression cases for the observed empty-edit and reproduction-file failures."""

import ast

import pytest

from examples.multiturn_ppo.smith_submission import submission_feedback, syntax_check_command


def test_empty_edit_is_not_a_submission():
    assert "no changes" in submission_feedback("", [], None)


def test_prohibited_patch_feedback_preserves_the_source_fix():
    feedback = submission_feedback("", ["src/foo.py", "test_repro.py"], "forbidden_test_or_config_change")
    assert "test_repro.py" in feedback
    assert "Keep your source fix" in feedback
    assert "/tmp" in feedback


def test_real_patch_is_graded_without_hinting_at_hidden_tests():
    assert submission_feedback("diff --git a/src/foo.py b/src/foo.py\n+fixed", ["src/foo.py"], None) is None


def test_syntax_check_only_parses_source_and_quotes_filenames(tmp_path):
    path = tmp_path / "source ' $(echo unexpected).py"
    path.write_text("raise RuntimeError('must not execute')\n")
    command = syntax_check_command([str(path), "deleted.py", "README.md"])
    program = command.split("\n", 1)[1].rsplit("\n", 1)[0]
    ast.parse(program)
    exec(program, {})
    path.write_text("def broken(:\n")
    with pytest.raises(SyntaxError):
        exec(program, {})
