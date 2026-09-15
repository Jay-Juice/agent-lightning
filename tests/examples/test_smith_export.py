# Copyright (c) Microsoft. All rights reserved.

from examples.multiturn_ppo.sandbox import split_export_paths


def test_new_reproduction_script_is_excluded_but_source_is_kept():
    paths = ["src/package.py", "test_repro.py", "tests/test_extra.py"]
    assert split_export_paths(paths, {"src/package.py"}, allow_new_repro=True) == (
        ["test_repro.py", "tests/test_extra.py"],
        [],
    )


def test_existing_tests_and_harness_changes_are_still_rejected():
    paths = ["tests/test_existing.py", "tests/conftest.py", "test_tmp/sitecustomize.py", "pyproject.toml"]
    assert split_export_paths(paths, {"tests/test_existing.py"}, allow_new_repro=True) == ([], paths)


def test_legacy_profile_still_rejects_new_reproduction_files():
    assert split_export_paths(["test_repro.py"], set()) == ([], ["test_repro.py"])
