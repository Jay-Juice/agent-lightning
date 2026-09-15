# Copyright (c) Microsoft. All rights reserved.

import importlib
from pathlib import Path

import pytest

from examples.multiturn_ppo.prepare_full_python import apply_exclusions, prepare_row


def test_preserves_all_f2p_and_same_file_p2p_without_mutating_source():
    row = {
        "instance_id": "repo.task",
        "image_name": "repo/image",
        "FAIL_TO_PASS": ["tests/a.py::test_fix"],
        "PASS_TO_PASS": [
            "tests/a.py::test_keep",
            "tests/ab.py::test_other",
            "tests/a.py.extra::test_other",
        ],
    }
    result = prepare_row(row, {"repo/image": "sha256:" + "a" * 64})
    assert result is not None
    assert result["FAIL_TO_PASS"] == row["FAIL_TO_PASS"]
    assert result["PASS_TO_PASS"] == ["tests/a.py::test_keep"]
    assert len(row["PASS_TO_PASS"]) == 3
    assert result["source_p2p_count"] == 3
    assert result["image"].endswith("@sha256:" + "a" * 64)


def test_non_python_excluded_explicitly():
    assert prepare_row({"FAIL_TO_PASS": ["TestFix"], "PASS_TO_PASS": ["TestOther"]}, {}) is None


def test_exclusions_preserve_validation_order_and_input():
    rows = [{"instance_id": iid, "source_row_sha256": iid * 64} for iid in ("a", "b", "c")]
    splits = {"train": rows[:2], "val": rows[2:]}
    record = {
        "instance_id": "a",
        "split": "train",
        "source_row_sha256": "a" * 64,
        "reason": "offline_service",
        "evidence": "reference-control.json",
    }
    result = apply_exclusions(splits, [record])
    assert result["train"] == [rows[1]] and result["val"] == [rows[2]]
    assert splits["train"] == rows[:2]
    for invalid in (
        [record, record],
        [{**record, "instance_id": "c"}],
        [{**record, "source_row_sha256": "changed"}],
        [{**record, "split": "val"}],
    ):
        with pytest.raises(ValueError):
            apply_exclusions(splits, invalid)


def test_unexpected_large_suite_is_not_silently_truncated():
    row = {
        "instance_id": "repo.task",
        "image_name": "repo/image",
        "FAIL_TO_PASS": ["tests/a.py::test_fix"],
        "PASS_TO_PASS": [f"tests/a.py::test_{i}" for i in range(200)],
    }
    result = prepare_row(row, {"repo/image": "sha256:" + "a" * 64})
    assert result is not None
    assert len(result["PASS_TO_PASS"]) == 200


def test_native_unittest_ids_use_files_before_p2p_filtering():
    row = {
        "instance_id": "tornado.task",
        "image_name": "repo/image",
        "FAIL_TO_PASS": ["test_fix (tornado.test.web_test.Case)"],
        "PASS_TO_PASS": [
            "test_keep (tornado.test.web_test.Case)",
            "A descriptive test.",
            "unrelated.py::test_other",
        ],
    }
    result = prepare_row(
        row,
        {"repo/image": "sha256:" + "a" * 64},
        {
            "A descriptive test.": ["tornado/test/web_test.py::Case::test_doc"],
        },
    )
    assert result is not None
    assert result["FAIL_TO_PASS"] == ["tornado/test/web_test.py::Case::test_fix"]
    assert result["PASS_TO_PASS"] == [
        "tornado/test/web_test.py::Case::test_keep",
        "tornado/test/web_test.py::Case::test_doc",
    ]


@pytest.fixture
def grader(monkeypatch):
    pytest.importorskip("docker")
    pytest.importorskip("openai")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples" / "multiturn_ppo"))
    return importlib.import_module("full_python_agent")


def test_forced_color_keeps_exact_status_and_node_id(grader):
    output = (
        "\x1b[32mPASSED\x1b[0m tests/a.py::\x1b[1mCase::test_fix\x1b[0m\n"
        "tests/a.py::test_keep \x1b[31mFAILED\x1b[0m [100%]\n"
    )
    assert grader.parse_statuses(output) == {
        "tests/a.py::Case::test_fix": "PASSED",
        "tests/a.py::test_keep": "FAILED",
    }


def test_paramiko_aliases_preserve_tests_and_input(grader):
    aliases = grader.PARAMIKO_TEST_ALIASES
    row = {
        "instance_id": "paramiko__paramiko.23f92003.example",
        "FAIL_TO_PASS": [next(iter(aliases))],
        "PASS_TO_PASS": [list(aliases)[1], "tests/x.py::test_ok"],
    }
    f2p, p2p = grader.grading_test_nodes(row)
    assert f2p == [aliases[row["FAIL_TO_PASS"][0]]]
    assert p2p == [aliases[row["PASS_TO_PASS"][0]], "tests/x.py::test_ok"]
    assert row["FAIL_TO_PASS"][0] in aliases
    unrelated = {**row, "instance_id": "other.repo"}
    assert grader.grading_test_nodes(unrelated) == (row["FAIL_TO_PASS"], row["PASS_TO_PASS"])


def test_summary_preserves_spaces_and_does_not_accept_partial_ids(grader):
    good = "tests/test_config.py::Case::test_value[not an int]"
    bad = "tests/test_config.py::Case::test_value[quoted spaced-neil]"
    absent = "tests/test_config.py::Case::test_value[not another int]"
    output = f"PASSED {good}\nFAILED {bad} - AssertionError: bad value\nPASSED {absent.split(' ')[0]}\n"
    statuses = grader.parse_statuses(output, [good, bad, absent])
    assert statuses[good] == "PASSED"
    assert statuses[bad] == "FAILED"
    assert absent not in statuses


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize(
    "candidate,rejection",
    [
        ("def test_value():\n    assert False\n", "embedded_test_change"),
        ("def test_value(:", "invalid_embedded_test_source"),
    ],
)
def test_rejected_embedded_source_exports_fresh_metadata(grader, stale, candidate, rejection):
    box = object.__new__(grader.FullPythonSandbox)
    box.restored_source_paths = ["inflect/__init__.py"]
    if stale:
        box.excluded_patch_paths = ["old_reproduction.py"]
    box.git = lambda *args: "def test_value():\n    assert True\n"
    box.root = lambda *args: candidate
    assert box.export_patch(allow_new_repro=True) == (
        "",
        ["inflect/__init__.py"],
        rejection,
    )
    assert box.excluded_patch_paths == []


def test_restore_mutated_tests_preserves_buggy_production_and_decorators(grader):
    buggy = "# keep this\ndef value():\n    return 0\n\n@pytest.mark.slow\ndef test_value():\n    assert value() != 1\n"
    trusted = "def value():\n    return 1\n\n@pytest.mark.fast\ndef test_value():\n    assert value() == 1\n"
    restored, names, has_bug = grader.recover_embedded_tests(buggy, trusted)
    assert restored.startswith("# keep this\ndef value():\n    return 0\n")
    assert "@pytest.mark.fast" in restored and "!= 1" not in restored
    assert names == ["test_value"] and has_bug
    assert grader.embedded_tests(restored) == grader.embedded_tests(trusted)


def test_restore_detects_test_only_tasks_and_refuses_unsupported_changes(grader):
    buggy = "def value():\n    return 1\ndef test_value():\n    assert value() != 1\n"
    trusted = buggy.replace("!=", "==")
    assert grader.recover_embedded_tests(buggy, trusted) == (
        trusted,
        ["test_value"],
        False,
    )
    with pytest.raises(ValueError, match="inventory"):
        grader.recover_embedded_tests(buggy, trusted + "def test_other():\n    pass\n")
    with pytest.raises(ValueError, match="nested test or doctest"):
        grader.recover_embedded_tests(
            'def value():\n    """>>> broken"""\n',
            'def value():\n    """>>> correct"""\n',
        )


def test_only_package_install_metadata_can_be_ignored(grader):
    assert grader.package_metadata_dirs(
        [
            "stackprinter.egg-info/PKG-INFO",
            "stackprinter.egg-info/SOURCES.txt",
            "new.py",
        ]
    ) == ["stackprinter.egg-info"]
    assert (
        grader.package_metadata_dirs(
            [
                "bad.egg-info/PKG-INFO",
                "bad.egg-info/code.py",
                "nested/good.egg-info/PKG-INFO",
            ]
        )
        == []
    )


def test_existing_fixture_predicates_and_checker_assertions_can_be_repaired(grader):
    before = "@pytest.fixture\ndef version(v):\n    return Checker(v, v > (3, 8))\n"
    assert grader.fixture_patch_allowed(before, before.replace(" > ", " >= "))
    before = "class Checker:\n    def check(self, x):\n        if not self.ok:\n            assert errors(x)\n"
    after = before.replace("if not self.ok", "if self.ok").replace("assert errors", "assert not errors")
    assert grader.fixture_patch_allowed(before, after)


@pytest.mark.parametrize(
    "after",
    [
        "def helper(x):\n    return True\n",
        "def helper(x):\n    pytest.skip('skip')\n    assert x > 0\n",
        "import pytest\ndef helper(x):\n    assert x > 0\n",
        "def helper(x=1):\n    assert x > 0\n",
        "def helper(x):\n    assert x > 0\n\ndef pytest_collection_modifyitems(items):\n    items.clear()\n",
    ],
)
def test_fixture_edits_cannot_drop_assertions_add_calls_or_change_harness(grader, after):
    assert not grader.fixture_patch_allowed("def helper(x):\n    assert x > 0\n", after)


def test_hook_and_test_predicates_are_not_allowed(grader):
    for name in ["pytest_collection_modifyitems", "test_case"]:
        before = f"def {name}(x):\n    assert x > 0\n"
        assert not grader.fixture_patch_allowed(before, before.replace(">", "<"))


def test_inline_tests_survive_source_repairs_but_cannot_be_modified(grader):
    before = (
        'def add(x):\n    """>>> add(1)\n    2\n    """\n    return x - 1\n\ndef test_add():\n    assert add(1) == 2\n'
    )
    fixed = before.replace("return x - 1", "return x + 1")
    assert grader.embedded_tests(before) == grader.embedded_tests(fixed)
    assert grader.embedded_tests(before) != grader.embedded_tests(fixed.replace("assert add(1) == 2", "assert True"))
    assert grader.embedded_tests(before) != grader.embedded_tests(fixed.replace("    2\n", "    0\n"))


def test_displaced_doctest_can_return_to_start_of_function(grader):
    before = 'def f():\n    return 1\n    """>>> f()\n    1\n    """\n'
    after = 'def f():\n    """>>> f()\n    1\n    """\n    return 1\n'
    assert grader.embedded_tests(before) == grader.embedded_tests(after)


def test_fixture_paths_can_be_repaired_without_adding_io_calls(grader):
    before = "def url():\n    sys.path.insert(0, os.path.dirname(__file__))\n    yield server.url + '/'\n"
    after = before.replace("dirname(__file__)", "dirname(os.path.dirname(__file__))").replace("+ '/'", "+ '/html'")
    assert grader.fixture_patch_allowed(before, after)
    assert not grader.fixture_patch_allowed(before, after.replace("yield server.url", "yield open('/tmp/x').read()"))
