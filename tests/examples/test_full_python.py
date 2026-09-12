# Copyright (c) Microsoft. All rights reserved.
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
    assert len(result["PASS_TO_PASS"]) == 200


def test_native_unittest_ids_use_files_before_p2p_filtering():
    row = {
        "instance_id": "tornado.task",
        "image_name": "repo/image",
        "FAIL_TO_PASS": ["test_fix (tornado.test.web_test.Case)"],
        "PASS_TO_PASS": ["test_keep (tornado.test.web_test.Case)", "A descriptive test.", "unrelated.py::test_other"],
    }
    result = prepare_row(
        row,
        {"repo/image": "sha256:" + "a" * 64},
        {
            "A descriptive test.": ["tornado/test/web_test.py::Case::test_doc"],
        },
    )
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
    import full_python_agent

    return full_python_agent


def test_forced_color_keeps_exact_status_and_node_id(grader):
    output = (
        "\x1b[32mPASSED\x1b[0m tests/a.py::\x1b[1mCase::test_fix\x1b[0m\n"
        "tests/a.py::test_keep \x1b[31mFAILED\x1b[0m [100%]\n"
    )
    assert grader.parse_statuses(output) == {
        "tests/a.py::Case::test_fix": "PASSED",
        "tests/a.py::test_keep": "FAILED",
    }


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
