# Copyright (c) Microsoft. All rights reserved.
"""Run the Docker adapter with fake I/O and compare termination with upstream."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "responses,limit,expected_calls,reason,rejected_source",
    [
        (["invalid"] * 6, 3, 3, "format_errors", None),
        (
            [
                "invalid",
                "invalid",
                "```bash\nfalse\n```",
                "invalid",
                "invalid",
                "invalid",
            ],
            3,
            6,
            "format_errors",
            None,
        ),
        (["invalid"] * 4, 0, 4, "turn_budget", None),
        (["invalid"] * 4, 1, 1, "format_errors", None),
        (["server-overflow"], 3, 1, "context_budget", None),
        (
            ["invalid"] * 3,
            3,
            3,
            "format_errors",
            "def test_value():\n    assert False\n",
        ),
        (["invalid"] * 3, 3, 3, "format_errors", "def test_value(:"),
    ],
)
def test_consecutive_format_error_termination_matches_upstream(
    monkeypatch, tmp_path, responses, limit, expected_calls, reason, rejected_source
):
    pytest.importorskip("docker")
    pytest.importorskip("openai")
    transformers = pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples" / "multiturn_ppo"))
    import smith_docker_agent as pilot
    from full_python_agent import FullPythonSandbox

    smith = pilot.load_smith()
    original_responses = iter(responses)

    def original_query(*args):
        content = next(original_responses)
        if content == "server-overflow":
            raise smith._ContextOverflow("maximum context length")
        return content, "length", 1

    monkeypatch.setattr(smith, "_query", original_query)
    monkeypatch.setattr(smith, "_run", lambda *args: ("command failed", 1))
    _, upstream_calls, _ = smith.run_agent_loop(
        None,
        "problem",
        max_turns=len(responses),
        cmd_timeout=1,
        obs_cap=100,
        max_tokens=100,
        max_format_errors=limit,
    )
    monkeypatch.setattr(pilot, "load_smith", lambda: smith)
    monkeypatch.setattr(pilot, "agent_task", lambda row: row)
    monkeypatch.setattr(pilot, "test_nodes", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot.docker, "from_env", lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1]),
    )

    class Box:
        def __init__(self, *args):
            self.restored_source_paths = ["inflect/__init__.py"]

        def git(self, *args):
            return "def test_value():\n    assert True\n"

        def root(self, *args):
            return rejected_source

        def prepare(self):
            return {}

        def execute(self, action):
            return 1, "command failed"

        def export_patch(self, **kwargs):
            if rejected_source is not None:
                return FullPythonSandbox.export_patch(self, **kwargs)
            self.excluded_patch_paths = []
            return "", [], None

        def close(self):
            pass

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            self.responses = iter(responses)
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            content = next(self.responses)
            if content == "server-overflow":
                exc = RuntimeError("maximum context length")
                exc.status_code = 400
                raise exc
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="length")]
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    client = Client()
    monkeypatch.setattr(pilot, "SmithSandbox", Box)
    monkeypatch.setattr(pilot, "OpenAI", lambda **kwargs: client)
    graded = []
    monkeypatch.setattr(pilot, "grade", lambda *args: graded.append(args) or {"reward": 0.0})
    events = []
    monkeypatch.setattr(
        pilot.httpx,
        "post",
        lambda *args, **kwargs: events.append(kwargs["json"]) or SimpleNamespace(raise_for_status=lambda: None),
    )
    env = {
        "AGL_TASK": json.dumps({"problem_statement": "problem"}),
        "AGL_KEY": "test-only",
        "AGL_EVENT_URL": "http://test/rollouts/test-id/events",
        "AGL_RUN_DIR": str(tmp_path),
        "AGL_TRAIN_MODEL": "fake",
        "AGL_OPENAI_BASE_URL": "http://test",
        "SMITH_MAX_TURNS": str(len(responses)),
        "SMITH_MAX_TOKENS": "100",
        "SMITH_CONTEXT": "1000",
        "SMITH_MAX_FORMAT_ERRORS": str(limit),
        "SMITH_VERIFY_SUBMISSION": "0",
        "SMITH_CHECKED_EDITOR": "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "agent").mkdir()
    pilot.SmithDockerAgent().run()
    records = [json.loads(line) for line in (tmp_path / "agent/test-id/trajectory.jsonl").read_text().splitlines()]
    assert client.calls == expected_calls
    assert upstream_calls == expected_calls - int(reason == "context_budget")
    if reason == "context_budget":
        assert records == [{"stop_reason": "server_context_limit", "prompt_tokens": 1}]
    else:
        assert len(records) == expected_calls and records[-1]["response"] == responses[expected_calls - 1]
    assert len(graded) == int(rejected_source is None) and events[0]["data"]["reason"] == reason
    if rejected_source is not None:
        report = json.loads((tmp_path / "agent/test-id/grade.json").read_text())
        assert report["reward"] == 0.0 and not report["resolved"]
        assert report["patch_rejection"] in {
            "embedded_test_change",
            "invalid_embedded_test_source",
        }
        assert events[0]["data"]["value"] == 0.0
        assert json.loads((tmp_path / "agent/test-id/sandbox.json").read_text())["excluded_reproduction_paths"] == []
    if reason == "format_errors":
        assert records[-1]["stop_reason"] == reason
        assert "observation" not in records[-1]


def test_gateway_pause_retries_same_request_and_has_bounded_wait(monkeypatch):
    pytest.importorskip("docker")
    pytest.importorskip("openai")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples" / "multiturn_ppo"))
    import smith_docker_agent as pilot

    smith = pilot.load_smith()
    clock = [0.0]
    monkeypatch.setattr(pilot.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pilot.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    paused = RuntimeError("gateway paused")
    paused.status_code = 429
    responses = iter([paused, "completion"])
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert pilot.query_completion(llm, smith, messages=[{"role": "user", "content": "x"}], seed=42) == "completion"
    assert len(calls) == 2 and calls[0] == calls[1] and clock[0] == 5
    responses = iter([paused] * 4)
    with pytest.raises(TimeoutError, match="wait budget"):
        pilot.query_completion(llm, smith, gateway_wait_s=10)
    assert clock[0] == 15
    responses = iter([RuntimeError("transport failure")])
    with pytest.raises(RuntimeError, match="transport failure"):
        pilot.query_completion(llm, smith)
