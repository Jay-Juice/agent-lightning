"""Exercise PI context fallback through the real SmithDockerAgent.run loop."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("prompt_tokens", range(61409, 61442))
def test_no_pi_margin_preserves_actor_context_domain(monkeypatch, tmp_path, prompt_tokens):
    pytest.importorskip("docker")
    pytest.importorskip("openai")
    transformers = pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples" / "multiturn_ppo"))
    import smith_docker_agent as pilot

    class FormatError(Exception):
        pass

    def parse_action(content):
        raise FormatError("controlled response ends after one call")

    smith = SimpleNamespace(
        SYSTEM_PROMPT="system", INSTANCE_PROMPT="Issue: {problem_statement}",
        FormatError=FormatError, parse_action=parse_action,
        _ContextOverflow=type("ContextOverflow", (Exception,), {}),
        _is_context_overflow=lambda exc: False,
    )
    monkeypatch.setattr(pilot, "load_smith", lambda: smith)
    monkeypatch.setattr(pilot, "agent_task", lambda row: row)
    monkeypatch.setattr(pilot, "test_nodes", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot.docker, "from_env", lambda **kwargs: SimpleNamespace(close=lambda: None))
    tokenized_messages = []

    def tokenize(messages, **kwargs):
        tokenized_messages.append(copy.deepcopy(messages))
        return [1] * prompt_tokens

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(apply_chat_template=tokenize),
    )

    class Box:
        def __init__(self, *args):
            self.container = object()
            self.excluded_patch_paths = []

        def prepare(self):
            return {"baseline_head": "a" * 40}

        def export_patch(self, **kwargs):
            return "", [], None

        def close(self):
            pass

    class Store:
        def __init__(self, container, exact_commit):
            self.exact_commit = exact_commit
            self.manifest_hash = "frozen-manifest"

        def initialize(self):
            return {"commit": self.exact_commit, "manifest_hash": self.manifest_hash}

        def write_manifest(self, path):
            pass

    captures = []

    class Snapshotter:
        def __init__(self, *args):
            pass

        def initialize(self):
            return {"schema_version": 1, "snapshot_duration_s": 0.0}

        def capture(self):
            captures.append(True)
            return {
                "delta": {"schema_version": 1, "filesystem": []},
                "state_hash": "same-pre-action-state", "state_ref": "fake-state.json.gz",
                "captured_at_start": 0.0, "captured_at_end": 0.0, "snapshot_duration_s": 0.0,
            }

    monkeypatch.setattr(pilot, "SmithSandbox", Box)
    monkeypatch.setattr(pilot, "PreparedBaselineBlobStore", Store)
    monkeypatch.setattr(pilot, "SandboxStateSnapshotter", Snapshotter)
    monkeypatch.setattr(pilot, "grade", lambda *args: {"reward": 0.0})
    requests, events = [], []

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="invalid"), finish_reason="stop")]
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(pilot, "OpenAI", Client)
    monkeypatch.setattr(
        pilot.httpx, "post",
        lambda *args, **kwargs: events.append(copy.deepcopy(kwargs["json"]))
        or SimpleNamespace(raise_for_status=lambda: None),
    )
    env = {
        "AGL_TASK": json.dumps({"problem_statement": "problem", "_agl_sampling_seed": 42}),
        "AGL_KEY": "test-only", "AGL_RUN_DIR": str(tmp_path), "AGL_TRAIN_MODEL": "fake",
        "AGL_OPENAI_BASE_URL": "http://test/mode/train/",
        "SMITH_MAX_TURNS": "1", "SMITH_MAX_TOKENS": "4096", "SMITH_CONTEXT": "65536",
        "SMITH_MAX_FORMAT_ERRORS": "1", "SMITH_AGENT_WALL_TIMEOUT": "0",
        "SMITH_VERIFY_SUBMISSION": "0", "SMITH_CHECKED_EDITOR": "0",
        "SMITH_ALLOW_REPRO_FILES": "0", "SMITH_CHECK_SYNTAX": "0",
        "SMITH_PRIVILEGED_ENCODING": "semantic_hunks_v2",
        "SMITH_PRIVILEGED_MAX_TOKENS": "4096", "SMITH_PRIVILEGED_SAFETY_MARGIN": "32",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "agent").mkdir()
    results = {}
    for mode in ("off", "critic"):
        monkeypatch.setenv("SMITH_PRIVILEGED_STATE", mode)
        monkeypatch.setenv("AGL_EVENT_URL", f"http://test/rollouts/{mode}/events")
        requests.clear()
        events.clear()
        tokenized_messages.clear()
        pilot.SmithDockerAgent().run()
        records = [
            json.loads(line)
            for line in (tmp_path / "agent" / mode / "trajectory.jsonl").read_text().splitlines()
        ]
        results[mode] = copy.deepcopy((requests, events, records, tokenized_messages))

    off_requests, _, off_records, _ = results["off"]
    pi_requests, pi_events, pi_records, pi_tokenizations = results["critic"]
    if prompt_tokens > 61440:
        assert off_requests == pi_requests == []
        assert captures == []
        assert off_records == pi_records == [{"stop_reason": "context_limit", "prompt_tokens": prompt_tokens}]
        assert pi_events == [{"event_type": "reward", "data": {
            "value": 0.0, "reason": "context_budget", "source": "isolated_pytest",
        }}]
    else:
        assert len(off_requests) == len(pi_requests) == 1
        assert off_records[-1]["stop_reason"] == pi_records[-1]["stop_reason"] == "format_errors"
        assert len(captures) == 1
        pi_request = pi_requests[0]
        assert "X-AgentLightning-Logical-Call-Id" in pi_request.pop("extra_headers")
        assert pi_request == off_requests[0]
        pi_event = [event["data"] for event in pi_events if event["event_type"] == "privileged_state"]
        assert len(pi_event) == 1
        event = pi_event[0]
        assert event["text"] == "" and event["fallback_no_pi"] is True
        assert event["serialized_tokens"] == 0
        assert event["actor_prompt_tokens"] == event["critic_prompt_tokens"] == prompt_tokens
        assert event["actor_prompt_hash"] == event["critic_prompt_hash"]
        assert pi_tokenizations == [off_requests[0]["messages"], off_requests[0]["messages"]]
