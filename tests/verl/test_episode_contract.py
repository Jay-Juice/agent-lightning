# Copyright (c) Microsoft. All rights reserved.
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agentlightning.verl.episode_contract import retryable_infrastructure, validate_completed


class Rollout(SimpleNamespace):
    def model_copy(self, update):
        return Rollout(**{**vars(self), **update})


def episode():
    calls = [{"event_type": "model_request", "attempt_id": "one", "data": {
        "logical_call_id": call, "http_status": 200, "prompt_token_ids": [1],
        "response_token_ids": [2], "server": {"version": 7},
    }} for call in ("a", "b")]
    reward = {"event_type": "reward", "attempt_id": "one", "data": {"value": 1, "reason": "submitted"}}
    outcome = {"event_type": "rollout_outcome", "attempt_id": "one", "data": {
        "protocol_version": "swe-v2", "terminal_kind": "task_terminal", "reason": "submitted",
        "grading_status": "completed", "accepted_call_ids": ["a", "b"], "rejected_call_ids": [],
    }}
    return Rollout(events=deepcopy([*calls, reward, outcome]), triplet_events=deepcopy(calls),
                   triplets=[object(), object()], rollout_state="succeeded", error_message=None, final_reward=1)


def test_identical_prompts_keep_both_real_actions():
    value = episode()
    assert len(validate_completed(value, 7).triplet_events) == 2


def test_only_acknowledged_no_output_http_rejections_are_filtered():
    value = episode()
    failed = {"event_type": "model_request", "attempt_id": "one", "data": {
        "logical_call_id": "c", "http_status": 400, "status": "error", "response_token_ids": [],
    }}
    value.events.insert(2, deepcopy(failed))
    value.triplet_events.append(deepcopy(failed))
    value.events[-1]["data"]["rejected_call_ids"] = ["c"]
    assert len(validate_completed(value, 7).triplet_events) == 2
    assert len(value.triplet_events) == 3  # original evidence is retained
    value.events[2]["data"]["response"] = {"choices": [{"message": {"content": "lost action"}}]}
    with pytest.raises(ValueError, match="output"):
        validate_completed(value, 7)


@pytest.mark.parametrize("mutation", ["unknown", "grading", "version", "attempt", "missing", "raw", "rejected"])
def test_fail_closed(mutation):
    value = episode()
    if mutation == "unknown":
        value.events[-1]["data"]["reason"] = "arbitrary"
    elif mutation == "grading":
        value.events[-1]["data"]["grading_status"] = "requires_review"
    elif mutation == "version":
        value.triplet_events[0]["data"]["server"]["version"] = 6
    elif mutation == "attempt":
        value.events[0]["attempt_id"] = "two"
    elif mutation == "missing":
        value.triplets.pop()
    elif mutation == "raw":
        value.events.pop(0)
    else:
        value.events[-1]["data"]["rejected_call_ids"] = ["unseen"]
    with pytest.raises(ValueError):
        validate_completed(value, 7)


def test_reward_zero_never_means_retry():
    value = episode()
    value.final_reward = 0
    assert not retryable_infrastructure(value)
    value.events[-1]["data"].update(terminal_kind="infrastructure_truncation", retryable=True)
    assert retryable_infrastructure(value)


def test_adjudicated_candidate_failure_requires_zero_reward():
    value = episode()
    value.events[-1]["data"]["grading_status"] = "candidate_failed"
    with pytest.raises(ValueError, match="zero reward"):
        validate_completed(value, 7)
    value.final_reward = 0
    value.events[-2]["data"]["value"] = 0
    assert validate_completed(value, 7).final_reward == 0
