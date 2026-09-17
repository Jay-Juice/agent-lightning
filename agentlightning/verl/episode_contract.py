# Copyright (c) Microsoft. All rights reserved.
"""Opt-in SWE episode contract. Infrastructure cutoffs never become PPO targets."""

import math

TASK_REASONS = frozenset({"submitted", "format_errors", "turn_budget", "context_budget"})
VALID_GRADING = frozenset({"completed", "candidate_rejected"})


def outcome(rollout):
    records = [e["data"] for e in rollout.events if e["event_type"] == "rollout_outcome"]
    if len(records) > 1:
        raise ValueError("Multiple episode outcomes; attempts must not be combined")
    return records[0] if records else {}


def retryable_infrastructure(rollout):
    record = outcome(rollout)
    return record.get("terminal_kind") == "infrastructure_truncation" and record.get("retryable") is True


def _has_output(data):
    if data.get("response_token_ids"):
        return True
    response = data.get("response", {})
    if isinstance(response, list):
        return any(_has_output({"response": chunk}) for chunk in response)
    if not isinstance(response, dict):
        return bool(response)
    return any(c.get("token_ids") or c.get("text") or (c.get("message") or {}).get("content")
               or (c.get("message") or {}).get("tool_calls") or c.get("delta")
               for c in response.get("choices", []))


def validate_completed(rollout, expected_version=None):
    """Return a training view; original error events stay in the saved trace.

    Only explicitly identified HTTP rejections with no output can be excluded.
    The trusted agent must acknowledge exactly the successful logical calls.
    Repeated prompts are never deduplicated.
    """
    record = outcome(rollout)
    if record.get("protocol_version") != "swe-v2":
        raise ValueError("Missing SWE v2 episode contract")
    if record.get("terminal_kind") != "task_terminal" or record.get("reason") not in TASK_REASONS:
        raise ValueError(f"Not a task terminal: {record.get('reason')}")
    if record.get("grading_status") not in VALID_GRADING:
        raise ValueError(f"Unresolved grading status: {record.get('grading_status')}")
    if rollout.rollout_state != "succeeded" or rollout.error_message:
        raise ValueError("Episode process did not finish successfully")
    if rollout.final_reward is None or not math.isfinite(rollout.final_reward) or rollout.final_reward not in (0, 1):
        raise ValueError("SWE requires a finite binary final reward")
    rewards = [e for e in rollout.events if e["event_type"] == "reward"]
    if len(rewards) != 1 or rewards[0]["data"].get("value") != rollout.final_reward:
        raise ValueError("Require exactly one matching reward event")
    if rewards[0]["data"].get("reason") != record["reason"]:
        raise ValueError("Reward and episode outcome disagree")
    attempts = {e.get("attempt_id") for e in rollout.events if e.get("attempt_id")}
    if len(attempts) > 1:
        raise ValueError("Events from multiple attempts cannot form one episode")
    accepted = record.get("accepted_call_ids", [])
    rejected = record.get("rejected_call_ids", [])
    if any(not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)
           for ids in (accepted, rejected)):
        raise ValueError("Logical call IDs must be lists of nonempty strings")
    if (not accepted or len(set(accepted)) != len(accepted) or len(set(rejected)) != len(rejected)
            or set(accepted) & set(rejected)):
        raise ValueError("Invalid accepted/rejected logical call IDs")
    success_ids, ignored_ids = [], set()
    for event in rollout.triplet_events:
        if event["event_type"] != "model_request":
            continue
        data = event["data"]
        call_id = data.get("logical_call_id")
        status = data.get("http_status")
        if not call_id:
            raise ValueError("A model request is missing its logical call ID")
        if isinstance(status, int) and status >= 400:
            if _has_output(data) or call_id not in set(accepted) | set(rejected):
                raise ValueError("Unacknowledged or partially generated failed request")
            ignored_ids.add((call_id, status))
            continue
        if (not isinstance(status, int) or not 200 <= status < 300 or data.get("status") == "error"
                or not data.get("response_token_ids") or not data.get("prompt_token_ids")):
            raise ValueError("Missing real action tokens")
        if expected_version is not None and data.get("server", {}).get("version") != expected_version:
            raise ValueError("Episode contains a different policy version")
        success_ids.append(call_id)
    if success_ids != accepted or len(success_ids) != len(rollout.triplets or []):
        raise ValueError("Agent acknowledgements and real actions disagree")
    if not set(rejected) <= {call_id for call_id, _ in ignored_ids}:
        raise ValueError("An acknowledged rejection has no corresponding HTTP event")
    raw_requests = [e for e in rollout.events if e["event_type"] == "model_request"]
    trimmed_requests = [e for e in rollout.triplet_events if e["event_type"] == "model_request"]
    def identity(event):
        return event["data"].get("logical_call_id"), event["data"].get("http_status")

    if [identity(e) for e in raw_requests] != [identity(e) for e in trimmed_requests]:
        raise ValueError("Raw and training request records disagree")

    def keep(event):
        data = event.get("data", {})
        if event["event_type"] == "model_request" and identity(event) in ignored_ids:
            if _has_output(data):
                raise ValueError("Raw failed request contains unaccounted generated output")
            return False
        return True

    return rollout.model_copy(update={
        "events": [e for e in rollout.events if keep(e)],
        "triplet_events": [e for e in rollout.triplet_events if keep(e)],
    })
