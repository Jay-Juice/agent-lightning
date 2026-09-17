# Copyright (c) Microsoft. All rights reserved.
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from agentlightning.server import proxy
from agentlightning.server.routes.events import _trim_model_request


@pytest.mark.parametrize("logical_id,expected_calls", [("logical-a", 1), (None, 2)])
def test_reliable_call_is_not_silently_resampled_inside_gateway(monkeypatch, logical_id, expected_calls):
    calls, events = [], []

    async def post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return httpx.Response(500, json={"error": "failed"})
        return httpx.Response(200, json={"choices": [{"token_ids": [1], "finish_reason": "stop"}]})

    async def no_wait(**kwargs):
        pass

    monkeypatch.setattr(proxy, "_sleep_before_retry", no_wait)
    monkeypatch.setattr(proxy, "_capture_event", lambda **kwargs: events.append(kwargs))
    server = SimpleNamespace(endpoint="http://test", model="model")
    result = asyncio.run(proxy.forward_request(
        client=SimpleNamespace(post=post), server=server, body={},
        rollout_id="r", attempt_id="a", logical_call_id=logical_id,
    ))
    assert len(calls) == expected_calls
    assert result.status_code == (500 if logical_id else 200)
    assert events[0]["logical_call_id"] == logical_id


def test_trim_preserves_call_identity_finish_and_policy_version():
    value = _trim_model_request({
        "logical_call_id": "call", "finish_reason": "length", "http_status": 200,
        "server": {"version": 4},
        "response": {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]},
    })
    assert value["logical_call_id"] == "call" and value["finish_reason"] == "length"
    assert value["server"]["version"] == 4
