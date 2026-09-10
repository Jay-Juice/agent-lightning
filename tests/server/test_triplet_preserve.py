# Copyright (c) Microsoft. All rights reserved.

"""Temporal PPO must retain distinct completions from the same prompt."""

import asyncio

from agentlightning.schemas import Event
from agentlightning.server.routes import events


def test_preserve_calls_does_not_change_legacy_deduplication(monkeypatch):
    calls = [
        Event(
            event_type="model_request",
            rollout_id="r",
            attempt_id="a",
            timestamp=i,
            data={
                "response": {
                    "prompt_token_ids": [1, 2],
                    "choices": [{"token_ids": [3 + i]}],
                }
            },
        )
        for i in range(2)
    ]
    monkeypatch.setattr(events, "_query_events", lambda **kwargs: calls)
    preserved = asyncio.run(events.query_events("r", format="triplet-preserve"))
    legacy = asyncio.run(events.query_events("r", format="triplet"))
    assert [e.data["response_token_ids"] for e in preserved] == [[3], [4]]
    assert [e.data["response_token_ids"] for e in legacy] == [[4]]
