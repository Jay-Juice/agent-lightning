# Copyright (c) Microsoft. All rights reserved.

"""Compare real backend output with the same call through isolated production Gateway routes."""

import argparse
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from agentlightning.server.app import create_app


def signature(response):
    choice = response["choices"][0]
    return {
        "prompt_token_ids": response["prompt_token_ids"],
        "token_ids": choice["token_ids"],
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--messages-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    # TestClient uses the real production routes and event code in this separate
    # CPU process. It never creates tasks in, or alters, the running experiment.
    key = "isolated-transport-audit"
    config = {
        "key": key,
        "default_proxy": {
            "model_name": args.model,
            "train": {"temperature": 0},
            "val": {"temperature": 0},
        },
    }
    files = sorted((args.messages_run / "agent").glob("*/initial-messages.json"))[:3]
    assert len(files) == 3
    results = []
    with TestClient(create_app(config), headers={"Authorization": f"Bearer {key}"}) as gateway:
        response = gateway.post("/api/models", json=[{"model": args.model, "endpoint": args.endpoint, "version": 0}])
        response.raise_for_status()
        for index, file in enumerate(files):
            messages = json.loads(file.read_text())
            body = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 128,
                "seed": 123,
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            direct = httpx.post(f"{args.endpoint}/chat/completions", json=body, timeout=240)
            direct.raise_for_status()
            repeated = httpx.post(f"{args.endpoint}/chat/completions", json=body, timeout=240)
            repeated.raise_for_status()
            rid = f"transport-audit-{index}"
            response = gateway.post("/api/rollouts", json=[{"rollout_id": rid, "input": {}, "is_train": False}])
            response.raise_for_status()
            response = gateway.post(f"/proxy/rollout/{rid}/attempt/0/mode/val/openai/v1/chat/completions", json=body)
            response.raise_for_status()
            events = gateway.get(f"/api/rollouts/{rid}/events").json()
            assert len(events) == 1 and events[0]["event_type"] == "model_request"
            result = {
                "messages_file": str(file),
                "request": body,
                "direct": direct.json(),
                "repeated": repeated.json(),
                "gateway": response.json(),
                "repeat_matches": signature(direct.json()) == signature(repeated.json()),
                "gateway_matches": signature(direct.json()) == signature(response.json()),
                "event_matches": signature(response.json()) == signature(events[0]["data"]["response"]),
            }
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k.endswith("matches")}), flush=True)
    args.output.write_text(json.dumps({"scope": "Live greedy transport probe, not SWE scoring", "probes": results}))
    assert all(r["repeat_matches"] and r["gateway_matches"] and r["event_matches"] for r in results)


if __name__ == "__main__":
    main()
