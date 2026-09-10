# Copyright (c) Microsoft. All rights reserved.

"""Two model turns with a visible tool observation; terminal exact-match reward."""

import json
import os
import re
from pathlib import Path

import httpx
from openai import OpenAI


class SmokeAgent:
    def run(self):
        task = json.loads(os.environ["AGL_TASK"])
        key = os.environ["AGL_KEY"]
        messages = [
            {
                "role": "user",
                "content": task["question"]
                + "\nPropose an integer answer; a format checker will then let you revise it.",
            }
        ]
        calls = []
        with OpenAI(base_url=os.environ["AGL_OPENAI_BASE_URL"], api_key=key, timeout=180) as client:
            for turn in range(2):
                response = client.chat.completions.create(
                    model="auto",
                    messages=messages,
                    max_tokens=128,
                    temperature=1.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content or ""
                calls.append(content)
                messages.append({"role": "assistant", "content": content})
                if turn == 0:
                    valid = bool(re.fullmatch(r"\s*[-+]?\d+\s*", content))
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Format checker: integer-only format = {valid}. "
                            "Now return your final integer answer without explanation.",
                        }
                    )
        numbers = re.findall(r"[-+]?\d[\d,]*", calls[-1])
        predicted = int(numbers[-1].replace(",", "")) if numbers else None
        reward = float(predicted == task["answer"])
        event_url = os.environ["AGL_EVENT_URL"]
        httpx.post(
            event_url,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "event_type": "reward",
                "data": {"value": reward, "reason": "two_turn_budget"},
            },
            timeout=20,
        ).raise_for_status()
        rid = event_url.split("/rollouts/")[1].split("/")[0]
        record = {
            "task": task,
            "responses": calls,
            "prediction": predicted,
            "reward": reward,
            "rollout_id": rid,
            "turns": 2,
        }
        (Path(os.environ["AGL_RUN_DIR"]) / "agent" / f"{rid}.json").write_text(json.dumps(record, indent=2))
