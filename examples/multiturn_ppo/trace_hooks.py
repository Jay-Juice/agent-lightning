# Copyright (c) Microsoft. All rights reserved.

"""Persist complete AGL events before the trainer removes completed rollouts."""

import json
import os
from pathlib import Path

from agentlightning.hooks import RolloutHooks


class SaveTraces(RolloutHooks):
    def on_succeeded(self, rollout, events, store):
        payload = {
            "rollout": rollout.model_dump(mode="json"),
            "events": {k: [e.model_dump(mode="json") for e in v] for k, v in events.items()},
        }
        (Path(os.environ["AGL_RUN_DIR"]) / "traces" / f"{rollout.rollout_id}.json").write_text(json.dumps(payload))

    def on_failed(self, rollout, store):
        (Path(os.environ["AGL_RUN_DIR"]) / "traces" / f"{rollout.rollout_id}.failed.json").write_text(
            rollout.model_dump_json()
        )
