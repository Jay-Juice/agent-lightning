# Copyright (c) Microsoft. All rights reserved.
"""Opt-in diagnostic: compare synchronized model replicas before whole-episode evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import trace_hooks


class ParityTraces(trace_hooks.SaveTraces):
    def on_startup(self, store=None):
        from agentlightning.verl.agl_rollout_manager import AglRolloutManager

        baseline = Path(os.environ["AGL_PARITY_BASELINE"])
        self.seeds = {}
        for f in (baseline.parent / "jobs").glob("*-original.json"):
            job = json.loads(f.read_text())
            self.seeds[job["row"]["instance_id"]] = job["seed"]
        original = AglRolloutManager.register_model

        def register_and_compare(manager, addresses):
            AglRolloutManager.register_model = original
            models = original(manager, addresses)
            for i, model in enumerate(models):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("audit_swe_backend_parity.py")),
                        "--endpoint",
                        model.endpoint,
                        "--model",
                        model.model,
                        "--baseline",
                        str(baseline),
                        "--output",
                        str(Path(os.environ["AGL_RUN_DIR"]) / f"backend-parity-{i}.json"),
                    ],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    check=True,
                    timeout=600,
                )
            return models

        AglRolloutManager.register_model = register_and_compare

    def on_enqueue(self, request):
        row = request.input
        return request.model_copy(update={"input": dict(row, _agl_sampling_seed=self.seeds[row["instance_id"]])})
