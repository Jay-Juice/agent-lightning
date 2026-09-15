"""Real Docker smoke for the v2 semantic-hunk representation."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import docker
from transformers import AutoTokenizer

from agentlightning.privileged_state import (
    PreparedBaselineBlobStore,
    SandboxStateSnapshotter,
    budgeted_semantic_state_text,
    build_semantic_delta,
)
from full_python_agent import FullPythonSandbox
from smith_docker_agent import agent_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    row = json.loads(args.data.read_text().splitlines()[0])
    client = docker.from_env(timeout=120)
    box = FullPythonSandbox(client, agent_task(row), "p2-audit-" + uuid.uuid4().hex[:12])
    try:
        preparation = box.prepare()
        store = PreparedBaselineBlobStore(box.container, exact_commit=preparation["baseline_head"])
        store.initialize()
        snapshotter = SandboxStateSnapshotter(box.container, Path("/tmp") / ("p2-state-" + uuid.uuid4().hex[:8]))
        snapshotter.initialize()
        rc, _ = box.execute("printf 'PI-V2\n' >> setup.cfg")
        if rc:
            raise RuntimeError("controlled edit failed")
        state = snapshotter.capture()
        semantic = build_semantic_delta(state["delta"], store)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        text, stats = budgeted_semantic_state_text(
            semantic,
            token_length=lambda value: len(tokenizer.encode(value, add_special_tokens=False)),
            max_tokens=4096,
        )
        print(json.dumps({"instance_id": row["instance_id"], "text": text, "stats": stats}, indent=2))
    finally:
        box.close()
        client.close()


if __name__ == "__main__":
    main()
