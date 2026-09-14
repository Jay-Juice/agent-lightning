# Copyright (c) Microsoft. All rights reserved.
"""Exercise the privileged-state collector in one disposable SWE container."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import docker
from full_python_agent import FullPythonAgent, FullPythonSandbox
from sandbox import load_smith
from smith_docker_agent import _critic_messages, agent_task
from smith_submission import EDITOR_HINT, WORKFLOW_HINT
from transformers import AutoTokenizer

from agentlightning.privileged_state import SandboxStateSnapshotter, budgeted_state_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory for each snapshot audit")
    args.output.mkdir(parents=True)
    row = json.loads(args.data.read_text().splitlines()[0])
    client = docker.from_env(timeout=120)
    box = FullPythonSandbox(client, agent_task(row), "p1-snapshot-audit-" + uuid.uuid4().hex[:12])
    try:
        preparation = box.prepare()
        snapshotter = SandboxStateSnapshotter(box.container, args.output / "state")
        initial = snapshotter.initialize()
        empty = snapshotter.capture()
        return_code, _ = box.execute("printf 'p1 snapshot audit\\n' > agl_pi_snapshot_audit.txt")
        if return_code:
            raise RuntimeError("Could not create the snapshot audit file")
        changed = snapshotter.capture()
        return_code, _ = box.execute("rm agl_pi_snapshot_audit.txt")
        if return_code:
            raise RuntimeError("Could not restore the snapshot audit state")
        restored = snapshotter.capture()
        changed_paths = [item["path"] for item in changed["delta"]["filesystem"]]
        if empty["delta"]["filesystem"] or changed_paths != ["agl_pi_snapshot_audit.txt"]:
            raise RuntimeError(f"Unexpected filesystem delta: empty={empty['delta']}; changed={changed_paths}")
        if restored["delta"]["filesystem"]:
            raise RuntimeError("Restored file remained in the cumulative delta")
        box.container.exec_run(
            ["/opt/miniconda3/envs/testbed/bin/python", "-m", "http.server", "8765"],
            user="65534:65534",
            workdir="/testbed",
            detach=True,
        )
        return_code, _ = box.execute(
            "python -c \"import socket,time; time.sleep(.2); "
            "socket.create_connection(('127.0.0.1',8765),timeout=2).close()\""
        )
        if return_code:
            raise RuntimeError("Could not start the runtime-state audit server")
        runtime = snapshotter.capture()
        added_processes = runtime["delta"]["processes"]["added"]
        added_sockets = runtime["delta"]["sockets"]["added"]
        if not any("http.server 8765" in str(process["command"]) for process in added_processes):
            raise RuntimeError(f"Background process missing from runtime delta: {added_processes}")
        if not any(socket["local"].endswith(":8765") for socket in added_sockets):
            raise RuntimeError(f"Listening socket missing from runtime delta: {added_sockets}")
        smith = load_smith()
        messages = [
            {"role": "system", "content": smith.SYSTEM_PROMPT},
            {"role": "user", "content": smith.INSTANCE_PROMPT.format(problem_statement=row["problem_statement"])},
        ]
        messages[-1]["content"] += WORKFLOW_HINT + EDITOR_HINT + FullPythonAgent.extra_workflow_hint
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        actor_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )

        def critic_ids(text: str) -> list[int]:
            return tokenizer.apply_chat_template(
                _critic_messages(messages, text),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        prompt_ceiling = 65536 - 4096 - 32
        privileged_text, serialization = budgeted_state_text(
            changed["delta"],
            token_length=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            max_tokens=4096,
            fits=lambda text: len(critic_ids(text)) <= prompt_ceiling,
        )
        rendered_critic_ids = critic_ids(privileged_text)
        if len(rendered_critic_ids) > prompt_ceiling or not privileged_text:
            raise RuntimeError("Real-tokenizer PI serialization did not fit the fixed response reserve")
        report = {
            "instance_id": row["instance_id"],
            "preparation": preparation,
            "initial_file_count": len(initial["filesystem"]),
            "initial_snapshot_seconds": initial["snapshot_duration_s"],
            "empty_snapshot_seconds": empty["snapshot_duration_s"],
            "changed_snapshot_seconds": changed["snapshot_duration_s"],
            "restored_snapshot_seconds": restored["snapshot_duration_s"],
            "runtime_snapshot_seconds": runtime["snapshot_duration_s"],
            "changed_delta": changed["delta"],
            "runtime_delta": {
                "processes": runtime["delta"]["processes"],
                "sockets": runtime["delta"]["sockets"],
            },
            "prompt_audit": {
                "actor_tokens": len(actor_ids),
                "critic_tokens": len(rendered_critic_ids),
                "pi_tokens": serialization["serialized_tokens"],
                "pi_truncated": serialization["truncated"],
                "prompt_ceiling": prompt_ceiling,
            },
            "restored_state_hash": restored["state_hash"],
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        box.close()
        client.close()


if __name__ == "__main__":
    main()
