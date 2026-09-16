"""Audit v1/v2 on identical controlled states in prepared SWE Docker tasks.

This runs no model inference or grading and exposes no gold patch to an Actor.
Only the disposable audit containers are modified; S0 blobs stay frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import uuid
from pathlib import Path

from agentlightning.privileged_state import (
    PreparedBaselineBlobStore,
    SandboxStateSnapshotter,
    budgeted_semantic_state_text,
    budgeted_state_text,
    build_semantic_delta,
)


def _write_files(container, files: dict[str, str | None]) -> None:
    """Write exact UTF-8 contents (or unlink audit files) without shell parsing."""
    script = """
import json, pathlib, sys
root = pathlib.Path('/testbed')
payload = pathlib.Path(sys.argv[1])
files = json.loads(payload.read_text())
payload.unlink()
for name, content in files.items():
    path = root / name
    if root not in path.resolve().parents:
        raise ValueError('audit path escapes workspace')
    if content is None:
        path.unlink()
    else:
        path.write_bytes(content.encode('utf-8'))
"""
    # Docker exec argv has a per-argument size limit; stage only the audit payload
    # outside /testbed so large source files do not hit that limit or enter PI.
    payload_name = "pi-audit-" + uuid.uuid4().hex + ".json"
    raw = json.dumps(files).encode()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as handle:
        member = tarfile.TarInfo(payload_name)
        member.size = len(raw)
        member.mode = 0o600
        handle.addfile(member, io.BytesIO(raw))
    if not container.put_archive("/tmp", archive.getvalue()):
        raise RuntimeError("Could not stage controlled audit edit")
    result = container.exec_run(
        ["/opt/miniconda3/envs/testbed/bin/python", "-c", script, "/tmp/" + payload_name],
        user="0:0",
        workdir="/testbed",
    )
    if result.exit_code:
        raise RuntimeError(result.output.decode("utf-8", errors="replace")[-2000:])


def _select_sources(store, token_length, max_tokens, initial_files):
    candidates = sorted(
        (
            (path, row)
            for path, row in store.entries.items()
            if path in initial_files
            and path.endswith(".py")
            and row["mode"] in ("100644", "100755")
            and 0 < row["size"] <= store.max_text_bytes
        ),
        key=lambda item: (-item[1]["size"], item[0]),
    )
    sources = {}
    for path, _ in candidates[:32]:
        record = store.read_text(path)
        if record["kind"] == "text":
            sources[path] = record["text"]
        if len(sources) >= 3:
            break
    if len(sources) < 3:
        raise RuntimeError("Task needs at least three tracked UTF-8 Python files for the audit")
    large = max(sources, key=lambda path: token_length(sources[path]))
    if token_length(sources[large]) <= max_tokens:
        raise RuntimeError("Task has no source file larger than the PI token budget; select another task")
    return large, sources


def _scenario_edits(large, sources, prefix):
    lines = sources[large].splitlines(keepends=True)
    index = len(lines) // 2
    marker = prefix + "_LARGE_LINE"
    lines[index] = "# " + marker + "\n"
    yield "large_file_one_line", {large: "".join(lines)}, [marker]
    edits, markers = {}, []
    for index, (path, source) in enumerate(sources.items()):
        marker = prefix + "_MULTI_" + str(index)
        edits[path] = source + ("" if source.endswith("\n") else "\n") + "# " + marker + "\n"
        markers.append(marker)
    yield "multiple_files", edits, markers
    marker = prefix + "_NEW_HELPER"
    yield "new_helper_file", {prefix.lower() + ".py": "# " + marker + "\nprint('audit helper')\n"}, [marker]


def audit_task(row, client, tokenizer, output_dir, max_tokens):
    # Keep Docker/agent imports out of helper-only unit tests.
    from full_python_agent import FullPythonSandbox
    from smith_docker_agent import agent_task

    def token_length(value):
        return len(tokenizer.encode(value, add_special_tokens=False))

    box = FullPythonSandbox(client, agent_task(row), "p2-audit-" + uuid.uuid4().hex[:12])
    try:
        preparation = box.prepare()
        store = PreparedBaselineBlobStore(box.container, exact_commit=preparation["baseline_head"])
        baseline = store.initialize()
        store.write_manifest(output_dir / "baseline-manifest.json.gz")
        snapshotter = SandboxStateSnapshotter(box.container, output_dir / "snapshots")
        initial = snapshotter.initialize()
        initial_files = {record["path"]: record for record in initial["filesystem"]}
        large, sources = _select_sources(store, token_length, max_tokens, initial_files)
        for path, source in sources.items():
            if initial_files[path]["sha256"] != hashlib.sha256(source.encode()).hexdigest():
                raise AssertionError(f"Frozen prepared S0 blob differs from actual workspace: {path}")
        prefix = "PI_AUDIT_" + uuid.uuid4().hex[:12].upper()
        reports = []
        for scenario, edits, markers in _scenario_edits(large, sources, prefix):
            # Only selected S0 files and one unique new root-level helper are touched.
            restore = {path: sources.get(path) for path in edits}
            _write_files(box.container, edits)
            try:
                captured = snapshotter.capture()
                delta = captured["delta"]
                semantic = build_semantic_delta(delta, store)
                v1_text, v1_stats = budgeted_state_text(delta, token_length=token_length, max_tokens=max_tokens)
                v2_text, v2_stats = budgeted_semantic_state_text(
                    semantic,
                    token_length=token_length,
                    max_tokens=max_tokens,
                )
                v1_markers = [marker for marker in markers if marker in v1_text]
                v2_markers = [marker for marker in markers if marker in v2_text]
                report = {
                    "scenario": scenario,
                    "state_hash": captured["state_hash"],
                    "edited_paths": list(edits),
                    "expected_markers": markers,
                    "v1": {**v1_stats, "markers_included": v1_markers, "text": v1_text},
                    "v2": {**v2_stats, "markers_included": v2_markers, "text": v2_text},
                }
                if v2_markers != markers:
                    raise AssertionError(f"{scenario}: v2 dropped a controlled changed line")
                if v2_stats["changed_lines_included"] != v2_stats["changed_lines_total"]:
                    raise AssertionError(f"{scenario}: v2 did not cover all controlled changed lines")
                if scenario == "large_file_one_line" and v1_markers:
                    raise AssertionError("Large-file scenario did not exercise v1 whole-body omission")
                reports.append(report)
            finally:
                _write_files(box.container, restore)
                restored = snapshotter.capture()
                if restored["delta"]["filesystem"]:
                    raise AssertionError("Controlled edits did not restore the original filesystem S0")
            report["restored_filesystem_delta_count"] = 0
            (output_dir / (scenario + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {
            "instance_id": row["instance_id"],
            "baseline": baseline,
            "s0_blobs_match_workspace": True,
            "max_tokens": max_tokens,
            "large_file_tokens": token_length(sources[large]),
            "scenarios": reports,
            "passed": True,
        }
    finally:
        box.close()


def main() -> None:
    import docker
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Prepared train/val JSONL")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.limit < 1 or args.max_tokens < 1:
        parser.error("limit and max-tokens must be positive")
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.instance_id:
        rows = [row for row in rows if row["instance_id"] in args.instance_id]
        missing = set(args.instance_id) - {row["instance_id"] for row in rows}
        if missing:
            parser.error(f"Requested instances not found: {sorted(missing)}")
    rows = rows[: args.limit]
    if not rows:
        parser.error("No tasks selected")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    client = docker.from_env(timeout=120)
    reports = []
    try:
        for index, row in enumerate(rows):
            output_dir = args.output / f"task-{index:02d}"
            output_dir.mkdir()
            report = audit_task(row, client, tokenizer, output_dir, args.max_tokens)
            reports.append(report)
            print(
                json.dumps({"instance_id": row["instance_id"], "passed": True, "scenarios": len(report["scenarios"])}),
                flush=True,
            )
        (args.output / "audit.json").write_text(
            json.dumps({"passed": True, "task_count": len(reports), "tasks": reports}, indent=2),
            encoding="utf-8",
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
