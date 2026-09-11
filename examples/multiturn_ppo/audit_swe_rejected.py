# Copyright (c) Microsoft. All rights reserved.

"""Replay rejected model actions offline; grade source changes without newly created test scripts."""

import argparse
import json
from pathlib import Path

import docker
from sandbox import forbidden_patch_path, load_smith
from smith_docker_agent import SmithSandbox, agent_task, grade


def replay(run, output, record):
    rid = record["rollout_id"]
    trace = json.loads((run / "traces" / f"{rid}.json").read_text())
    row = trace["rollout"]["input"]
    turns = [json.loads(line) for line in (run / "agent" / rid / "trajectory.jsonl").read_text().splitlines()]
    directory = output / rid
    directory.mkdir()
    smith = load_smith()
    client = docker.from_env(timeout=120)
    box = SmithSandbox(client, agent_task(row), output.name + "-" + rid)
    executions = []
    try:
        box.prepare()
        for turn in turns:
            if "action" not in turn:
                continue
            action = turn["action"]
            blocked = smith._forbidden_action(action)
            code, _text = (1, blocked) if blocked else box.execute(action)
            executions.append({"turn": turn["turn"], "returncode": code, "original_returncode": turn["returncode"]})
        box.git("add", "-A")
        paths = [path for path in box.git("diff", "--cached", "--name-only", "-z").split("\0") if path]
        excluded = [path for path in paths if forbidden_patch_path(path)]
        assert excluded and set(paths) == set(record["changed_paths"]), "Replay changed an unexpected set of files"
        for path in excluded:
            # This counterfactual excludes ONLY new reproduction tests.
            # Tracked tests/config changes remain invalid and cannot be bypassed.
            assert path.endswith(".py") and any(part.startswith("test_") for part in Path(path).parts)
            assert not box.git("ls-tree", "--name-only", "HEAD", "--", path).strip()
        box.git("reset", "HEAD", "--", *excluded)
        assert all(path.startswith("src/") and path.endswith(".py") for path in paths if path not in excluded)
        raw = box.git("diff", "--cached", "--raw")
        assert not any("120000" in line.split("\t", 1)[0] for line in raw.splitlines())
        patch = box.git("diff", "--cached", "--binary", "--no-ext-diff", "--full-index")
        (directory / "source-only.patch").write_text(patch)
    finally:
        box.close()
        client.close()
    report = grade(row, patch, directory)
    result = {
        "instance_id": row["instance_id"],
        "rollout_id": rid,
        "excluded_new_files": excluded,
        "executions": executions,
        "source_only_grade": report,
        "original_reward": record["reward"],
        "scope": "Counterfactual diagnosis only; original model evaluation is unchanged",
    }
    (directory / "replay.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"instance_id": row["instance_id"], "source_only_resolved": report["resolved"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.run / "evaluation-summary.json").read_text())
    results = []
    for record in summary["records"]:
        if record["patch_rejection"] == "forbidden_test_or_config_change":
            results.append(replay(args.run, args.output, record))
            (args.output / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
