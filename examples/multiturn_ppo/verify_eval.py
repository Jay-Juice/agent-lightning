# Copyright (c) Microsoft. All rights reserved.

"""Verify a completed SWE val-only run and summarize independent grading artifacts."""

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run):
    cfg = read_json(run / "resolved-config.json")
    assert (run / "run.exit").read_text().strip() == "0", "Run did not exit successfully"
    assert cfg["trainer"]["val_only"] and cfg["trainer"]["val_before_train"]
    assert cfg["trainer"]["resume_mode"] == "disable", "Expected the original model, not a checkpoint"
    assert cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"] == 1
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert metrics and all(row["step"] == 0 for row in metrics), "Unexpected training step"
    assert all("actor/grad_norm" not in row["data"] and "critic/grad_norm" not in row["data"] for row in metrics)
    assert not list((run / "checkpoints").glob("global_step_*")), "Unexpected training checkpoint"
    assert not list((run / "ppo-audit").glob("step-*.pt")), "Unexpected PPO update audit"
    assert not list((run / "traces").glob("*.failed.json")), "Failed rollout"
    expected = [row["instance_id"] for row in read_json(run / "datasets.json")["validation"]]
    records = []
    for file in sorted((run / "traces").glob("*.json")):
        trace = read_json(file)
        rollout = trace["rollout"]
        assert not rollout["is_train"] and rollout["status"]["state"] == "succeeded"
        directory = run / "agent" / rollout["rollout_id"]
        grade = read_json(directory / "grade.json")
        assert not grade.get("reference_control", False), "A gold control is not a model success"
        events = [event for attempt in trace["events"].values() for event in attempt]
        requests = [event["data"] for event in events if event["event_type"] == "model_request"]
        rewards = [event["data"] for event in events if event["event_type"] == "reward"]
        assert requests and all(request["model_version"] == 0 for request in requests)
        assert len(rewards) == 1 and rewards[0]["value"] == grade["reward"] == float(grade["resolved"])
        turns = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        actions = Counter(turn["action"] for turn in turns if "action" in turn)
        records.append(
            {
                "instance_id": rollout["input"]["instance_id"],
                "rollout_id": rollout["rollout_id"],
                "resolved": grade["resolved"],
                "reward": grade["reward"],
                "calls": len(requests),
                "completion_tokens": sum(request["usage"]["completion_tokens"] for request in requests),
                "stop_reason": rewards[0].get("reason"),
                "patch_bytes": (directory / "model.patch").stat().st_size,
                "patch_rejection": grade.get("patch_rejection"),
                "changed_paths": read_json(directory / "sandbox.json")["changed_paths"],
                "format_errors": sum("response" in turn and "action" not in turn for turn in turns),
                "nonzero_command_exits": sum(turn.get("returncode", 0) != 0 for turn in turns),
                "duplicate_actions": sum(count - 1 for count in actions.values()),
                "pytest_exit": grade.get("pytest_exit"),
                "f2p_passed": grade.get("f2p_passed"),
                "f2p_total": grade.get("f2p_total"),
                "p2p_passed": grade.get("p2p_passed"),
                "p2p_total": grade.get("p2p_total"),
            }
        )
    assert Counter(record["instance_id"] for record in records) == Counter(expected), "Missing or duplicate tasks"
    successes = sum(record["resolved"] for record in records)
    assert abs(metrics[-1]["data"]["val/reward"] - successes / len(records)) < 1e-6
    result = {
        "run": str(run),
        "model": cfg["actor_rollout_ref"]["model"]["path"],
        "gpu_ids": read_json(run / "provenance.json")["cuda_visible_devices"],
        "status": "verified",
        "model_versions": [0],
        "ppo_updates": 0,
        "tasks": len(records),
        "resolved": successes,
        "success_rate": successes / len(records),
        "calls": sum(record["calls"] for record in records),
        "nonempty_patches": sum(record["patch_bytes"] > 0 for record in records),
        "patch_rejections": sum(bool(record["patch_rejection"]) for record in records),
        "records": records,
    }
    (run / "evaluation-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    result = verify(parser.parse_args().run)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
