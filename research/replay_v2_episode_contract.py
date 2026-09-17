"""Replay saved real actions through production adapters, without model calls.

Any adjudicated reward is an explicit in-memory audit overlay; original files
are hashed before/after and never rewritten. This does not resume training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlightning.schemas import Event, Rollout
from agentlightning.server.routes.events import _to_triplet_format
from agentlightning.verl.agl_rollout_manager import AglRolloutManager, EnqueuedRollout
from agentlightning.verl.episode_contract import validate_completed
from agentlightning.verl.full_dataset import merge_validation_metrics
from agentlightning.verl.reliability_metrics import rollout_diagnostics
from agentlightning.verl.rollout_adapter import RolloutAdapter
import torch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def adjudications(path):
    if path is None:
        return {}
    summary = json.loads((path / "summary.json").read_text())
    assert summary["status"] == "complete" and len(summary["cases"]) == 28
    accepted = {}
    for case in summary["cases"]:
        assert case["outcome"] == "candidate_reproduced_exit", case
        directory = Path(case["directory"])
        original = json.loads((directory / "original-grade.json").read_text())
        reference = json.loads((Path(case["legs"]["reference"]["output_directory"]) / "grade.json").read_text())
        candidate = json.loads((Path(case["legs"]["candidate"]["output_directory"]) / "grade.json").read_text())
        assert original["patch_sha256"] == candidate["patch_sha256"] == case["patch_sha256"]
        assert original["pytest_exit"] == candidate["pytest_exit"] in {2, 3, 4, 5, 124, 137}
        assert reference["resolved"] and reference["reward"] == 1 and reference["pytest_exit"] == 0
        assert reference["reference_control"] is True and candidate["reference_control"] is False
        assert original["reward"] == candidate["reward"] == 0
        for key in ("container_memory_bytes", "container_nano_cpus", "eval_timeout_seconds",
                    "grading_protocol", "test_runner", "f2p_total", "p2p_total"):
            assert original[key] == reference[key] == candidate[key], (case["index"], key)
        assert original["baseline"]["image_id"] == reference["baseline"]["image_id"] == candidate["baseline"]["image_id"]
        if original["pytest_exit"] == 137:
            evidence = json.loads((Path(case["legs"]["candidate"]["output_directory"]) / "container-evidence.json").read_text())
            assert any(e.get("Actor", {}).get("ID") == candidate["container_id"] for e in evidence["oom_events"])
        if original["pytest_exit"] == 124:
            assert case["legs"]["candidate"]["duration_s"] >= original["eval_timeout_seconds"]
        accepted[case["rollout_id"]] = case
    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite replay evidence"
    args.output.mkdir()
    audit_path, = (args.original / "checkpoints/episode-audit").glob("val-step-0-*.json")
    audit = json.loads(audit_path.read_text())
    assert len(audit["episodes"]) == 470
    reviewed = adjudications(args.adjudication)
    adapter = RolloutAdapter(max_prompt_length=32768, max_response_length=4096,
                             device=torch.device("cpu"), pad_token_id=0)
    manager = object.__new__(AglRolloutManager)
    records, metrics, held = [], [], []
    for episode in audit["episodes"]:
        rid = episode["rollout_id"]
        path = args.original / "traces" / (rid + ".json")
        raw = path.read_bytes()
        saved = json.loads(raw)
        assert saved["rollout"]["is_train"] is False and saved["rollout"]["rollout_id"] == rid
        events = copy.deepcopy(saved["events"][saved["rollout"]["status"]["last_attempt_id"]])
        if episode["reward"] is None:
            if rid not in reviewed:
                held.append(rid)
                continue
            case = reviewed[rid]
            assert sha(raw) == case["trace_sha256"]
            assert sha(json.dumps(saved["rollout"]["input"], sort_keys=True).encode()) == case["task_sha256"]
            patch = args.original / "agent" / rid / "model.patch"
            assert sha(patch.read_bytes()) == case["patch_sha256"]
            outcomes = [e for e in events if e["event_type"] == "rollout_outcome"]
            assert len(outcomes) == 1 and not any(e["event_type"] == "reward" for e in events)
            outcomes[0]["data"]["grading_status"] = "candidate_failed"
            events.append({**{k: v for k, v in outcomes[0].items() if k != "data"},
                           "event_type": "reward", "data": {"value": 0.0,
                           "reason": outcomes[0]["data"]["reason"], "source": "offline_fixed_patch_audit_overlay"}})
        raw_events = [Event.model_validate(e) for e in events]
        trimmed = [_to_triplet_format(e) for e in raw_events]
        manager._fetch_rollout_events = lambda _rid: (raw_events, trimmed)
        rollout = Rollout.model_validate(saved["rollout"])
        queued = EnqueuedRollout(rollout_id=rid, data_id=rid, step=0,
                                  sample_idx_in_step=episode["sample_idx"], enqueue_time=0,
                                  input=saved["rollout"]["input"])
        built = manager._build_completed_rollout(queued, rollout)
        checked = validate_completed(built, audit["policy_version"])
        values = adapter.get_test_metrics([checked], global_steps=0)
        values.update(rollout_diagnostics([checked]))
        metrics.append(values)
        records.append({"rollout_id": rid, "trace_sha256": sha(raw), "original_reward": episode["reward"],
                        "adjudicated_overlay": rid in reviewed, "checked_reward": checked.final_reward,
                        "calls": len(checked.triplets), "action_tokens": sum(len(t.response['token_ids']) for t in checked.triplets)})
        assert sha(path.read_bytes()) == sha(raw)
    merged = merge_validation_metrics(metrics)
    result = {"accepted": len(records), "held": held, "passed": len(records) == 470 and not held,
              "original_files_unchanged": True, "audit_overlay_only": True,
              "source_sha256": {name: sha((ROOT / name).read_bytes()) for name in (
                  "agentlightning/verl/episode_contract.py", "agentlightning/verl/reliability_metrics.py",
                  "agentlightning/verl/agl_rollout_manager.py", "agentlightning/server/routes/events.py",
                  "agentlightning/verl/rollout_adapter.py", "agentlightning/verl/full_dataset.py")},
              "metrics": {k: v for k, v in merged.items() if isinstance(v, (int, float))}, "episodes": records}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"episodes", "held"}}))
    if args.adjudication and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
