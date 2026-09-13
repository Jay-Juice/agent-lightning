"""Replay affected Paramiko controls and saved model patches without GPU work."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from full_python_agent import PARAMIKO_TEST_ALIASES, grade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--model-only", action="store_true", help="Only regrade saved validation patches")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = []
    for split in ("train", "val"):
        for line in (args.data / (split + ".jsonl")).read_text().splitlines():
            row = json.loads(line)
            if row["instance_id"].startswith("paramiko__paramiko.23f92003.") and any(
                node in PARAMIKO_TEST_ALIASES for node in row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]
            ):
                selected.append((split, row))
    patches = {}
    for path in (args.run / "traces").glob("*.json"):
        rollout = json.loads(path.read_text())["rollout"]
        if rollout["is_train"]:
            continue
        patches[rollout["input"]["instance_id"]] = args.run / "agent" / rollout["rollout_id"] / "model.patch"
    if args.model_only:
        selected = [(split, row) for split, row in selected if row["instance_id"] in patches]

    def check(item):
        split, row = item
        root = args.output / row["instance_id"]
        root.mkdir()
        result = {"split": split, "instance_id": row["instance_id"]}
        for kind in (("model",) if args.model_only else ("empty", "reference", "model")):
            if kind == "model" and row["instance_id"] not in patches:
                continue
            output = root / kind
            output.mkdir()
            patch = patches[row["instance_id"]].read_text() if kind == "model" else ""
            try:
                result[kind] = grade(row, patch, output, reference=kind == "reference")
            except Exception as exc:
                result[kind] = {"error": repr(exc)}
        if args.model_only:
            result["passed"] = "reward" in result["model"] and result["model"].get("pytest_exit") != 5
        else:
            result["passed"] = (
                result["empty"].get("reward") == 0
                and result["empty"].get("pytest_exit") == 1
                and result["reference"].get("reward") == 1
            )
        (root / "result.json").write_text(json.dumps(result, indent=2))
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(check, item) for item in selected]):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(selected),
                        "id": result["instance_id"],
                        "passed": result["passed"],
                    }
                ),
                flush=True,
            )
    summary = {
        "scope": "saved-model-replay" if args.model_only else "empty-and-reference-controls-with-model-replay",
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "model_replayed": sum("model" in r for r in results),
        "model_success": sum(r.get("model", {}).get("reward") == 1 for r in results),
        "reference_passed": sum(r.get("reference", {}).get("reward") == 1 for r in results),
        "empty_already_passes": [r["instance_id"] for r in results if r.get("empty", {}).get("reward") == 1],
        "failures": [r["instance_id"] for r in results if not r["passed"]],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
