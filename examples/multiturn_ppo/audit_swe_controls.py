# Copyright (c) Microsoft. All rights reserved.

"""Recheck every SWE task with broken-code and known-correct-code controls, without an LLM."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sandbox import load_smith
from smith_docker_agent import grade


def control(row, index, output):
    result = {"instance_id": row["instance_id"], "controls": {}}
    for name, reference in (("bug", False), ("reference", True)):
        directory = output / f"task-{index:03d}-{name}"
        directory.mkdir()
        report = grade(row, "", directory, reference=reference)
        text = (directory / "test-output.txt").read_text()
        report["pytest_started"] = "test session starts" in text
        statuses = load_smith().parse_test_statuses(text)
        expected = set(row["FAIL_TO_PASS"] + row["PASS_TO_PASS"])
        report["missing_statuses"] = len(expected - statuses.keys())
        result["controls"][name] = report
    print(
        json.dumps(
            {
                "instance_id": row["instance_id"],
                "bug": result["controls"]["bug"]["resolved"],
                "reference": result["controls"]["reference"]["resolved"],
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four CPU workers on the shared server")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(control, row, index, args.output): row["instance_id"] for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"instance_id": futures[future], "error": repr(exc)})
            (args.output / "progress.json").write_text(json.dumps(results, indent=2))
    summary = {
        "dataset": str(args.dataset),
        "tasks": len(rows),
        "errors": sum("error" in result for result in results),
        "bug_resolved": sum(result.get("controls", {}).get("bug", {}).get("resolved", False) for result in results),
        "reference_resolved": sum(
            result.get("controls", {}).get("reference", {}).get("resolved", False) for result in results
        ),
        "results": sorted(results, key=lambda row: row["instance_id"]),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)
    if summary["errors"] or summary["bug_resolved"] or summary["reference_resolved"] != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
