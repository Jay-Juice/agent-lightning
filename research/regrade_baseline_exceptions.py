"""Serial, model-free candidate/reference regrading using a frozen baseline grader.

Only reads the original task and candidate patch. Writes exclusively under a new
output directory; does not change training, original logs, or reward protocols.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def resources(path: Path) -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    return {
        "available_memory_gib": mem["MemAvailable"] / 2**30,
        "free_disk_gib": shutil.disk_usage(path).free / 2**30,
        "load1": os.getloadavg()[0],
        "cpu_count": os.cpu_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory to preserve prior artifacts")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    os.environ["SMITH_EVAL_TIMEOUT"] = "600"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    source = args.baseline_source.resolve()
    sys.path[:0] = [str(source / "examples" / "multiturn_ppo"), str(source)]
    grader = importlib.import_module("full_python_agent")
    if Path(grader.__file__).resolve() != source / "examples" / "multiturn_ppo" / "full_python_agent.py":
        raise RuntimeError("Imported grader differs from the requested frozen baseline source")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    report = {
        "baseline_source": str(source), "baseline_commit_expected": "5fa18418331e0910ea2cd19ce5fb449282810ce4",
        "grader_sha256": hashlib.sha256(Path(grader.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "eval_timeout_seconds": 600, "concurrency": 1, "actor_calls": 0,
        "started_at": time.time(), "resource_checks": [], "cases": [], "status": "running",
    }
    out = args.output / "summary.json"

    def save():
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def log(event):
        text = json.dumps({"time": time.time(), **event})
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text, flush=True)

    reference_failures = 0
    for index, case in enumerate(cases):
        original = args.logs_root / case["run_name"]
        rid = case["rollout_id"]
        trace = json.loads((original / "traces" / (rid + ".json")).read_text())
        row = trace["rollout"]["input"]
        if trace["rollout"]["is_train"] or row["instance_id"] != case["instance_id"]:
            raise RuntimeError("Task identity or validation provenance mismatch")
        patch_bytes = (original / "agent" / rid / "model.patch").read_bytes()
        patch = patch_bytes.decode("utf-8")
        item = {**case, "candidate_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
                "task_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(), "legs": {}}
        report["cases"].append(item)
        for leg in ("reference", "candidate"):
            state = resources(args.output)
            report["resource_checks"].append({"case_index": index, "leg": leg, **state})
            if state["available_memory_gib"] < 64 or state["free_disk_gib"] < 100 or state["load1"] > 0.9 * state["cpu_count"]:
                report["status"] = "stopped_resource_guard"
                log({"event": report["status"], "resources": state})
                save()
                return
            directory = args.output / f"audit-{args.output.name[-18:]}-{index:02d}-{leg}"
            directory.mkdir()
            log({"event": "grade_started", "case_index": index, "instance_id": row["instance_id"], "leg": leg})
            started = time.monotonic()
            try:
                result = grader.grade(row, patch, directory, reference=leg == "reference")
                details = {k: result.get(k) for k in (
                    "reward", "resolved", "pytest_exit", "f2p_passed", "f2p_total",
                    "p2p_passed", "p2p_total", "test_runner", "reference_control", "eval_timeout_seconds",
                )}
                details["status"] = "graded"
            except Exception as exc:
                details = {"status": "exception", "exception_type": type(exc).__name__, "error": str(exc)[-2000:]}
            details["duration_s"] = time.monotonic() - started
            details["output_directory"] = str(directory)
            item["legs"][leg] = details
            log({"event": "grade_finished", "case_index": index, "leg": leg,
                 **{k: details.get(k) for k in ("status", "resolved", "pytest_exit", "duration_s")}})
            if leg == "reference" and not details.get("resolved", False):
                reference_failures += 1
                item["outcome"] = "reference_failed_candidate_not_interpretable"
                report["reference_failures"] = reference_failures
                save()
                if reference_failures >= 2:
                    report["status"] = "stopped_after_two_reference_failures"
                    save()
                    return
                # Do not spend another grading budget on an uninterpretable pair.
                break
            if leg == "candidate":
                if details.get("resolved"):
                    item["outcome"] = "candidate_and_reference_pass"
                elif details.get("status") == "exception" or details.get("pytest_exit") in (124, 137):
                    item["outcome"] = "reference_pass_candidate_execution_failure_cause_unresolved"
                else:
                    item["outcome"] = "reference_pass_candidate_fail"
            save()
    report["status"] = "complete"
    report["finished_at"] = time.time()
    report["reference_failures"] = reference_failures
    save()
    log({"event": "complete", "cases": len(report["cases"]), "reference_failures": reference_failures})


if __name__ == "__main__":
    main()
