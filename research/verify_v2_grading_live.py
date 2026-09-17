"""Exercise the production fixed-patch adjudicator in isolated main processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


SAMPLES = (92, 248, 4, 251, 249, 331)
MODULES = ("full_python_agent.py", "swe_reliability.py", "swe_grading_evidence.py",
           "smith_docker_agent.py", "sandbox.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_hashes(source):
    return {name: sha((source / "examples/multiturn_ppo" / name).read_bytes()) for name in MODULES}


def run_case(args):
    case = json.loads((args.case / "case.json").read_text())
    manifest = json.loads((args.case.parent / "manifest.json").read_text())
    started = time.time()
    result = {"sample_idx": case["sample_idx"], "rollout_id": case["rollout_id"],
              "started_at": started, "passed": False}
    try:
        if source_hashes(args.source) != manifest["source_hashes"]:
            raise RuntimeError("Production source changed after audit preparation")
        sys.path[:0] = [str(args.source / "examples/multiturn_ppo"), str(args.source)]
        grader = importlib.import_module("full_python_agent")
        reliability = importlib.import_module("swe_reliability")
        if Path(grader.__file__).resolve() != args.source / "examples/multiturn_ppo/full_python_agent.py":
            raise RuntimeError("Wrong production grader imported")
        row = json.loads((args.case / "task.json").read_text())
        patch = (args.case / "candidate.patch").read_bytes()
        if sha(patch) != case["patch_sha256"] or sha(json.dumps(row, sort_keys=True).encode()) != case["task_sha256"]:
            raise RuntimeError("Frozen input changed")
        import docker
        import httpx
        import requests

        report = reliability.grade_fixed_patch(
            grader.grade, row, patch.decode("utf-8"), args.case,
            retry_errors=(docker.errors.DockerException, requests.exceptions.RequestException, httpx.TransportError),
            deadline=time.monotonic() + 3000,
        )
        raw_paths = sorted(args.case.glob("grading-*/grade.json"))
        raw_reports = [json.loads(path.read_text()) for path in raw_paths]
        limits_valid = bool(raw_reports) and all(
            r.get("container_memory_bytes") == 4 * 2**30 and r.get("container_nano_cpus") == 2_000_000_000
            and r.get("eval_timeout_seconds") == 600 for r in raw_reports
        )
        result.update(
            grading_status=report.get("grading_status"), reward=report.get("reward"),
            controlled_failure=report.get("controlled_failure"), pytest_exit=report.get("pytest_exit"),
            limits_valid=limits_valid, raw_report_paths=[str(p.relative_to(args.case)) for p in raw_paths],
            passed=report.get("grading_status") == "candidate_failed" and report.get("reward") == 0
            and report.get("resolved") is False and limits_valid,
        )
        if source_hashes(args.source) != manifest["source_hashes"]:
            raise RuntimeError("Production source changed during audit")
    except Exception as exc:
        result.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
    result.update(finished_at=time.time(), elapsed_seconds=time.time() - started)
    write(args.case / "result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void", SMITH_EVAL_TIMEOUT="600",
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    if args.case:
        return run_case(args)
    if args.original is None or args.output is None:
        parser.error("--original and --output are required outside case mode")
    output = args.output
    if (output / "manifest.json").exists():
        raise RuntimeError("Refuse to overwrite audit evidence")
    output.mkdir(exist_ok=True)
    audits = list((args.original / "checkpoints/episode-audit").glob("val-step-0-*.json"))
    if len(audits) != 1:
        raise RuntimeError("Ambiguous original validation audit")
    episodes = json.loads(audits[0].read_text())["episodes"]
    if len(episodes) != 470:
        raise RuntimeError("Expected 470 original validation episodes")
    cases = []
    for idx in SAMPLES:
        matches = [e for e in episodes if e["sample_idx"] == idx]
        if len(matches) != 1 or matches[0]["outcome"]["grading_status"] != "requires_review":
            raise RuntimeError(f"Unexpected original case: {idx}")
        episode = matches[0]
        rid = episode["rollout_id"]
        trace_bytes = (args.original / "traces" / (rid + ".json")).read_bytes()
        trace = json.loads(trace_bytes)
        row = trace["rollout"]["input"]
        if trace["rollout"]["is_train"] or trace["rollout"]["rollout_id"] != rid:
            raise RuntimeError("Original task identity mismatch")
        patch = (args.original / "agent" / rid / "model.patch").read_bytes()
        grade_bytes = (args.original / "agent" / rid / "grade.json").read_bytes()
        original_grade = json.loads(grade_bytes)
        if original_grade["patch_sha256"] != sha(patch):
            raise RuntimeError("Original candidate/grade hash mismatch")
        case_dir = output / f"sample-{idx:03d}-{rid}"
        case_dir.mkdir()
        case = dict(sample_idx=idx, rollout_id=rid, instance_id=row["instance_id"], directory=str(case_dir),
                    patch_sha256=sha(patch), task_sha256=sha(json.dumps(row, sort_keys=True).encode()),
                    trace_sha256=sha(trace_bytes), original_grade_sha256=sha(grade_bytes),
                    original_exit=original_grade["pytest_exit"])
        write(case_dir / "case.json", case)
        write(case_dir / "task.json", row)
        write(case_dir / "episode.json", episode)
        (case_dir / "original-grade.json").write_bytes(grade_bytes)
        (case_dir / "candidate.patch").write_bytes(patch)
        cases.append(case)
    hashes = source_hashes(args.source)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in MODULES:
        data = (args.source / "examples/multiturn_ppo" / name).read_bytes()
        if sha(data) != hashes[name]:
            raise RuntimeError("Production source changed while snapshotting")
        (snapshot / name).write_bytes(data)
    write(output / "manifest.json", dict(original=str(args.original), source=str(args.source),
          source_hashes=hashes, script_sha256=sha(Path(__file__).read_bytes()), cases=cases,
          audit_sha256=sha(audits[0].read_bytes()), workers=2, case_deadline_seconds=3000,
          model_calls=0, started_at=time.time()))

    def launch(case):
        directory = Path(case["directory"])
        print(json.dumps({"event": "started", "sample_idx": case["sample_idx"], "time": time.time()}), flush=True)
        with (directory / "worker.log").open("w") as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--source", str(args.source),
                                        "--case", str(directory)], stdout=log, stderr=subprocess.STDOUT, check=False)
        write(directory / "worker.exit.json", {"exit_code": completed.returncode})
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            "sample_idx": case["sample_idx"], "passed": False, "error": "Worker exited without result"}
        if completed.returncode:
            result["passed"] = False
        result["worker_exit"] = completed.returncode
        print(json.dumps({"event": "finished", **result}), flush=True)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, case): case for case in cases}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"sample_idx": futures[future]["sample_idx"], "passed": False, "error": repr(exc)})
            write(output / "summary.json", {"status": "running", "cases": results})
    passed = len(results) == 6 and all(r["passed"] for r in results)
    write(output / "summary.json", {"status": "complete", "passed": passed,
          "finished_at": time.time(), "cases": sorted(results, key=lambda r: r["sample_idx"])})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
