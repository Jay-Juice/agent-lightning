"""Replay saved validation patches without model calls or original-log changes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if (output / "manifest.json").exists():
        raise RuntimeError("Refusing to overwrite audit evidence")
    output.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void", SMITH_EVAL_TIMEOUT="600", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    source = args.source.resolve()
    sys.path[:0] = [str(source / "examples/multiturn_ppo"), str(source)]
    grader = importlib.import_module("full_python_agent")
    if Path(grader.__file__).resolve() != source / "examples/multiturn_ppo/full_python_agent.py":
        raise RuntimeError("Wrong grader imported")
    audits = list((args.original / "checkpoints/episode-audit").glob("val-step-0-*.json"))
    if len(audits) != 1:
        raise RuntimeError("Ambiguous original validation audit")
    audit = json.loads(audits[0].read_text())
    episodes = [e for e in audit["episodes"] if e["outcome"]["grading_status"] == "requires_review"]
    if len(audit["episodes"]) != 470 or len(episodes) != 28:
        raise RuntimeError("Unexpected original audit counts")
    cases = []
    for index, episode in enumerate(episodes):
        rid = episode["rollout_id"]
        trace_path = args.original / "traces" / (rid + ".json")
        trace_bytes = trace_path.read_bytes()
        trace = json.loads(trace_bytes)
        row = trace["rollout"]["input"]
        patch_bytes = (args.original / "agent" / rid / "model.patch").read_bytes()
        original_grade = json.loads((args.original / "agent" / rid / "grade.json").read_text())
        if trace["rollout"]["is_train"] or trace["rollout"]["rollout_id"] != rid:
            raise RuntimeError("Original task identity mismatch")
        if original_grade["patch_sha256"] != digest(patch_bytes):
            raise RuntimeError("Original grade/patch hash mismatch")
        directory = output / f"case-{index:02d}-{rid}"
        directory.mkdir()
        write(directory / "task.json", row)
        write(directory / "original-grade.json", original_grade)
        write(directory / "episode.json", episode)
        (directory / "candidate.patch").write_bytes(patch_bytes)
        cases.append(dict(index=index, rollout_id=rid, instance_id=row["instance_id"], trace_sha256=digest(trace_bytes), task_sha256=digest(json.dumps(row, sort_keys=True).encode()), patch_sha256=digest(patch_bytes), original_exit=original_grade["pytest_exit"], directory=str(directory)))
    source_hashes = {}
    frozen = output / "source-snapshot"
    frozen.mkdir()
    for name in ("full_python_agent.py", "sandbox.py", "smith_docker_agent.py", "swe_reliability.py"):
        data = (source / "examples/multiturn_ppo" / name).read_bytes()
        (frozen / name).write_bytes(data)
        source_hashes[name] = digest(data)
    manifest = dict(original=str(args.original), audit_sha256=digest(audits[0].read_bytes()), source=str(source), source_hashes=source_hashes, script_sha256=digest(Path(__file__).read_bytes()), started_at=time.time(), timeout=600, workers=2, model_calls=0, cases=cases)
    write(output / "manifest.json", manifest)
    lock = threading.Lock()
    local = threading.local()
    original_box = grader.FullPythonSandbox

    class CapturedSandbox(original_box):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            local.evidence["container_id"] = self.container.id
            self.container.reload()
            local.evidence["host_config"] = {k: self.container.attrs["HostConfig"].get(k) for k in ("Memory", "NanoCpus", "PidsLimit", "NetworkMode")}
            if self.container.attrs["HostConfig"]["Memory"] != 4 * 2**30:
                self.close()
                raise RuntimeError("Replay must use original 4GiB limit")

        def close(self):
            try:
                self.container.reload()
                local.evidence["state_before_close"] = self.container.attrs.get("State")
            except Exception as exc:
                local.evidence["inspect_error"] = repr(exc)
            return super().close()

    grader.FullPythonSandbox = CapturedSandbox

    def log(event):
        with lock:
            line = json.dumps({"time": time.time(), **event})
            with (output / "progress.jsonl").open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def replay(case):
        directory = Path(case["directory"])
        row = json.loads((directory / "task.json").read_text())
        patch_bytes = (directory / "candidate.patch").read_bytes()
        if digest(patch_bytes) != case["patch_sha256"] or digest(json.dumps(row, sort_keys=True).encode()) != case["task_sha256"]:
            raise RuntimeError("Frozen task/patch changed")
        result = {**case, "legs": {}}
        for leg in ("reference", "candidate"):
            mem = {k: int(v.split()[0]) * 1024 for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
            if mem["MemAvailable"] < 64 * 2**30 or shutil.disk_usage(output).free < 100 * 2**30:
                raise RuntimeError("Resource guard")
            leg_dir = directory / f"recheck-{case['rollout_id']}-{leg}"
            leg_dir.mkdir()
            local.evidence = {"started_at": time.time()}
            log(dict(event="started", index=case["index"], leg=leg, instance_id=case["instance_id"]))
            try:
                grade = grader.grade(row, patch_bytes.decode(), leg_dir, reference=leg == "reference")
                if bool(grade["reference_control"]) != (leg == "reference"):
                    raise RuntimeError("Reference identity mismatch")
                if leg == "candidate" and grade["patch_sha256"] != case["patch_sha256"]:
                    raise RuntimeError("Candidate changed during grading")
                details = {k: grade.get(k) for k in ("resolved", "reward", "pytest_exit", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total", "container_id", "container_memory_bytes", "container_nano_cpus", "patch_sha256", "reference_control")}
                details["status"] = "graded"
            except Exception as exc:
                details = dict(status="exception", error=repr(exc))
            ended = time.time()
            local.evidence["finished_at"] = ended
            local.evidence["duration_s"] = ended - local.evidence["started_at"]
            cid = local.evidence.get("container_id")
            if cid:
                client = grader.docker.from_env(timeout=15)
                try:
                    # Bounded historical query for the exact container; never a live unbounded stream.
                    since, until = int(local.evidence["started_at"]) - 1, int(ended) + 1
                    events = list(client.events(since=since, until=until, filters={"container": cid}, decode=True))
                    local.evidence.update(events_since=since, events_until=until, events=events, oom_events=[e for e in events if e.get("Action", e.get("status")) == "oom"])
                except Exception as exc:
                    local.evidence["events_error"] = repr(exc)
                finally:
                    client.close()
            write(leg_dir / "container-evidence.json", local.evidence)
            details.update(duration_s=local.evidence["duration_s"], oom_events=len(local.evidence.get("oom_events", [])), output_directory=str(leg_dir))
            result["legs"][leg] = details
            write(directory / "result.json", result)
            log(dict(event="finished", index=case["index"], leg=leg, **details))
            if leg == "reference" and not details.get("resolved"):
                result["outcome"] = "reference_failed"
                break
        if "candidate" in result["legs"]:
            c = result["legs"]["candidate"]
            result["outcome"] = "candidate_passed" if c.get("resolved") else "candidate_reproduced_exit" if c.get("pytest_exit") == case["original_exit"] else "candidate_changed_exit"
        write(directory / "result.json", result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(replay, case): case for case in cases}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({**futures[future], "outcome": "exception", "error": repr(exc)})
            write(output / "summary.json", dict(status="running", cases=sorted(results, key=lambda x: x["index"])))
    write(output / "summary.json", dict(status="complete", finished_at=time.time(), cases=sorted(results, key=lambda x: x["index"])))
    log(dict(event="complete", cases=len(results)))
    if any(r["outcome"] in ("exception", "reference_failed") for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
