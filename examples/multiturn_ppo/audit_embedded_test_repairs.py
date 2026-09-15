# Copyright (c) Microsoft. All rights reserved.

"""Audit original inline-test mutations and their normalized patch round trip."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from check_full_python_ready import verify_branches
from full_python_agent import FullPythonSandbox, grade, pilot
from prepare_full_python import load_prepared_data


def check_task(row, directory, test_only):
    directory.mkdir()
    result = {"instance_id": row["instance_id"], "source_row_sha256": row["source_row_sha256"], "test_only": test_only}
    try:
        empty = directory / "empty"
        empty.mkdir()
        result["empty"] = grade(row, "", empty)
        if not test_only:
            reference = directory / "reference"
            reference.mkdir()
            result["reference"] = grade(row, "", reference, reference=True)
            client = docker.from_env(timeout=370)
            box = FullPythonSandbox(client, pilot.agent_task(row), row["instance_id"] + "-roundtrip")
            try:
                result["preparation"] = box.prepare()
                paths = box.git("diff", "--name-only", "-z", "HEAD~1", "HEAD~2").strip("\0").split("\0")
                trusted_patch = box.git("diff", "HEAD", "HEAD~2", "--binary", "--", *paths)
                box.copy_bytes(trusted_patch.encode())
                box.git("apply", "--check", "/root/candidate.patch")
                box.git("apply", "/root/candidate.patch")
                patch, changed_paths, rejection = box.export_patch()
                result["export"] = {"bytes": len(patch.encode()), "paths": changed_paths, "rejection": rejection}
                if not patch or rejection:
                    raise RuntimeError(f"Reference source repair was rejected: {result['export']}")
                (directory / "reference-export.patch").write_text(patch)
            finally:
                box.close()
                client.close()
            exported = directory / "exported"
            exported.mkdir()
            result["exported"] = grade(row, patch, exported)
            result["passed"] = (
                result["empty"]["reward"] == 0
                and result["reference"]["reward"] == 1
                and result["exported"]["reward"] == 1
            )
        else:
            # A test-only mutation has no remaining source-repair target. Its
            # normalized empty baseline should pass; it must not enter training.
            result["diagnosis_confirmed"] = result["empty"]["reward"] == 1
    except Exception as exc:
        result["error"] = repr(exc)
        result["passed"] = False
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="pydata__patsy.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    _, splits = load_prepared_data(args.data)
    rows = {
        row["instance_id"]: row
        for values in splits.values()
        for row in values
        if row["instance_id"].startswith(args.prefix)
    }
    client = docker.from_env(timeout=120)
    branches = []
    try:
        for image in sorted({row["image"] for row in rows.values()}):
            branches.append(verify_branches(client, image, [iid for iid, row in rows.items() if row["image"] == image]))
    finally:
        client.close()
    (args.output / "branches.json").write_text(json.dumps(branches, indent=2))
    signature = {
        "data": {
            name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "val.jsonl", "manifest.json")
        },
        "source": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "full_python_agent.py",
                "check_full_python_ready.py",
                "audit_embedded_test_repairs.py",
                "sandbox.py",
            )
        },
    }
    (args.output / "signature.json").write_text(json.dumps(signature, indent=2))
    affected = {record["instance_id"] for branch in branches for record in branch["embedded_test_repairs"]}
    test_only = {record["instance_id"] for branch in branches for record in branch["test_only_tasks"]}
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {
            pool.submit(check_task, rows[iid], args.output / iid, iid in test_only): iid for iid in sorted(affected)
        }
        for future in as_completed(pending):
            record = future.result()
            results.append(record)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in ("instance_id", "test_only", "passed", "diagnosis_confirmed", "error")
                        if k in record
                    }
                ),
                flush=True,
            )
    summary = {
        "affected": len(affected),
        "test_only": len(test_only),
        "test_only_confirmed": sum(bool(r.get("diagnosis_confirmed")) for r in results),
        "mixed_passed": sum(bool(r.get("passed")) for r in results if not r["test_only"]),
        "mixed_total": len(affected - test_only),
        "errors": [r["instance_id"] for r in results if "error" in r],
        "unsupported": [r for branch in branches for r in branch["inline_test_conflicts"]],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
