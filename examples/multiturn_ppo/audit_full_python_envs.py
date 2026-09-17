# Copyright (c) Microsoft. All rights reserved.

"""Cache pinned Python images and check their grader without using any GPU."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from full_python_agent import grade
from prepare_full_python import load_prepared_data


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_image(image, row, output_root, reserve, cached_only):
    directory = output_root / image.split("@sha256:")[1][:16]
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    client = docker.from_env(timeout=120)
    try:
        try:
            client.images.get(image)
            cached = True
        except docker.errors.ImageNotFound:
            cached = False
        if cached_only and not cached:
            return None
        free = shutil.disk_usage(output_root).free / 2**30
        if free < reserve:
            raise RuntimeError(f"Disk reserve reached ({free:.1f} GiB); no further images pulled")
        print(json.dumps({"image": image, "free_gib": free}), flush=True)
        directory.mkdir(exist_ok=True)
        if not cached:
            with (directory / "pull.log").open("a") as log:
                for attempt in range(3):
                    try:
                        subprocess.run(
                            ["docker", "pull", image], stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True
                        )
                        break
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        if attempt == 2:
                            raise
                        time.sleep(5 * (attempt + 1))
        started = time.monotonic()
        result = {"image": image, "instance_id": row["instance_id"]}
        for reference in (False, True):
            name = "reference" if reference else "empty"
            output = directory / name
            output.mkdir(exist_ok=True)
            try:
                result[name] = grade(row, "", output, reference=reference)
            except Exception as exc:
                result[name] = {"error": repr(exc)}
        result["seconds"] = time.monotonic() - started
        result["passed"] = result["empty"].get("reward") == 0 and result["reference"].get("reward") == 1
        result_path.write_text(json.dumps(result, indent=2))
        return result
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-free-gib", type=int, default=240)
    parser.add_argument("--cached-only", action="store_true", help="Audit available images without downloading more")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()
    if args.min_free_gib < 150:
        raise ValueError("Keep at least 150 GiB free while preparing images")
    manifest, splits = load_prepared_data(args.data)
    selected = {}
    for split in ("train", "val"):
        for row in splits[split]:
            # Exercise a larger same-file test suite in each environment.
            score = len(row["FAIL_TO_PASS"]) + len(row["PASS_TO_PASS"])
            if row["image"] not in selected or score > selected[row["image"]][0]:
                selected[row["image"]] = (score, row)
    assert set(selected) == set(manifest["images"])
    signature = {
        "data": {name: fingerprint(args.data / name) for name in ("train.jsonl", "val.jsonl", "manifest.json")},
        "grader": fingerprint(Path(__file__).with_name("full_python_agent.py")),
        "sandbox": fingerprint(Path(__file__).with_name("sandbox.py")),
        "checked_agent": fingerprint(Path(__file__).with_name("smith_docker_agent.py")),
        "data_preparation": fingerprint(Path(__file__).with_name("prepare_full_python.py")),
        "grading_reliability": fingerprint(Path(__file__).with_name("swe_reliability.py")),
        "grading_evidence": fingerprint(Path(__file__).with_name("swe_grading_evidence.py")),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    signature_path = args.output / "signature.json"
    if signature_path.exists():
        assert json.loads(signature_path.read_text()) == signature, "Audit source or data changed; use a new output"
    else:
        signature_path.write_text(json.dumps(signature, indent=2))
    # Use already-cached environments first. Slow pulls should not block the
    # CPU controls for images concurrently prefetched from the other direction.
    client = docker.from_env(timeout=120)
    cached = set()
    try:
        for image in selected:
            try:
                client.images.get(image)
                cached.add(image)
            except docker.errors.ImageNotFound:
                pass
    finally:
        client.close()
    ordered = sorted(selected.items(), key=lambda item: (item[0] not in cached, item[0]))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(check_image, image, row, args.output, args.min_free_gib, args.cached_only): image
            for image, (_, row) in ordered
        }
        for future in as_completed(pending):
            result = future.result()
            if result is None:
                continue
            results.append(result)
            summary = {
                "images_checked": len(results),
                "passed": sum(r["passed"] for r in results),
                "total_images": len(selected),
                "ready": len(results) == len(selected) and all(r["passed"] for r in results),
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "image": result["image"],
                        "passed": result["passed"],
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
        summary = {
            "images_checked": len(results),
            "passed": sum(r["passed"] for r in results),
            "total_images": len(selected),
            "ready": len(results) == len(selected) and all(r["passed"] for r in results),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        if not all(result["passed"] for result in results):
            raise RuntimeError("Environment controls failed; inspect results before training")


if __name__ == "__main__":
    main()
