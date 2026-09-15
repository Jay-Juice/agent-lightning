# Copyright (c) Microsoft. All rights reserved.

"""Bounded image prefetch while the independent environment audit checks tests."""

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def fetch(image, output, reserve):
    if "@sha256:" not in image:
        raise ValueError("Only pinned images may be prefetched")
    started = time.monotonic()
    existing = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    cached = existing.returncode == 0
    if not cached:
        available = shutil.disk_usage(output).free / 2**30
        if available < reserve:
            raise RuntimeError(f"Prefetch stopped at disk reserve: {available:.1f} GiB")
        with (output / (image.split("@sha256:")[1] + ".log")).open("x") as log:
            for attempt in range(3):
                try:
                    subprocess.run(
                        ["docker", "pull", image], stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True
                    )
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    if attempt == 2:
                        return {"image": image, "passed": False, "error": str(exc)}
                    time.sleep(5 * (attempt + 1))
    return {"image": image, "passed": True, "already_cached": cached, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--min-free-gib", type=int, default=360)
    args = parser.parse_args()
    if args.min_free_gib < 240:
        raise ValueError("Prefetch must reserve at least 240 GiB")
    manifest = json.loads((args.data / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 6315, "val": 470} and len(manifest["images"]) == 124
    args.output.mkdir(parents=True, exist_ok=False)
    # The grader progresses forward; fetch from the other end to avoid waiting
    # for the same image. Docker also coalesces pulls of an identical digest.
    images = iter(reversed(manifest["images"]))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, next(images), args.output, args.min_free_gib) for _ in range(args.workers)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for task in done:
                result = task.result()
                results.append(result)
                print(json.dumps({"completed": len(results), **result}), flush=True)
                (args.output / "results.json").write_text(json.dumps(results, indent=2))
                image = next(images, None)
                if image is not None:
                    pending.add(pool.submit(fetch, image, args.output, args.min_free_gib))
    if any(not result["passed"] for result in results):
        raise RuntimeError("Some image pulls failed after retries; see results.json")


if __name__ == "__main__":
    main()
