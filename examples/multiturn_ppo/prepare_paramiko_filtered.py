# Copyright (c) Microsoft. All rights reserved.
"""Record the three approved empty-patch successes in a new derived dataset."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_full_python import apply_exclusions, load_prepared_data

IDS = (
    "paramiko__paramiko.23f92003.lm_rewrite__22j0m5ms",
    "paramiko__paramiko.23f92003.lm_rewrite__ekwh2bqe",
    "paramiko__paramiko.23f92003.func_basic__pi2mww3n",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, splits = load_prepared_data(args.source)
    rows = {row["instance_id"]: row for row in splits["train"]}
    records = []
    for iid in IDS:
        path = args.controls / iid / "result.json"
        control = json.loads(path.read_text())
        assert control["instance_id"] == iid and control["split"] == "train"
        for mode in ("empty", "reference"):
            grade = control[mode]
            assert grade["reward"] == 1 and grade["pytest_exit"] == 0
            assert grade["f2p_passed"] == grade["f2p_total"] > 0
            assert grade["p2p_passed"] == grade["p2p_total"]
        records.append(
            {
                "instance_id": iid,
                "split": "train",
                "source_row_sha256": rows[iid]["source_row_sha256"],
                "reason": "Empty patch passes all declared F2P and same-file P2P tests; false-positive reward.",
                "evidence": str(path),
                "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "empty_reward": control["empty"]["reward"],
                "reference_reward": control["reference"]["reward"],
            }
        )
    filtered = apply_exclusions(splits, records)
    authorization = (
        manifest["exclusion_authorization"]
        + " On 2026-09-14 the user explicitly approved excluding these three Paramiko empty-patch successes "
        "from the new GPU 4-7 baseline, preserving original data and all 470 validation tasks."
    )
    exclusions = {
        "schema_version": 1,
        "authorization": authorization,
        "tasks": manifest["excluded_invalid_tasks"] + records,
    }
    exclusions_bytes = (json.dumps(exclusions, indent=2) + "\n").encode()
    manifest.update(
        {
            "counts": {key: len(value) for key, value in filtered.items()},
            "excluded_invalid_tasks": exclusions["tasks"],
            "exclusion_authorization": authorization,
            "exclusions_sha256": hashlib.sha256(exclusions_bytes).hexdigest(),
            "parent_dataset": str(args.source),
            "parent_sha256": {
                name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                for name in ("train.jsonl", "val.jsonl", "manifest.json")
            },
            "images": sorted({row["image"] for values in filtered.values() for row in values}),
            "steps_for_four_complete_epochs": 4 * ((len(filtered["train"]) + 31) // 32),
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    # Retained records and the entire validation file remain byte-identical.
    lines = (args.source / "train.jsonl").read_bytes().splitlines(keepends=True)
    (args.output / "train.jsonl").write_bytes(
        b"".join(line for line in lines if line.strip() and json.loads(line)["instance_id"] not in IDS)
    )
    (args.output / "val.jsonl").write_bytes((args.source / "val.jsonl").read_bytes())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "exclusions.json").write_bytes(exclusions_bytes)
    checked, _ = load_prepared_data(args.output)
    assert checked["counts"] == {"train": 6248, "val": 470}
    print(
        json.dumps(
            {
                "counts": checked["counts"],
                "steps": checked["steps_for_four_complete_epochs"],
                "new_exclusions": list(IDS),
            }
        )
    )


if __name__ == "__main__":
    main()
