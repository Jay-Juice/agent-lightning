# Copyright (c) Microsoft. All rights reserved.
"""Prepare every Python row in the published split, with pinned image digests."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize_test_node(node, aliases):
    if "::" in node:
        return node
    match = re.fullmatch(r"(\w+) \(([\w.]+)\.(\w+)\)", node)
    if match:
        return match[2].replace(".", "/") + ".py::" + match[3] + "::" + match[1]
    if node in aliases and len(aliases[node]) == 1:
        return aliases[node][0]
    # These source-module doctests are outside the F2P test files in this
    # release, but still need a file identity for correct same-file filtering.
    if node.startswith("Doctest: tornado."):
        name = node.split(": ", 1)[1]
        return "/".join(name.split(".")[:2]) + ".py::" + name
    if any(char.isspace() for char in node):
        raise ValueError(f"Unmapped native test identifier: {node}")
    return node


def normalize_test_nodes(nodes, aliases):
    # A unittest shortDescription can name one inherited method exercised by
    # several concrete test classes. Preserve all of those cases, not its
    # abstract defining class (which is skipped by the native runner).
    return list(
        dict.fromkeys(
            resolved
            for node in nodes
            for resolved in (aliases[node] if node in aliases else [normalize_test_node(node, aliases)])
        )
    )


def prepare_row(row, image_digests, test_aliases=None):
    f2p, p2p = row["FAIL_TO_PASS"], row["PASS_TO_PASS"]
    if not isinstance(f2p, list) or not f2p or not isinstance(p2p, list):
        raise ValueError("Expected nonempty F2P and list P2P metadata")
    # Go rows in this release use bare TestName identifiers. Keep their IDs in
    # the manifest; the user explicitly selected the complete Python subset.
    if not any(".py" in node for node in f2p + p2p):
        return None
    normalized_f2p = normalize_test_nodes(f2p, test_aliases or {})
    normalized_p2p = normalize_test_nodes(p2p, test_aliases or {})
    paths = {node.split("::", 1)[0] for node in normalized_f2p}
    selected_p2p = [node for node in normalized_p2p if node.split("::", 1)[0] in paths and node not in normalized_f2p]
    if any("::" not in node for node in normalized_f2p + selected_p2p):
        raise ValueError(f"Unsupported test identifiers for {row['instance_id']}")
    digest = image_digests[row["image_name"]]
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ValueError("Image digest must be SHA256")
    result = dict(row)
    result["FAIL_TO_PASS"] = normalized_f2p
    result["PASS_TO_PASS"] = selected_p2p
    result["image"] = row["image_name"] + "@" + digest
    result["grading_protocol"] = "f2p_file"
    result["source_p2p_count"] = len(p2p)
    result["source_row_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return result


def apply_exclusions(splits, records):
    """Apply only individually recorded training exclusions; keep validation fixed."""
    train = {row["instance_id"]: row for row in splits["train"]}
    excluded = {}
    for record in records:
        iid = record["instance_id"]
        if iid in excluded or iid not in train or record["split"] != "train":
            raise ValueError(f"Duplicate, unknown, or non-training exclusion: {iid}")
        if not record["reason"] or not record["evidence"]:
            raise ValueError(f"Missing exclusion provenance: {iid}")
        if record["source_row_sha256"] != train[iid]["source_row_sha256"]:
            raise ValueError(f"Exclusion refers to changed source data: {iid}")
        excluded[iid] = record
    return {**splits, "train": [row for row in splits["train"] if row["instance_id"] not in excluded]}


def load_prepared_data(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    splits = {
        split: [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        for split in ("train", "val")
    }
    ids = {key: {row["instance_id"] for row in rows} for key, rows in splits.items()}
    counts = {key: len(rows) for key, rows in splits.items()}
    assert counts == manifest["counts"] and all(len(ids[key]) == counts[key] for key in splits)
    assert not ids["train"] & ids["val"]
    exclusions = manifest.get("excluded_invalid_tasks", [])
    excluded_ids = {record["instance_id"] for record in exclusions}
    assert len(excluded_ids) == len(exclusions)
    assert all(record["split"] == "train" and record["reason"] and record["evidence"] for record in exclusions)
    assert not excluded_ids & (ids["train"] | ids["val"])
    assert counts["train"] + len(exclusions) == 6315 and counts["val"] == 470
    assert sorted({row["image"] for rows in splits.values() for row in rows}) == manifest["images"]
    assert manifest["steps_for_four_complete_epochs"] == 4 * ((counts["train"] + 31) // 32)
    return manifest, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-aliases", type=Path, help="Verified native-test docstring to pytest node aliases")
    parser.add_argument("--exclude-tasks", type=Path, help="Explicitly approved training exclusions with evidence")
    args = parser.parse_args()
    inventory = json.loads(args.images.read_text())
    aliases = json.loads(args.test_aliases.read_text()) if args.test_aliases else {}
    digests = {row["image"]: row["digest"] for row in inventory["images"] if row["status"] == "verified"}
    splits, excluded, hashes = {}, {}, {}
    for split, name in [
        ("train", "train_dataset_mixed.jsonl"),
        ("val", "val_dataset_filtered.jsonl"),
    ]:
        source = args.source / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        splits[split], excluded[split] = [], []
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            converted = prepare_row(row, digests, aliases)
            if converted is None:
                excluded[split].append(row["instance_id"])
            else:
                splits[split].append(converted)
    ids = {key: {row["instance_id"] for row in rows} for key, rows in splits.items()}
    assert all(len(ids[key]) == len(rows) for key, rows in splits.items()), "Duplicate instance IDs"
    assert not ids["train"] & ids["val"], "Train/validation overlap"
    assert [len(splits[k]) for k in ("train", "val")] == [6315, 470], "Published Python split changed"
    exclusions = json.loads(args.exclude_tasks.read_text()) if args.exclude_tasks else None
    if exclusions is not None:
        assert exclusions["authorization"] and exclusions["schema_version"] == 1
        splits = apply_exclusions(splits, exclusions["tasks"])
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        with (args.output / f"{split}.jsonl").open("x") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source_sha256": hashes,
        "test_aliases_sha256": hashlib.sha256(args.test_aliases.read_bytes()).hexdigest()
        if args.test_aliases
        else None,
        "counts": {key: len(rows) for key, rows in splits.items()},
        "excluded_non_python_ids": excluded,
        "excluded_invalid_tasks": exclusions["tasks"] if exclusions else [],
        "exclusion_authorization": exclusions["authorization"] if exclusions else None,
        "exclusions_sha256": hashlib.sha256(args.exclude_tasks.read_bytes()).hexdigest() if exclusions else None,
        "grading_protocol": "all F2P plus P2P in the same files, matching the official f2p_only protocol",
        "images": sorted({row["image"] for rows in splits.values() for row in rows}),
        "task_batch_size": 32,
        "steps_for_four_complete_epochs": 4 * ((len(splits["train"]) + 31) // 32),
    }
    with (args.output / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps({"counts": manifest["counts"], "images": len(manifest["images"])}))


if __name__ == "__main__":
    main()
