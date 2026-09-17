# Copyright (c) Microsoft. All rights reserved.

"""Verify complete image coverage and unchanged audit inputs before full PPO."""

import argparse
import inspect
import json
from pathlib import Path

import docker
from audit_full_python_envs import fingerprint
from full_python_agent import embedded_tests, fixture_patch_allowed, recover_embedded_tests
from prepare_full_python import load_prepared_data
from sandbox import forbidden_patch_path


def verify_branches(client, image, ids):
    # checkout <iid> auto-tracks origin/<iid> in these images. Resolve both forms
    # without changing any checkout; inspect every task, including real PR bugs.
    code = (
        "import ast\nfrom pathlib import PurePosixPath\n"
        + inspect.getsource(fixture_patch_allowed)
        + "\n"
        + inspect.getsource(forbidden_patch_path)
        + "\n"
        + inspect.getsource(embedded_tests)
        + "\n"
        + inspect.getsource(recover_embedded_tests)
        + "\n"
        + """import json, subprocess, sys
results = []
protected = []
inline_conflicts = []
source_recoveries = []
test_repairs = []
test_only_tasks = []
for iid in json.loads(sys.argv[1]):
    for ref in [iid, 'origin/' + iid]:
        proc = subprocess.run(['git', '-c', 'safe.directory=/testbed', 'log', '-2', '--format=%s', ref, '--'],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            break
    if proc.returncode or proc.stdout.splitlines() != ['Remove F2P Tests', 'Bug Patch']:
        results.append({'instance_id': iid, 'subjects': proc.stdout.splitlines(), 'error': proc.stderr[-500:]})
        continue
    command = ['git', 'diff', '--no-ext-diff', '--name-only', '-z', ref+'~2', ref+'~1', '--']
    paths = subprocess.check_output(command).decode().split('\\0')
    removed = subprocess.check_output(
        ['git', 'diff', '--name-only', '--diff-filter=D', '-z', ref+'~1', ref, '--']).decode().split('\\0')
    restored = [p for p in removed if p.endswith('.py') and PurePosixPath(p).name != 'test.py'
                and not forbidden_patch_path(p)]
    if restored:
        source_recoveries.append({'instance_id': iid, 'paths': restored})
    production_paths = set(filter(None, paths))
    repaired_paths = []
    for path in set(paths) & set(restored):
        before = subprocess.check_output(['git', 'show', ref+'~1:'+path]).decode()
        after = subprocess.check_output(['git', 'show', ref+'~2:'+path]).decode()
        try:
            supported = embedded_tests(before) == embedded_tests(after)
            if not supported:
                repaired, names, has_bug = recover_embedded_tests(before, after)
                supported = True
                repaired_paths.append(path)
                test_repairs.append({'instance_id': iid, 'path': path, 'tests': names,
                                     'production_bug_retained': has_bug})
                if not has_bug:
                    production_paths.discard(path)
        except (SyntaxError, ValueError):
            supported = False
        if not supported:
            inline_conflicts.append({'instance_id': iid, 'path': path})
    if repaired_paths and not production_paths:
        test_only_tasks.append({'instance_id': iid, 'paths': repaired_paths})
    for path in filter(None, paths):
        if not forbidden_patch_path(path):
            continue
        candidate = PurePosixPath(path)
        allowed = False
        if candidate.name == 'conftest.py' and not forbidden_patch_path(str(candidate.parent / 'fixture.py')):
            before = subprocess.check_output(['git', 'show', ref+':'+path]).decode()
            after = subprocess.check_output(['git', 'show', ref+'~2:'+path]).decode()
            allowed = fixture_patch_allowed(before, after)
        protected.append({'instance_id': iid, 'path': path, 'supported': allowed})
print(json.dumps({'failures': results, 'protected_reference_paths': protected,
                  'inline_test_conflicts': inline_conflicts, 'source_recoveries': source_recoveries,
                  'embedded_test_repairs': test_repairs, 'test_only_tasks': test_only_tasks}))
"""
    )
    box = client.containers.run(
        image,
        entrypoint=[],
        command=["sleep", "infinity"],
        detach=True,
        read_only=True,
        network_mode="none",
        mem_limit="1g",
        nano_cpus=1_000_000_000,
        pids_limit=128,
        security_opt=["no-new-privileges:true"],
        labels={"agl.purpose": "full-python-branch-audit"},
    )
    try:
        result = box.exec_run(
            ["/opt/miniconda3/envs/testbed/bin/python", "-c", code, json.dumps(ids)],
            workdir="/testbed",
            user="0:0",
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        if result.exit_code:
            raise RuntimeError(f"Could not inspect task branches in {image}")
        return {"image": image, "tasks": len(ids), **json.loads(result.output)}
    finally:
        box.remove(force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    manifest, splits = load_prepared_data(args.data)
    summary = json.loads((args.audit / "summary.json").read_text())
    signature = json.loads((args.audit / "signature.json").read_text())
    assert summary["ready"] and summary["passed"] == summary["images_checked"] == len(manifest["images"])
    for name, digest in signature["data"].items():
        assert fingerprint(args.data / name) == digest, f"Changed data: {name}"
    for key, name in [
        ("grader", "full_python_agent.py"),
        ("sandbox", "sandbox.py"),
        ("checked_agent", "smith_docker_agent.py"),
        ("data_preparation", "prepare_full_python.py"),
        ("grading_reliability", "swe_reliability.py"),
        ("grading_evidence", "swe_grading_evidence.py"),
    ]:
        assert fingerprint(Path(__file__).with_name(name)) == signature[key], f"Changed source: {name}"
    client = docker.from_env(timeout=120)
    tasks = {}
    for split in ("train", "val"):
        for row in splits[split]:
            tasks.setdefault(row["image"], []).append(row["instance_id"])
    branches = []
    try:
        for image in manifest["images"]:
            record = json.loads((args.audit / image.split("@sha256:")[1][:16] / "result.json").read_text())
            assert record["image"] == image and record["passed"]
            assert record["reference"]["baseline"]["image_id"] == client.images.get(image).id
            branch = verify_branches(client, image, tasks[image])
            branches.append(branch)
            (args.audit / "all-task-branches.json").write_text(json.dumps(branches, indent=2))
            if branch["failures"]:
                raise RuntimeError(f"Unsupported task branches: {branch['failures'][:3]}")
            unsupported = [item for item in branch["protected_reference_paths"] if not item["supported"]]
            if unsupported:
                raise RuntimeError(f"Reference repairs conflict with the export policy: {unsupported[:3]}")
            if branch["inline_test_conflicts"]:
                raise RuntimeError(f"Reference repairs change embedded tests: {branch['inline_test_conflicts'][:3]}")
            if branch["test_only_tasks"]:
                raise RuntimeError(f"Tasks have no production bug after restoring tests: {branch['test_only_tasks']}")
    finally:
        client.close()
    assert sum(branch["tasks"] for branch in branches) == sum(manifest["counts"].values())
    print(
        "FULL_PYTHON_READY "
        + json.dumps(
            {
                "counts": manifest["counts"],
                "images": len(manifest["images"]),
                "steps": manifest["steps_for_four_complete_epochs"],
                "epochs": 4,
            }
        )
    )


if __name__ == "__main__":
    main()
