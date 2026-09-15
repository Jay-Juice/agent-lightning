# Copyright (c) Microsoft. All rights reserved.

"""Diagnose the pinned Paramiko image's exact-node collection on CPU."""

import argparse
import json
from pathlib import Path

import docker
import smith_docker_agent as pilot
from full_python_agent import FullPythonSandbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = next(
        json.loads(line)
        for line in args.data.read_text().splitlines()
        if json.loads(line)["instance_id"] == args.instance
    )
    args.output.mkdir(parents=True, exist_ok=False)
    client = docker.from_env(timeout=200)
    box = FullPythonSandbox(client, pilot.agent_task(row), args.output.name)
    try:
        box.prepare()
        f2p, p2p = pilot.test_nodes(row, max_tests=None)
        nodes = list(dict.fromkeys(f2p + p2p))
        files = sorted({node.split("::", 1)[0] for node in nodes})
        patch = box.git("diff", "HEAD~1", "HEAD~2", "--binary")
        box.copy_bytes(patch.encode())
        box.git("apply", "/root/candidate.patch")
        box.git("checkout", "HEAD~1", "--", *files)
        env = {
            "HOME": "/tmp/agl-home",
            "PYTHONPATH": "/testbed",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "",
            "PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin:/bin",
        }
        cases = {
            "original": (2, nodes),
            "serial": (0, nodes),
            "grouped": (2, sorted(nodes, key=lambda n: n.split("::", 1)[0])),
            "files": (0, files),
        }
        results = {}
        for name, (workers, selected) in cases.items():
            cmd = [
                "/usr/bin/timeout",
                "--kill-after=5",
                "120",
                "/opt/miniconda3/envs/testbed/bin/python",
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "--color=no",
                "-o",
                "cache_dir=/tmp/agl-pytest-cache",
                "-p",
                "no:snail",
                "-n",
                str(workers),
                *selected,
            ]
            result = box.container.exec_run(cmd, user="65534:65534", workdir="/testbed", environment=env)
            output = result.output.decode(errors="replace")
            (args.output / (name + ".txt")).write_text(output)
            results[name] = {"exit": result.exit_code, "tail": output[-1500:]}
            print(name, json.dumps(results[name]), flush=True)
        result = box.container.exec_run(
            [
                "/opt/miniconda3/envs/testbed/bin/python",
                "-c",
                "import inspect,pytest_relaxed; print(inspect.getsource(pytest_relaxed))",
            ],
            workdir="/testbed",
        )
        (args.output / "plugin.txt").write_bytes(result.output)
        (args.output / "result.json").write_text(json.dumps(results, indent=2))
    finally:
        box.close()
        client.close()


if __name__ == "__main__":
    main()
