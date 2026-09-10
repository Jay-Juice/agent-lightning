# Copyright (c) Microsoft. All rights reserved.

"""Copy the user's CAPO PPO snapshot, retaining licenses and explicit import edits."""

import argparse
import ast
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    target = Path(__file__).resolve().parents[1] / "agentlightning/verl/vendor/capo"
    target.mkdir(parents=True, exist_ok=True)
    mappings = {
        "arft_core_algos.py": ("arft/core_algos.py", {}),
        "verl_core_algos.py": ("verl/verl/trainer/ppo/core_algos.py", {}),
        "dp_actor.py": (
            "verl/verl/workers/actor/dp_actor.py",
            {"from verl.trainer.ppo.core_algos import": "from .verl_core_algos import"},
        ),
        "dp_critic.py": (
            "verl/verl/workers/critic/dp_critic.py",
            {"from verl.trainer.ppo import core_algos": "from . import verl_core_algos as core_algos"},
        ),
        "reference_run_ppo.sh": ("examples/alfworld/run_ppo.sh", {}),
        "LICENSE": ("verl/LICENSE", {}),
    }
    files = {}
    for dest, (origin, replacements) in mappings.items():
        original = (source / origin).read_text(encoding="utf-8")
        copied = original
        for before, after in replacements.items():
            assert copied.count(before) == 1
            copied = copied.replace(before, after)
        notice = "# Agent Lightning modification: redirect PPO imports to the vendored CAPO snapshot.\n"
        if replacements:
            copied = notice + copied
        (target / dest).write_text(copied, encoding="utf-8", newline="\n")
        files[dest] = {
            "source": origin,
            "source_sha256_lf": sha(original),
            "copied_sha256_lf": sha(copied),
            "replacements": replacements,
            "added_notice": notice.strip() if replacements else None,
        }
    origin = "arft/ray_agent_trainer.py"
    original = (source / origin).read_text(encoding="utf-8")
    wanted = {
        "get_valid_data",
        "_agent_adv_estimator_key",
        "_critic_vf_loss_response_mask",
        "_scatter_step_gae_diagnostics",
        "compute_advantage",
        "_pad_dataproto_to_world_size",
    }
    segments = []
    for node in ast.walk(ast.parse(original)):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            segment = textwrap.dedent("\n".join(original.splitlines()[node.lineno - 1 : node.end_lineno]))
            segments.append((node.lineno, segment))
    assert len(segments) == len(wanted)
    preamble = (
        original.split('"""', 1)[0]
        + '''"""Selected CAPO trainer functions; only local imports are redirected."""

import math
from functools import reduce
from typing import Optional

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.ray_trainer import compute_response_mask

from .verl_core_algos import AdvantageEstimator

'''
    )
    copied = preamble + "\n\n\n".join(segment for _, segment in sorted(segments)) + "\n"
    copied = copied.replace("from arft.core_algos import", "from .arft_core_algos import")
    (target / "trajectory.py").write_text(copied, encoding="utf-8", newline="\n")
    files["trajectory.py"] = {
        "source": origin,
        "source_sha256_lf": sha(original),
        "copied_sha256_lf": sha(copied),
        "selected_functions": sorted(wanted),
        "replacements": {"from arft.core_algos import": "from .arft_core_algos import"},
    }
    (target / "__init__.py").write_text("# Vendored CAPO PPO; see LICENSE and manifest.json.\n", encoding="utf-8")
    manifest = {
        "source_repository": "https://github.com/Jay-Juice/Agent-R1",
        "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "scope": "CAPO repository token PPO baseline, not the CAPO action-ratio algorithm",
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"target": str(target), "files": list(files), "commit": manifest["source_commit"]}))


if __name__ == "__main__":
    main()
