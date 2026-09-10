# Copyright (c) Microsoft. All rights reserved.

"""Compare temporal GAE against explicitly supplied, trusted reference sources.

Read reference source files already present on the execution host.
Only the token GAE function and its trajectory validator are compiled. This
diagnostic does not train, change a configuration, or import either old project.
Run inside the same veRL environment as the current training implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from verl import DataProto
from verl.utils import torch_functional as verl_F

from agentlightning.verl.multi_turn_ppo import compute_advantage


def reference_function(source: str):
    names = {"compute_token_gae_advantage_return", "_validate_trajectory_step_indices"}
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    namespace = {"np": np, "torch": torch, "verl_F": verl_F, "defaultdict": defaultdict}
    exec(compile(ast.fix_missing_locations(module), "<trusted-reference>", "exec"), namespace)
    return namespace["compute_token_gae_advantage_return"]


def batch_from(tensors, metadata):
    return DataProto.from_dict(
        tensors={k: tensors[k].clone() for k in ("values", "token_level_rewards", "response_mask")},
        non_tensors={k: np.asarray(v) for k, v in metadata.items()},
    )


def compare(tensors, metadata, references, gamma, lam):
    current = compute_advantage(batch_from(tensors, metadata), gamma=gamma, lam=lam, whiten=True)
    mask = tensors["response_mask"].bool()
    results = {}
    for name, function in references.items():
        advantage, returns = function(
            tensors["token_level_rewards"],
            tensors["values"],
            tensors["response_mask"],
            np.asarray(metadata["rollout_id_list"]),
            np.asarray(metadata["turn_index_list"]),
            gamma,
            lam,
        )
        # Reference whitening leaves nonzero padding entries; their PPO loss masks
        # these out. Compare action advantages and the complete masked returns.
        torch.testing.assert_close(current.batch["advantages"][mask], advantage[mask], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(current.batch["returns"], returns, rtol=2e-5, atol=2e-6)
        results[name] = {
            "advantage_max_abs_error": (current.batch["advantages"][mask] - advantage[mask]).abs().max().item(),
            "return_max_abs_error": (current.batch["returns"] - returns).abs().max().item(),
        }
    return results


def synthetic_cases(references):
    generator = torch.Generator().manual_seed(9137)
    metadata = {
        "rollout_id_list": ["a", "b", "b", "c", "c", "c", "c"],
        "turn_index_list": [0, 0, 1, 0, 1, 2, 3],
        "episode_turn_count": [1, 2, 2, 4, 4, 4, 4],
        "episode_terminal": [True, False, True, False, False, False, True],
    }
    permutation = [5, 1, 3, 0, 6, 2, 4]
    metadata = {k: np.asarray(v)[permutation] for k, v in metadata.items()}
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for zero_rewards in (False, True):
            mask = (torch.rand(7, 13, generator=generator) > 0.4).long()
            mask[:, 0] = 1
            rewards = torch.randn(7, 13, generator=generator) * mask
            if zero_rewards:
                rewards.zero_()
            tensors = {
                "values": torch.randn(7, 13, generator=generator).to(dtype),
                "response_mask": mask,
                "token_level_rewards": rewards,
            }
            for gamma, lam in ((1.0, 1.0), (0.99, 1.0), (0.97, 0.95)):
                results.append(
                    {
                        "dtype": str(dtype),
                        "zero_rewards": zero_rewards,
                        "gamma": gamma,
                        "lambda": lam,
                        "references": compare(tensors, metadata, references, gamma, lam),
                    }
                )
    return results


def saved_case(path, references):
    # Inputs are our own, trusted training audit files; original artifacts stay untouched.
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    metadata = json.loads(path.with_suffix(".json").read_text())
    config = json.loads((path.parent.parent / "resolved-config.json").read_text())
    gamma, lam = config["algorithm"]["gamma"], config["algorithm"]["lam"]
    whiten = config["agentlightning"]["multi_turn_ppo"]["whiten_advantages"]
    current = compute_advantage(batch_from(tensors, metadata), gamma=gamma, lam=lam, whiten=whiten)
    torch.testing.assert_close(current.batch["advantages"], tensors["advantages"])
    torch.testing.assert_close(current.batch["returns"], tensors["returns"])
    normalized = compute_advantage(batch_from(tensors, metadata), gamma=gamma, lam=lam, whiten=True)
    whitening_difference = (current.batch["advantages"] - normalized.batch["advantages"]).abs().max().item()
    return {
        "audit": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(current),
        "action_tokens": int(tensors["response_mask"].sum()),
        "gamma": gamma,
        "lambda": lam,
        "as_run_whiten": whiten,
        "as_run_reproduced": True,
        "as_run_vs_whitened_max_abs_difference": whitening_difference,
        "references_with_matching_whitening": compare(tensors, metadata, references, gamma, lam),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--audit", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(item.split("=", 1) for item in args.reference)
    sources = {name: Path(path).read_text() for name, path in paths.items()}
    references = {name: reference_function(source) for name, source in sources.items()}
    if not references:
        raise ValueError("At least one reference source is required")
    report = {
        "scope": "Token GAE numerical equivalence with matched gamma/lambda/whitening; not full PPO equivalence",
        "reference_paths": paths,
        "reference_source_sha256": {
            name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()
        },
        "synthetic_cases": synthetic_cases(references),
        "saved_cases": [saved_case(path, references) for path in args.audit],
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
