# Copyright (c) Microsoft. All rights reserved.

"""Verify completed PPO updates, complete temporal audits and both optimizers."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def verify(run):
    if (run / "run.exit").read_text().strip() != "0":
        raise RuntimeError("Training did not exit successfully")
    cfg = json.loads((run / "resolved-config.json").read_text())
    capo = cfg["agentlightning"]["multi_turn_ppo"].get("backend", "agl") == "capo"
    if cfg["algorithm"]["adv_estimator"] != ("token_gae" if capo else "gae") or not cfg["critic"]["enable"]:
        raise RuntimeError("Not a PPO run with an enabled critic")
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [row for row in metrics if "critic/grad_norm" in row["data"]]
    if not any("actor/grad_norm" in row["data"] for row in steps):
        raise RuntimeError("No actor updates")
    summary = []
    for row in steps:
        step, data = row["step"], row["data"]
        actor_updated = step >= cfg["trainer"].get("critic_warmup", 0)
        if not actor_updated and "actor/grad_norm" in data:
            raise RuntimeError("Actor updated during critic warmup")
        for key in ("actor/grad_norm", "critic/grad_norm") if actor_updated else ("critic/grad_norm",):
            if not math.isfinite(data[key]) or data[key] <= 0:
                raise RuntimeError(f"No finite nonzero {key} at step {step}")
        if data["training/n_sample_collected"] != data["training/n_sample_trained"]:
            raise RuntimeError("Training dropped calls")
        audit_dir = Path(cfg["agentlightning"]["multi_turn_ppo"]["audit_dir"])
        tensors = torch.load(audit_dir / f"step-{step:06d}.pt", map_location="cpu", weights_only=True)
        meta = json.loads((audit_dir / f"step-{step:06d}.json").read_text())
        mask = tensors["response_mask"].bool()
        padded = meta.get("is_pad", [False] * len(mask))
        if capo:
            import numpy as np
            from verl import DataProto

            from agentlightning.verl.vendor.capo.trajectory import compute_advantage

            check = DataProto.from_dict(
                tensors={k: tensors[k] for k in ("response_mask", "values", "token_level_rewards")},
                non_tensors={
                    "is_pad": np.array(padded),
                    "trajectory_uids": np.array(meta["rollout_id_list"]),
                    "step_indices": np.array(meta["turn_index_list"]),
                },
            )
            check = compute_advantage(check, "token_gae", gamma=cfg["algorithm"]["gamma"], lam=cfg["algorithm"]["lam"])
            torch.testing.assert_close(check.batch["advantages"], tensors["advantages"])
            torch.testing.assert_close(check.batch["returns"], tensors["returns"])
            assert tensors["advantages"][padded].count_nonzero() == 0
            assert tensors["returns"][padded].count_nonzero() == 0
            mask[padded] = False
        for key in ("values", "raw_advantages", "advantages", "returns", "old_log_probs"):
            if not torch.isfinite(tensors[key][mask]).all():
                raise RuntimeError(f"Nonfinite {key}")
        if not cfg["agentlightning"]["multi_turn_ppo"]["whiten_advantages"]:
            assert torch.equal(tensors["advantages"][mask], tensors["raw_advantages"][mask])
        if cfg["actor_rollout_ref"]["actor"]["use_kl_loss"]:
            assert torch.isfinite(tensors["ref_log_prob"][mask]).all()
        groups = defaultdict(list)
        for i, rid in enumerate(meta["rollout_id_list"]):
            if not padded[i]:
                groups[rid].append(i)
        for indices in groups.values():
            indices.sort(key=lambda i: meta["turn_index_list"][i])
            assert [meta["turn_index_list"][i] for i in indices] == list(range(len(indices)))
            assert all(meta["episode_turn_count"][i] == len(indices) for i in indices)
            assert [meta["episode_terminal"][i] for i in indices] == [False] * (len(indices) - 1) + [True]
            for i in indices[:-1]:
                assert tensors["token_level_scores"][i].count_nonzero() == 0
            # At gamma=lambda=1 and no reward KL, all raw returns equal terminal reward.
            if cfg["algorithm"]["gamma"] == cfg["algorithm"]["lam"] == 1 and not cfg["algorithm"]["use_kl_in_reward"]:
                reward = tensors["token_level_scores"][indices].sum()
                for i in indices:
                    assert torch.allclose(tensors["returns"][i][mask[i]], reward.expand(int(mask[i].sum())), atol=1e-5)
        summary.append(
            {
                "step": step,
                "episodes": len(groups),
                "calls": len(mask) - sum(padded),
                "padding_rows": sum(padded),
                "action_tokens": int(mask.sum()),
                "reward_sum": float((tensors["token_level_scores"] * mask).sum()),
                "actor_updated": actor_updated,
                "actor_grad_norm": data.get("actor/grad_norm"),
                "critic_grad_norm": data["critic/grad_norm"],
            }
        )
    checkpoint = run / "checkpoints" / f"global_step_{steps[-1]['step']}"
    optimizer_steps = {}
    world = cfg["trainer"]["n_gpus_per_node"] * cfg["trainer"]["nnodes"]
    for role in ("actor", "critic"):
        rank_steps = []
        for rank in range(world):
            suffix = f"world_size_{world}_rank_{rank}.pt"
            assert (checkpoint / role / f"model_{suffix}").is_file()
            assert (checkpoint / role / f"extra_state_{suffix}").is_file()
            state = torch.load(checkpoint / role / f"optim_{suffix}", map_location="cpu", weights_only=False, mmap=True)
            updates = sorted({int(value["step"]) for value in state["state"].values() if "step" in value})
            assert updates and min(updates) > 0
            rank_steps.append(updates)
            del state
        assert all(steps == rank_steps[0] for steps in rank_steps)
        optimizer_steps[role] = rank_steps[0] if world == 1 else rank_steps
    assert not list((run / "traces").glob("*.failed.json"))
    result = {
        "run": str(run),
        "backend": "capo" if capo else "agl",
        "steps": summary,
        "optimizer_steps": optimizer_steps,
        "status": "verified",
        "verification_scope": "PPO execution, temporal tensors and checkpoints; not task quality",
        "task_success_observed": any(
            row["data"].get("training/reward", 0) > 0 or row["data"].get("val/reward", 0) > 0 for row in metrics
        ),
    }
    (run / "verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
