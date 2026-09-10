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
    if cfg["algorithm"]["adv_estimator"] != "gae" or not cfg["critic"]["enable"]:
        raise RuntimeError("Not a PPO run with an enabled critic")
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [row for row in metrics if "actor/grad_norm" in row["data"]]
    if not steps:
        raise RuntimeError("No actor updates")
    summary = []
    for row in steps:
        step, data = row["step"], row["data"]
        for key in ("actor/grad_norm", "critic/grad_norm"):
            if not math.isfinite(data[key]) or data[key] <= 0:
                raise RuntimeError(f"No finite nonzero {key} at step {step}")
        if data["training/n_sample_collected"] != data["training/n_sample_trained"]:
            raise RuntimeError("Training dropped calls")
        audit_dir = Path(cfg["agentlightning"]["multi_turn_ppo"]["audit_dir"])
        tensors = torch.load(audit_dir / f"step-{step:06d}.pt", map_location="cpu", weights_only=True)
        meta = json.loads((audit_dir / f"step-{step:06d}.json").read_text())
        mask = tensors["response_mask"].bool()
        for key in ("values", "raw_advantages", "advantages", "returns", "old_log_probs"):
            if not torch.isfinite(tensors[key][mask]).all():
                raise RuntimeError(f"Nonfinite {key}")
        groups = defaultdict(list)
        for i, rid in enumerate(meta["rollout_id_list"]):
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
                "calls": len(mask),
                "action_tokens": int(mask.sum()),
                "reward_sum": float(tensors["token_level_scores"].sum()),
                "actor_grad_norm": data["actor/grad_norm"],
                "critic_grad_norm": data["critic/grad_norm"],
            }
        )
    checkpoint = run / "checkpoints" / f"global_step_{steps[-1]['step']}"
    optimizer_steps = {}
    for role in ("actor", "critic"):
        assert (checkpoint / role / "model_world_size_1_rank_0.pt").is_file()
        assert (checkpoint / role / "extra_state_world_size_1_rank_0.pt").is_file()
        state = torch.load(
            checkpoint / role / "optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True
        )
        updates = sorted({int(value["step"]) for value in state["state"].values() if "step" in value})
        assert updates and min(updates) > 0
        optimizer_steps[role] = updates
        del state
    assert not list((run / "traces").glob("*.failed.json"))
    result = {"run": str(run), "steps": summary, "optimizer_steps": optimizer_steps, "status": "verified"}
    (run / "verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
