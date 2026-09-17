"""Immutable, episode-split input and sufficient statistics for critic diagnosis."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from verl import DataProto


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_episodes(tensors, metadata, seed=1729):
    mask = tensors["response_mask"].bool()
    flags = metadata.get("is_pad", [False] * len(mask))
    groups = {}
    for row, (uid, padded) in enumerate(zip(metadata["rollout_id_list"], flags, strict=True)):
        if not padded:
            groups.setdefault(uid, []).append(row)
    labels = {}
    for uid, rows in groups.items():
        rows.sort(key=lambda row: metadata["turn_index_list"][row])
        if [metadata["turn_index_list"][r] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Incomplete episode: {uid}")
        if any(metadata["episode_turn_count"][r] != len(rows) for r in rows):
            raise ValueError(f"Missing calls: {uid}")
        if [bool(metadata["episode_terminal"][r]) for r in rows] != [False] * (len(rows) - 1) + [True]:
            raise ValueError(f"Invalid terminal: {uid}")
        if not mask[rows].any(-1).all():
            raise ValueError("Empty action")
        targets = tensors["returns"][rows][mask[rows]].float()
        reward = round(targets[0].item())
        if reward not in (0, 1) or not torch.isfinite(targets).all() or not torch.allclose(
            targets, torch.full_like(targets, reward), atol=1e-6, rtol=0
        ):
            raise ValueError("Diagnostic requires the audited gamma=lambda=1 binary targets")
        labels[uid] = reward
    rng = random.Random(seed)
    holdout = []
    for reward in (0, 1):
        ids = sorted(uid for uid in groups if labels[uid] == reward)
        if len(ids) < 2:
            raise ValueError("Both labels need at least two complete episodes")
        rng.shuffle(ids)
        holdout.extend(ids[:max(1, round(len(ids) / 4))])
    held = set(holdout)
    fitted = sorted(set(groups) - held)
    fit_rows = [r for uid in fitted for r in groups[uid]]
    check_rows = [r for uid in sorted(held) for r in groups[uid]]
    # Fixed, paired call order in both arms; this is a diagnostic schedule,
    # not a silently introduced online PPO shuffle policy.
    rng.shuffle(fit_rows)
    train_mask = mask[fit_rows]
    train_mean = tensors["returns"][fit_rows][train_mask].double().mean().item()
    return {
        "seed": seed, "fit_episodes": fitted, "check_episodes": sorted(held),
        "fit_rows": fit_rows, "check_rows": check_rows, "train_target_mean": train_mean,
        "episode_labels": labels,
        "fit_positive_episodes": sum(labels[u] for u in fitted),
        "check_positive_episodes": sum(labels[u] for u in held),
    }


def load_inputs(snapshot, seed):
    tensors = torch.load(snapshot, map_location="cpu", weights_only=True)
    metadata = json.loads(Path(snapshot).with_suffix(".json").read_text())
    split = split_episodes(tensors, metadata, seed)
    return tensors, metadata, split


def local_batch(tensors, rows, rank, world):
    # Duplicate padding exists only to equalize collective participation;
    # the existing strict-padding adapter excludes it from the value loss.
    selected = list(rows)
    pads = (-len(selected)) % world
    if not selected:
        raise ValueError("Empty diagnostic split")
    selected += [selected[-1]] * pads
    flags = [False] * len(rows) + [True] * pads
    width = len(selected) // world
    start, end = rank * width, (rank + 1) * width
    keys = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask", "returns")
    batch = DataProto.from_dict(
        tensors={key: tensors[key][selected[start:end]].clone() for key in keys},
        non_tensors={"is_pad": np.asarray(flags[start:end], dtype=bool)},
    )
    batch.meta_info["global_token_num"] = tensors["attention_mask"][rows].sum(-1).tolist()
    return batch


def prediction_sums(values, batch, train_mean):
    mask = batch.batch["response_mask"].bool().clone()
    mask[batch.non_tensor_batch["is_pad"]] = False
    targets = batch.batch["returns"].double()
    values = values.double()
    if not torch.isfinite(values[mask]).all():
        raise ValueError("Nonfinite critic prediction")
    first = torch.zeros_like(mask)
    last = torch.zeros_like(mask)
    for row in mask.any(-1).nonzero().flatten().tolist():
        positions = mask[row].nonzero().flatten()
        first[row, positions[0]] = True
        last[row, positions[-1]] = True
    output = {}
    for name, selection in (("all", mask), ("success", mask & (targets > .5)),
                            ("failure", mask & (targets < .5)), ("first_call_token", first),
                            ("last_call_token", last)):
        value, target = values[selection], targets[selection]
        error = value - target
        output[name] = {
            "n": value.numel(), "v": value.sum().item(), "v2": value.square().sum().item(),
            "t": target.sum().item(), "t2": target.square().sum().item(),
            "e": error.sum().item(), "e2": error.square().sum().item(),
            "train_constant_e2": (target - train_mean).square().sum().item(),
        }
    counts = mask.sum(-1)
    real = counts > 0
    output["call_mse"] = {
        "n": int(real.sum()),
        "sum": (((values - targets).square().masked_fill(~mask, 0).sum(-1) / counts.clamp_min(1))[real]).sum().item(),
    }
    return output


def reduce_prediction_sums(parts):
    output = {}
    for group in parts[0]:
        sums = {key: sum(part[group][key] for part in parts) for key in parts[0][group]}
        n = sums["n"]
        if not n:
            output[group] = {"count": 0}
            continue
        if group == "call_mse":
            output[group] = {"count": n, "mse": sums["sum"] / n}
            continue
        variance = max(0, sums["t2"] / n - (sums["t"] / n) ** 2)
        result = {
            "count": n, "mean": sums["v"] / n,
            "std": max(0, sums["v2"] / n - (sums["v"] / n) ** 2) ** .5,
            "mse": sums["e2"] / n, "zero_constant_mse": sums["t2"] / n,
            "train_constant_mse": sums["train_constant_e2"] / n,
            "split_oracle_constant_mse": variance, "return_variance": variance,
        }
        if variance > 0:
            result["explained_variance"] = 1 - (sums["e2"] / n - (sums["e"] / n) ** 2) / variance
        output[group] = result
    return output
