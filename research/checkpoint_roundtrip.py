"""Real four-rank veRL worker checkpoint roundtrip, not a full trainer resume.

One immutable 32-call batch makes Adam state nonempty. Save step 1; branch to
step 2; corrupt live state; load step 1 and replay step 2. All local parameter
and optimizer tensors, scheduler, RNG, and predictions must match exactly.
Actor and Critic run in separate torchrun processes and keep their checkpoints.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def logical_parameter_view(parameter):
    """FSDP shard tail padding is not model state and is not checkpointed."""
    padding = int(getattr(parameter, "_shard_numel_padded", 0))
    if padding < 0 or padding > parameter.numel():
        raise ValueError("Invalid FSDP shard padding metadata")
    if hasattr(parameter, "_shard_numel_padded") and (
        parameter.ndim != 1 or tuple(parameter.shape) != tuple(parameter._sharded_size)
        or parameter.data_ptr() != parameter._local_shard.data_ptr()
    ):
        raise ValueError("Expected a local sharded flat parameter before hashing")
    return parameter.reshape(-1)[:parameter.numel() - padding]


def fingerprint(value):
    """Hash every tensor byte without making a second full model/Adam copy."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            array = item.detach().reshape(-1).contiguous().cpu().view(torch.uint8).numpy()
            digest.update(memoryview(array))
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def prepare(args):
    import torch

    config = json.loads(args.config.read_text())
    actor = config["actor_rollout_ref"]["actor"]
    critic = config["critic"]
    model = config["actor_rollout_ref"]["model"]["path"]
    if Path(model).name != "Qwen3-4B-Instruct-2507" or critic["model"]["path"] != model:
        raise ValueError("Only the existing matching Qwen3-4B-Instruct-2507 Actor/Critic are supported")
    if not Path(model).is_dir():
        raise ValueError(f"Local model directory is absent: {model}")
    if config["agentlightning"]["multi_turn_ppo"]["backend"] != "capo":
        raise ValueError("Require the existing CAPO recipe")
    for worker in (actor, critic):
        if worker["strategy"] != "fsdp" or worker["ppo_mini_batch_size"] != 32 or worker["ppo_epochs"] != 1:
            raise ValueError("Require original FSDP, minibatch32, one-epoch recipe")
        if worker["ppo_micro_batch_size_per_gpu"] != 2:
            raise ValueError("Require original microbatch2 recipe")
        if worker["optim"].get("lr_warmup_steps_ratio", 0) != 0:
            raise ValueError("Require the existing zero LR warmup")
    tensors = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    metadata = json.loads(args.snapshot.with_suffix(".json").read_text())
    needed = {
        "input_ids", "attention_mask", "position_ids", "responses", "response_mask", "returns", "advantages"
    }
    if not needed.issubset(tensors):
        raise ValueError(f"Snapshot lacks {needed - tensors.keys()}")
    mask = tensors["response_mask"].bool()
    groups = {0: [], 1: []}
    for row, padded in enumerate(metadata["is_pad"]):
        if padded or not mask[row].any():
            continue
        targets = tensors["returns"][row][mask[row]]
        label = round(float(targets[0]))
        if label not in groups or not torch.isfinite(targets).all() or not torch.allclose(
            targets, torch.full_like(targets, label), atol=1e-6, rtol=0
        ):
            raise ValueError("Require the audited gamma=lambda=1 binary return snapshot")
        groups[label].append(row)
    for group in groups.values():
        group.sort(key=lambda i: (int(tensors["attention_mask"][i].sum()), i))
        if len(group) < 16:
            raise ValueError("Need at least 16 real calls for each terminal-return label")
    # Interleave labels before sharding: every rank gets four success and four
    # failure calls, preventing a zero-head check from seeing only zero targets.
    rows = [row for pair in zip(groups[0][:16], groups[1][:16], strict=True) for row in pair]
    if len(rows) != 32 or not torch.isfinite(tensors["advantages"][rows][mask[rows]]).all():
        raise ValueError("Need 32 real calls with finite advantages")
    if not tensors["advantages"][rows][mask[rows]].count_nonzero():
        raise ValueError("Selected actor advantages are all zero")
    if args.output.exists():
        raise ValueError("Output must be a fresh directory")
    parent = args.output.parent
    if not parent.is_dir() or shutil.disk_usage(parent).free < 112 * 1024**3:
        raise ValueError("Need existing output parent with >=112 GiB free for one complete checkpoint pair")
    args.output.mkdir()
    manifest = {
        "scope": "worker roundtrip only; no online rollout, trainer/dataloader resume, or learning claim",
        "world_size": 4, "physical_gpus": "4,5,6,7", "seed": args.seed,
        "snapshot": str(args.snapshot.resolve()), "snapshot_sha256": file_hash(args.snapshot),
        "metadata_sha256": file_hash(args.snapshot.with_suffix(".json")),
        "source_config": str(args.config.resolve()), "source_config_sha256": file_hash(args.config),
        "selected_rows": rows,
        "selection": "16 shortest real calls per binary-return label; interleaved; fixed in both replay branches",
        "selected_return_labels": [0, 1] * 16,
        "worker_minibatch": 32,
        "minibatch_scope": "one worker optimizer step; not the subsequent online candidate minibatch128",
        "fields": sorted(tensors), "critic_head_initialization": "zero",
        "checkpoint_estimate_gib": 91.39, "initial_free_gib": shutil.disk_usage(parent).free / 1024**3,
        "script_sha256": file_hash(__file__),
        "diagnostic_determinism": "torch deterministic algorithms and FLASH_ATTENTION_DETERMINISTIC=1",
        "target_semantics": "snapshot advantages/returns; refreshed old log-probs/values each update",
        "actor_reference": "original Actor predictions on this fixed batch, retained across both updates",
    }
    write_json(args.output / "inputs.json", manifest)
    write_json(args.output / "source-config.json", config)
    print("ROUNDTRIP_PREPARED " + json.dumps(manifest), flush=True)


def run(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.utils.config import omega_conf_to_dataclass

    from agentlightning.verl.capo_ppo import register_in_worker, verify_vendor

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 4 or os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise ValueError("Exactly four ranks on physical GPUs 4-7 required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if os.environ.get("FLASH_ATTENTION_DETERMINISTIC") != "1":
        raise ValueError("Exact replay diagnostic requires deterministic FlashAttention backward")
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.prepared / "inputs.json").read_text())
    for path, expected in (
        (manifest["snapshot"], manifest["snapshot_sha256"]),
        (Path(manifest["snapshot"]).with_suffix(".json"), manifest["metadata_sha256"]),
        (manifest["source_config"], manifest["source_config_sha256"]),
        (__file__, manifest["script_sha256"]),
    ):
        if file_hash(path) != expected:
            raise ValueError(f"Prepared input/source changed: {path}")
    random.seed(manifest["seed"] + rank)
    np.random.seed(manifest["seed"] + rank)
    torch.manual_seed(manifest["seed"] + rank)
    torch.cuda.manual_seed_all(manifest["seed"] + rank)
    os.environ["AGL_CAPO_STRICT_PADDING"] = "1"
    os.environ["AGL_SWE_RELIABILITY"] = "1"
    os.environ["AGL_CRITIC_HEAD_INIT"] = "zero" if args.role == "critic" else "default"
    register_in_worker()
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

    config = json.loads((args.prepared / "source-config.json").read_text())
    role_config = config["actor_rollout_ref"]["actor"] if args.role == "actor" else config["critic"]
    role_config["optim"]["total_training_steps"] = 4
    role_config["checkpoint"].update(
        save_contents=["model", "optimizer", "extra"],
        load_contents=["model", "optimizer", "extra"], async_save=False,
    )
    directory = args.prepared / args.role
    if directory.exists():
        raise ValueError(f"Refuse to overwrite a previous roundtrip: {directory}")
    if args.role == "actor":
        worker = ActorRolloutRefWorker(OmegaConf.create(config["actor_rollout_ref"]), role="actor")
    else:
        worker = CriticWorker(omega_conf_to_dataclass(OmegaConf.create(config["critic"])))
    if rank == 0:
        directory.mkdir()
        if shutil.disk_usage(directory).free < (112 if args.role == "actor" else 66) * 1024**3:
            raise ValueError("Insufficient remaining checkpoint disk reserve")
    dist.barrier()
    worker.init_model()
    module = worker.actor_module_fsdp if args.role == "actor" else worker.critic_module
    optimizer = worker.actor_optimizer if args.role == "actor" else worker.critic_optimizer
    scheduler = worker.actor_lr_scheduler if args.role == "actor" else worker.critic_lr_scheduler
    update = worker.update_actor if args.role == "actor" else worker.update_critic
    manager = worker.checkpoint_manager
    tensors = torch.load(manifest["snapshot"], map_location="cpu", weights_only=True)
    rows = manifest["selected_rows"]
    local_rows = rows[rank * 8:(rank + 1) * 8]
    keys = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask", "returns", "advantages")
    batch = DataProto.from_dict(
        tensors={key: tensors[key][local_rows].clone() for key in keys},
        non_tensors={"is_pad": np.zeros(8, dtype=bool)},
    )
    batch.meta_info.update(
        global_token_num=tensors["attention_mask"][rows].sum(-1).tolist(),
        temperature=config["actor_rollout_ref"]["rollout"]["temperature"],
    )
    del tensors

    def predict():
        with torch.no_grad():
            output = worker.compute_log_prob(batch) if args.role == "actor" else worker.compute_values(batch)
        key = "old_log_probs" if args.role == "actor" else "values"
        result = output.batch[key].detach().cpu().clone()
        if not torch.isfinite(result[batch.batch["response_mask"].bool()]).all():
            raise ValueError("Nonfinite prediction")
        return result

    def refresh_and_update():
        batch.batch["old_log_probs" if args.role == "actor" else "values"] = predict()
        result = update(batch)
        return result.meta_info

    def state():
        return {
            "parameters": fingerprint({name: logical_parameter_view(p) for name, p in module.named_parameters()}),
            "buffers": fingerprint(dict(module.named_buffers())),
            "optimizer": fingerprint(optimizer.state_dict()),
            "scheduler": fingerprint(scheduler.state_dict()),
            "rng": fingerprint(manager.get_rng_state()),
        }

    def assert_collective(condition, label):
        flags = [None] * world
        dist.all_gather_object(flags, {"rank": rank, "passed": bool(condition)})
        if not all(item["passed"] for item in flags):
            raise RuntimeError(f"{label}: {flags}")

    steps = 0

    def did_step(_optimizer, _args, _kwargs):
        nonlocal steps
        steps += 1

    hook = optimizer.register_step_post_hook(did_step)
    evidence = {"rank": rank, "world_size": world, "role": args.role, "vendor": verify_vendor(),
                "physical_gpu": 4 + int(os.environ["LOCAL_RANK"]), "config": config,
                "excluded_fsdp_tail_padding": {
                    name: int(getattr(p, "_shard_numel_padded", 0)) for name, p in module.named_parameters()
                    if getattr(p, "_shard_numel_padded", 0)
                },
                "critic_head_initialization": os.environ["AGL_CRITIC_HEAD_INIT"]}
    try:
        initial = state()
        initial_prediction = predict()
        if args.role == "actor":
            batch.batch["ref_log_prob"] = initial_prediction
        else:
            assert_collective(not bool(initial_prediction[batch.batch["response_mask"].bool()].count_nonzero()),
                              "Critic zero-head initialization must produce exact zero initial values")
        evidence["first_update_metrics"] = refresh_and_update()
        trained = state()
        nonzero_adam = any(
            torch.is_tensor(item.get("exp_avg")) and bool(item["exp_avg"].count_nonzero())
            for item in optimizer.state.values()
        )
        first_update_parts = [None] * world
        dist.all_gather_object(first_update_parts, {
            "rank": rank, "optimizer_steps": steps, "optimizer_initialized": bool(optimizer.state),
            "nonzero_adam": nonzero_adam, "parameters_changed": initial["parameters"] != trained["parameters"],
        })
        # With a zero-initialized value head, the first backward has zero
        # backbone gradients. The small head may live on only one FSDP rank.
        assert_collective(
            all(part["optimizer_steps"] == 1 and part["optimizer_initialized"] for part in first_update_parts)
            and any(part["nonzero_adam"] for part in first_update_parts)
            and any(part["parameters_changed"] for part in first_update_parts),
            "First update must create Adam state on four ranks and a real nonzero model update",
        )
        evidence["first_update_rank_checks"] = first_update_parts
        checkpoint = directory / "global_step_1"
        worker.save_checkpoint(str(checkpoint), hdfs_path=None, global_step=1, max_ckpt_to_keep=None)
        extra_path = checkpoint / f"extra_state_world_size_4_rank_{rank}.pt"
        extra = torch.load(extra_path, map_location="cpu", weights_only=False)
        saved_rng = extra["rng"]
        manager.load_rng_state(saved_rng)
        saved = state()
        saved_prediction = predict()
        if args.role == "critic":
            nonzero_values = [None] * world
            dist.all_gather_object(nonzero_values, bool(
                saved_prediction[batch.batch["response_mask"].bool()].count_nonzero()
            ))
            assert_collective(any(nonzero_values), "Trained critic values must become nonzero before save")
        manager.load_rng_state(saved_rng)
        evidence["reference_second_update_metrics"] = refresh_and_update()
        reference_next = state()
        next_prediction = predict()
        # Corrupt state beyond the second real update, so load cannot pass by
        # accidentally keeping an already-correct in-memory model or optimizer.
        with torch.no_grad():
            for parameter in module.parameters():
                logical_parameter_view(parameter).add_(0.01)
            for item in optimizer.state.values():
                for tensor in item.values():
                    if torch.is_tensor(tensor):
                        tensor.zero_()
        scheduler.last_epoch += 17
        random.random()
        np.random.random()
        torch.rand(8)
        torch.rand(8, device=torch.cuda.current_device())
        corrupted = state()
        assert_collective(corrupted["parameters"] != saved["parameters"] and
                          corrupted["optimizer"] != saved["optimizer"] and corrupted["rng"] != saved["rng"],
                          "Intentional corruption must be observable")
        worker.load_checkpoint(str(checkpoint), hdfs_path=None, del_local_after_load=False)
        restored = state()
        restored_prediction = predict()
        evidence.update(initial=initial, saved=saved, corrupted=corrupted, restored=restored,
                        restored_state_exact=(saved == restored),
                        restored_prediction_exact=torch.equal(saved_prediction, restored_prediction))
        write_json(directory / f"rank-{rank}.json", evidence)
        assert_collective(saved == restored, "Restored parameters/optimizer/scheduler/RNG must exactly match")
        assert_collective(torch.equal(saved_prediction, restored_prediction), "Restored predictions must match")
        manager.load_rng_state(saved_rng)
        evidence["replayed_second_update_metrics"] = refresh_and_update()
        replay_next = state()
        replay_prediction = predict()
        evidence.update(reference_next=reference_next, replay_next=replay_next,
                        replay_state_exact=(reference_next == replay_next),
                        replay_prediction_exact=torch.equal(next_prediction, replay_prediction),
                        actual_optimizer_steps=steps,
                        checkpoint_files={p.name: p.stat().st_size for p in checkpoint.glob(f"*rank_{rank}.pt")})
        write_json(directory / f"rank-{rank}.json", evidence)
        assert_collective(reference_next == replay_next, "Next update parameters/Adam/scheduler/RNG must match exactly")
        assert_collective(torch.equal(next_prediction, replay_prediction), "Next update predictions must match exactly")
        assert_collective(steps == 3 and len(evidence["checkpoint_files"]) == 3,
                          "Expected three actual optimizer steps and model/optim/extra files per rank")
        dist.barrier()
        if rank == 0:
            write_json(directory / "completed.json", {"role": args.role, "world_size": world, "passed": True})
        print(f"ROUNDTRIP_ROLE_OK role={args.role} rank={rank}", flush=True)
    finally:
        hook.remove()
        dist.destroy_process_group()


def summarize(args):
    results = {}
    if args.actor_proof:
        original = json.loads((args.actor_proof.parent / "inputs.json").read_text())
        current = json.loads((args.prepared / "inputs.json").read_text())
        for key in ("source_config_sha256", "snapshot_sha256", "selected_rows", "seed", "world_size"):
            if original[key] != current[key]:
                raise ValueError(f"Actor proof uses different diagnostic inputs: {key}")
    for role in ("actor", "critic"):
        directory = args.actor_proof if role == "actor" and args.actor_proof else args.prepared / role
        completed = json.loads((directory / "completed.json").read_text())
        ranks = [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(4)]
        if not completed["passed"] or any(not all(record.get(key, False) for key in (
            "restored_state_exact", "restored_prediction_exact", "replay_state_exact", "replay_prediction_exact"
        )) for record in ranks):
            raise RuntimeError(f"Incomplete/failed roundtrip: {role}")
        results[role] = {"world_size": 4, "rank_ids": [record["rank"] for record in ranks],
                         "evidence_directory": str(directory.resolve()),
                         "checkpoint_bytes": sum(sum(record["checkpoint_files"].values()) for record in ranks)}
    report = {"passed": True, "roles": results,
              "scope": "Real four-rank worker model/optimizer/scheduler/RNG and next-update replay only",
              "not_validated": ["online trainer/dataloader state", "rollout server", "online learning stability"]}
    write_json(args.prepared / "completed.json", report)
    print("CHECKPOINT_ROUNDTRIP_COMPLETE " + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--snapshot", type=Path, required=True)
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=1729)
    role = commands.add_parser("run")
    role.add_argument("--prepared", type=Path, required=True)
    role.add_argument("--role", choices=("actor", "critic"), required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--prepared", type=Path, required=True)
    summary.add_argument("--actor-proof", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
