"""Paired, offline critic-only diagnostic using the installed veRL/CAPO workers.

No actor, rollout service, rewards, or model checkpoints are produced/modified.
The two arms differ only in value-head initialization before FSDP wrapping.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from critic_fit_data import file_hash, load_inputs, local_batch, prediction_sums, reduce_prediction_sums


def find_value_head(model):
    candidates = [(name, getattr(model, name)) for name in ("score", "classifier", "v_head")
                  if isinstance(getattr(model, name, None), torch.nn.Module)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one identifiable value head; got {[name for name, _ in candidates]}")
    name, head = candidates[0]
    if not list(head.parameters()) or sum(p.numel() for p in head.parameters()) > 1_000_000:
        raise ValueError("Unexpected value-head structure")
    return name, head


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def initialize_head(model, mode):
    name, head = find_value_head(model)
    with torch.no_grad():
        if mode == "zero":
            for parameter in head.parameters():
                parameter.zero_()
        elif mode != "default":
            raise ValueError("Unknown initialization")
    return name, head


def prepare(args):
    tensors, _metadata, split = load_inputs(args.snapshot, args.seed)
    source = json.loads(args.config.read_text())
    if source["algorithm"]["gamma"] != 1 or source["algorithm"]["lam"] != 1:
        raise ValueError("Only gamma=lambda=1 fixed binary targets supported")
    critic = source["critic"]
    if critic["ppo_epochs"] != 1 or critic["loss_agg_mode"] != "seq-mean-token-mean":
        raise ValueError("Expected the current CAPO critic recipe")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "snapshot": str(args.snapshot.resolve()), "snapshot_sha256": file_hash(args.snapshot),
        "metadata_sha256": file_hash(args.snapshot.with_suffix(".json")),
        "config_source": str(args.config.resolve()), "config_sha256": file_hash(args.config),
        "seed": args.seed, "passes": args.passes, "actor_loaded": False,
        "returns": "immutable audited gamma=lambda=1 binary returns",
        "old_values": "recomputed with each arm at the start of each fixed-data pass",
        "split": split,
        "physical_rows": len(tensors["response_mask"]),
        "calls": {name: len(split[name + "_rows"]) for name in ("fit", "check")},
        "context_max": int(tensors["attention_mask"].sum(-1).max()),
        "context_mean": float(tensors["attention_mask"].sum(-1).float().mean()),
        "action_tokens": int(tensors["response_mask"][split["fit_rows"] + split["check_rows"]].sum()),
        "source_files": {
            p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("critic_fit_data.py"))
        },
    }
    (args.output / "inputs.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "critic-config.json").write_text(json.dumps(critic, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"split", "source_files"}}, indent=2))
    print(json.dumps({key: value for key, value in split.items() if key.endswith("episodes")}, indent=2))


def run(args):
    import torch.distributed as dist
    import verl.utils.model as model_api
    from omegaconf import OmegaConf
    from verl.utils.config import omega_conf_to_dataclass

    from agentlightning.verl.capo_ppo import register_in_worker, verify_vendor

    manifest = json.loads((args.prepared / "inputs.json").read_text())
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 4 or os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise ValueError("This diagnostic reserves exactly physical GPUs 4-7")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if rank == 0:
        for source, expected in ((manifest["snapshot"], manifest["snapshot_sha256"]),
                                 (Path(manifest["snapshot"]).with_suffix(".json"), manifest["metadata_sha256"])):
            if file_hash(source) != expected:
                raise ValueError("Frozen diagnostic input changed")
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    torch.cuda.manual_seed_all(manifest["seed"])
    os.environ["AGL_CAPO_STRICT_PADDING"] = "1"
    os.environ["AGL_SWE_RELIABILITY"] = "1"
    register_in_worker()
    from verl.workers.fsdp_workers import CriticWorker

    critic = json.loads((args.prepared / "critic-config.json").read_text())
    critic["optim"]["total_training_steps"] = manifest["passes"]
    # Keep constant LR and original warmup; do not change the model/head dtype,
    # optimizer, value clipping, minibatch32 or microbatch2.
    if critic["optim"]["lr_warmup_steps_ratio"] != 0:
        raise ValueError("Expected the existing zero learning-rate warmup")
    cfg = omega_conf_to_dataclass(OmegaConf.create(critic))
    worker = CriticWorker(cfg)
    directory = args.prepared / args.arm
    if rank == 0:
        directory.mkdir(exist_ok=False)
    dist.barrier()
    init_record, head_flow = {}, {"output_squared": 0., "input_squared": 0., "backward_calls": 0}
    original_factory = model_api.load_valuehead_model

    def factory(*factory_args, **factory_kwargs):
        model = original_factory(*factory_args, **factory_kwargs)
        name, head = find_value_head(model)
        if rank == 0:
            # A deterministic probe, explicitly not a whole-weight checksum.
            probe = hashlib.sha256()
            for parameter_name, parameter in model.named_parameters():
                if not parameter_name.startswith(name + "."):
                    probe.update(parameter_name.encode())
                    probe.update(parameter.detach().reshape(-1)[:16].float().cpu().numpy().tobytes())
            init_record.update({"head": name, "backbone_probe_sha256": probe.hexdigest(),
                                "head_before": {n: tensor_hash(p) for n, p in head.named_parameters()},
                                "head_trainable": all(p.requires_grad for p in head.parameters()),
                                "backbone_trainable": all(p.requires_grad for n, p in model.named_parameters()
                                                          if not n.startswith(name + "."))})
        initialize_head(model, args.arm)
        if rank == 0:
            init_record["head_after"] = {n: tensor_hash(p) for n, p in head.named_parameters()}

        def backward_hook(module, inputs, outputs):
            head_flow["backward_calls"] += 1
            for key, values in (("input_squared", inputs), ("output_squared", outputs)):
                for value in values:
                    if isinstance(value, torch.Tensor):
                        norm = torch.linalg.vector_norm(value.detach(), dtype=torch.float32)
                        head_flow[key] += norm.square().item()

        head.register_full_backward_hook(backward_hook)
        return model

    model_api.load_valuehead_model = factory
    try:
        worker.init_model()
    finally:
        model_api.load_valuehead_model = original_factory
    if rank == 0 and not init_record:
        raise RuntimeError("Value-head initialization hook did not run")
    decoder_parameters = [(name, parameter) for name, parameter in worker.critic_module.named_parameters()
                          if ".layers." in name]
    if not decoder_parameters:
        raise RuntimeError("Cannot identify sharded decoder parameters for gradient diagnosis")
    if rank == 0:
        (directory / "initialization.json").write_text(json.dumps({
            **init_record, "arm": args.arm, "seed": manifest["seed"],
            "critic_config": critic, "vendor": verify_vendor(),
            "decoder_gradient_parameter_names": [name for name, _ in decoder_parameters],
            "diagnostic_source_sha256": file_hash(__file__),
            "data_source_sha256": file_hash(Path(__file__).with_name("critic_fit_data.py")),
        }, indent=2))
    tensors, _, observed_split = load_inputs(manifest["snapshot"], manifest["seed"])
    if observed_split != manifest["split"]:
        raise ValueError("Episode split differs from frozen manifest")
    split = manifest["split"]
    fit = local_batch(tensors, split["fit_rows"], rank, world)
    check = local_batch(tensors, split["check_rows"], rank, world)
    del tensors
    step = 0
    pass_index = 0

    def record(value):
        if rank == 0:
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
            print("CRITIC_FIT " + json.dumps(value, allow_nan=False), flush=True)

    def before_optimizer(optimizer, optimizer_args, optimizer_kwargs):
        nonlocal step
        squared = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float64)
        grad_count = 0
        for _name, parameter in decoder_parameters:
            if parameter.grad is not None:
                grad_count += 1
                squared += torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float32).double().square()
        if not grad_count:
            raise RuntimeError("Decoder parameters have no gradient tensors")
        dist.all_reduce(squared)
        flows = [None] * world
        dist.all_gather_object(flows, dict(head_flow))
        step += 1
        record({"kind": "gradient", "pass": pass_index, "optimizer_step": step,
                "decoder_layers_grad_norm_after_clipping": squared.sqrt().item(),
                "local_decoder_gradient_tensor_count": grad_count,
                "head_output_grad_l2_over_microbatches": sum(f["output_squared"] for f in flows) ** .5,
                "head_input_grad_l2_over_microbatches": sum(f["input_squared"] for f in flows) ** .5,
                "head_backward_calls": sum(f["backward_calls"] for f in flows)})
        head_flow.update(output_squared=0., input_squared=0., backward_calls=0)

    hook = worker.critic_optimizer.register_step_pre_hook(before_optimizer)

    def evaluate():
        predictions = {}
        for name, batch in (("fit", fit), ("check", check)):
            with torch.no_grad():
                value = worker.compute_values(batch).batch["values"].cpu()
            try:
                local_result = {"statistics": prediction_sums(value, batch, split["train_target_mean"])}
            except (ValueError, RuntimeError) as exc:
                local_result = {"error": str(exc), "rank": rank}
            parts = [None] * world
            dist.all_gather_object(parts, local_result)
            errors = [part for part in parts if "error" in part]
            if errors:
                raise RuntimeError(f"Collective prediction audit failed: {errors}")
            record({"kind": "evaluation", "pass": pass_index, "optimizer_steps": step,
                    "split": name, "statistics": reduce_prediction_sums([part["statistics"] for part in parts])})
            predictions[name] = value
        return predictions["fit"]

    started = time.monotonic()
    try:
        old_values = evaluate()
        for pass_index in range(1, manifest["passes"] + 1):
            fit.batch["values"] = old_values
            result = worker.update_critic(fit)
            # Original CAPO accumulates vf_loss across minibatches. Preserve the
            # raw worker record; comparison uses the independently measured MSE.
            if rank == 0:
                (directory / f"worker-pass-{pass_index}.json").write_text(json.dumps(result.meta_info, default=str))
            old_values = evaluate()
        record({"kind": "complete", "passes": pass_index, "optimizer_steps": step,
                "seconds": time.monotonic() - started, "actor_loaded": False})
        if rank == 0:
            (directory / "completed.json").write_text(json.dumps({"passes": pass_index, "optimizer_steps": step}))
    finally:
        hook.remove()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--snapshot", type=Path, required=True)
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=1729)
    prep.add_argument("--passes", type=int, default=4)
    arm = sub.add_parser("run")
    arm.add_argument("--prepared", type=Path, required=True)
    arm.add_argument("--arm", choices=("default", "zero"), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if not 1 <= args.passes <= 8:
            raise ValueError("Bound the diagnostic to 1-8 fixed-data passes")
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
