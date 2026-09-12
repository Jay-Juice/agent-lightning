# Copyright (c) Microsoft. All rights reserved.
"""Four-rank memory stress check using a saved real PPO call and installed workers.

This performs a disposable actor update; it never saves weights or alters the
source audit/checkpoint. It is a memory check, not a task-quality experiment.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from verl import DataProto

from agentlightning.verl.capo_ppo import register_in_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-step", type=int, default=2)
    parser.add_argument("--checkpoint-step", type=int, default=1)
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--cohort", action="store_true", help="Include critic and reference in the memory check")
    parser.add_argument("--fsdp-size", type=int, choices=[-1, 2], default=-1)
    parser.add_argument("--actor-zero2", action="store_true", help="Keep actor parameters gathered through backward")
    parser.add_argument("--inference-batch", type=int, choices=[1, 2], default=1)
    parser.add_argument("--train-batch", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--fresh-model", action="store_true", help="Initialize from the original model without resuming"
    )
    parser.add_argument(
        "--resident-cohort",
        action="store_true",
        help="Keep actor, critic, reference and optimizer states on GPU for a combined memory check",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"
    assert int(os.environ["WORLD_SIZE"]) == 4
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    os.environ["AGL_CAPO_STRICT_PADDING"] = "1"
    register_in_worker()
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    config = json.loads((args.run / "resolved-config.json").read_text())
    cfg = OmegaConf.create(config["actor_rollout_ref"])
    cfg.actor.calculate_entropy = args.entropy
    cfg.actor.ppo_micro_batch_size_per_gpu = args.train_batch
    cfg.actor.fsdp_config.fsdp_size = args.fsdp_size
    cfg.ref.fsdp_config.fsdp_size = args.fsdp_size
    cfg.rollout.log_prob_micro_batch_size_per_gpu = args.inference_batch
    cfg.ref.log_prob_micro_batch_size_per_gpu = args.inference_batch
    if args.actor_zero2:
        cfg.actor.fsdp_config.reshard_after_forward = False
    if args.resident_cohort:
        cfg.actor.fsdp_config.param_offload = False
        cfg.actor.fsdp_config.optimizer_offload = False
        cfg.ref.fsdp_config.param_offload = False
    if args.fused:
        cfg.model.use_fused_kernels = True
        cfg.actor.use_fused_kernels = True
        cfg.model.fused_kernel_options.impl_backend = "torch"
    assert cfg.actor.entropy_coeff == 0
    worker = ActorRolloutRefWorker(cfg, role="actor")
    worker.init_model()
    checkpoint = args.run / "checkpoints" / f"global_step_{args.checkpoint_step}" / "actor"
    if not args.fresh_model:
        worker.load_checkpoint(str(checkpoint), del_local_after_load=False)
    # Installed veRL offloads optimizer tensors with non_blocking=True. CPU
    # assertions must wait for those copies rather than reading pending data.
    torch.cuda.synchronize()
    before_steps = sorted({int(s["step"]) for s in worker.actor_optimizer.state.values() if "step" in s})
    critic = reference = None
    if args.resident_cohort or args.cohort:
        from verl.workers.fsdp_workers import CriticWorker

        critic_cfg = OmegaConf.create(config["critic"])
        critic_cfg.ppo_micro_batch_size_per_gpu = args.train_batch
        critic_cfg.model.fsdp_config.fsdp_size = args.fsdp_size
        critic_cfg.forward_micro_batch_size_per_gpu = args.inference_batch
        if args.resident_cohort:
            critic_cfg.model.fsdp_config.param_offload = False
            critic_cfg.model.fsdp_config.optimizer_offload = False
        critic = CriticWorker(critic_cfg)
        critic.init_model()
        if not args.fresh_model:
            critic.load_checkpoint(str(checkpoint.parent / "critic"), del_local_after_load=False)
        reference_cfg = OmegaConf.create(config["actor_rollout_ref"])
        reference_cfg.actor.fsdp_config.fsdp_size = args.fsdp_size
        reference_cfg.ref.fsdp_config.fsdp_size = args.fsdp_size
        reference_cfg.ref.log_prob_micro_batch_size_per_gpu = args.inference_batch
        if args.resident_cohort:
            reference_cfg.ref.fsdp_config.param_offload = False
        reference_cfg.model.use_fused_kernels = args.fused
        reference_cfg.model.fused_kernel_options.impl_backend = "torch"
        reference = ActorRolloutRefWorker(reference_cfg, role="ref")
        reference.init_model()

    audit = args.run / "ppo-audit" / f"step-{args.audit_step:06d}"
    tensors = torch.load(audit.with_suffix(".pt"), map_location="cpu", weights_only=True, mmap=True)
    meta = json.loads(audit.with_suffix(".json").read_text())
    lengths = tensors["attention_mask"].sum(-1)
    lengths[torch.tensor(meta["is_pad"], dtype=torch.bool)] = -1
    index = int(lengths.argmax())
    length = int(lengths[index])
    train_indices = lengths.topk(args.train_batch).indices
    # Every rank receives the same longest real call: a deliberate worst-case
    # memory test. No new reward, advantage or token content is manufactured.
    data = DataProto.from_dict(
        tensors={key: value[train_indices].clone() for key, value in tensors.items()},
        non_tensors={"is_pad": np.zeros(args.train_batch, dtype=bool)},
        meta_info={
            "temperature": cfg.rollout.temperature,
            "global_token_num": lengths[train_indices].tolist() * 4,
        },
    )
    timings = {}
    if args.inference_batch > 1:
        indices = lengths.topk(args.inference_batch).indices
        pair = DataProto.from_dict(
            tensors={key: value[indices].clone() for key, value in tensors.items()},
            non_tensors={"is_pad": np.zeros(args.inference_batch, dtype=bool)},
            meta_info={"temperature": cfg.rollout.temperature, "global_token_num": lengths[indices].tolist() * 4},
        )
        mask = pair.batch["response_mask"].bool()
        checks = [("actor", worker.compute_log_prob, "old_log_probs")]
        if critic is not None:
            checks.extend(
                [
                    ("reference", reference.compute_ref_log_prob, "ref_log_prob"),
                    ("critic", critic.compute_values, "values"),
                ]
            )
        torch.cuda.reset_peak_memory_stats()
        timings["inference_batch_parity"] = {}
        for role, forward, key in checks:
            grouped = forward(pair).batch[key].float()[mask]
            singles = torch.cat([forward(pair[i : i + 1]).batch[key] for i in range(len(pair))]).float()[mask]
            torch.testing.assert_close(grouped, singles, rtol=5e-3, atol=2e-2)
            difference = (grouped - singles).abs()
            timings["inference_batch_parity"][role] = {"mae": difference.mean().item(), "max": difference.max().item()}
        timings["inference_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if rank == 0:
            print("BATCH_SINGLE_PARITY " + json.dumps(timings["inference_batch_parity"]), flush=True)
        del pair
    del tensors
    if critic is not None:
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        reference.compute_ref_log_prob(data)
        critic.compute_values(data)
        critic_result = critic.update_critic(data)
        torch.cuda.synchronize()
        timings["reference_and_critic_seconds"] = time.monotonic() - start
        norms = np.asarray(critic_result.meta_info["metrics"]["critic/grad_norm"])
        assert np.isfinite(norms).all() and (norms > 0).all()
        timings["critic_grad_norm"] = norms.tolist()
        timings["cohort_peak_before_actor_gib"] = torch.cuda.max_memory_allocated() / 2**30
    measured = worker.compute_log_prob(data).batch["old_log_probs"]
    mask = data.batch["response_mask"].bool()
    difference = (measured.float() - data.batch["old_log_probs"].float())[mask].abs()
    assert torch.isfinite(difference).all()
    probability_check = {
        "tokens": int(mask.sum()),
        "mae": difference.mean().item(),
        "max": difference.max().item(),
    }
    if rank == 0:
        print("SAVED_LOGPROB_PARITY " + json.dumps(probability_check), flush=True)
    torch.cuda.reset_peak_memory_stats()
    if rank == 0:
        print(
            f"LONG_CONTEXT_BACKWARD_START tokens={length} row={index} entropy={args.entropy}",
            flush=True,
        )
    start = time.monotonic()
    result = worker.update_actor(data)
    torch.cuda.synchronize()
    timings["actor_seconds"] = time.monotonic() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    if args.fresh_model:
        # Adam states are lazy: the first backward does not yet include their
        # persistent allocation. Repeat with initialized optimizers before
        # claiming the configuration fits a continuing training run.
        if rank == 0:
            print("STEADY_STATE_BACKWARD_START", flush=True)
        torch.cuda.reset_peak_memory_stats()
        if critic is not None:
            start = time.monotonic()
            critic_result = critic.update_critic(data)
            torch.cuda.synchronize()
            timings["steady_critic_seconds"] = time.monotonic() - start
            critic_norms = np.asarray(critic_result.meta_info["metrics"]["critic/grad_norm"])
            assert np.isfinite(critic_norms).all() and (critic_norms > 0).all()
            timings["steady_critic_grad_norm"] = critic_norms.tolist()
        start = time.monotonic()
        result = worker.update_actor(data)
        torch.cuda.synchronize()
        timings["steady_actor_seconds"] = time.monotonic() - start
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    metrics = result.meta_info["metrics"]
    norms = np.asarray(metrics["actor/grad_norm"])
    assert np.isfinite(norms).all() and (norms > 0).all()
    torch.cuda.synchronize()
    after_steps = sorted({int(s["step"]) for s in worker.actor_optimizer.state.values() if "step" in s})
    record = {
        "rank": rank,
        "tokens": length,
        "row": index,
        "rollout_id": meta["rollout_id_list"][index],
        "entropy_statistics": args.entropy,
        "fused_torch_backend": args.fused,
        "resident_cohort": args.resident_cohort,
        "cohort": args.resident_cohort or args.cohort,
        "fsdp_size": args.fsdp_size,
        "fresh_model": args.fresh_model,
        "actor_zero2": args.actor_zero2,
        "inference_batch": args.inference_batch,
        "train_batch": args.train_batch,
        "train_lengths": lengths[train_indices].tolist(),
        "timings": timings,
        "saved_logprob_parity": probability_check,
        "memory_fraction": args.memory_fraction,
        "peak_allocated_gib": peak,
        "grad_norm": norms.tolist(),
        "optimizer_before": before_steps,
        "optimizer_after": after_steps,
        "optimizer_state_count": len(worker.actor_optimizer.state),
    }
    assert math.isfinite(record["peak_allocated_gib"])
    gathered = [None] * 4
    dist.all_gather_object(gathered, record)
    if rank == 0:
        with args.output.open("x") as file:
            json.dump(
                {
                    "scope": "disposable memory stress update; no weights saved",
                    "ranks": gathered,
                },
                file,
                indent=2,
            )
        print(json.dumps(gathered), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    expected_steps = [2] if args.fresh_model else [step + 1 for step in before_steps]
    assert after_steps == expected_steps and (args.fresh_model or before_steps), (before_steps, after_steps)


if __name__ == "__main__":
    main()
