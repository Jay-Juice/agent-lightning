# Copyright (c) Microsoft. All rights reserved.

"""A800 single-machine Agent Lightning training entrypoint, with bounded defaults."""

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def build_config(args):
    with initialize_config_dir(
        config_dir=str(importlib.resources.files("agentlightning.verl")),
        version_base=None,
    ):
        base = compose(config_name="config")
    OmegaConf.set_struct(base, False)
    run = Path(os.environ["AGL_RUN_DIR"])
    # VERL's built-in file backend survives Ray stdout forwarding delays.
    os.environ["VERL_FILE_LOGGER_PATH"] = str(run / "metrics.jsonl")
    overrides = {
        "algorithm": {
            "adv_estimator": "gae",
            "use_kl_in_reward": False,
            "enable_rollout_level_advantage": False,
            "gamma": 1.0,
            "lam": 1.0,
        },
        "data": {
            "train_batch_size": 2,
            "val_batch_size": 2,
            "max_prompt_length": 1024,
            "max_response_length": 128,
            "dataloader_num_workers": 0,
        },
        "actor_rollout_ref": {
            "model": {
                "path": args.model,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "ppo_mini_batch_size": 1,
                "ppo_micro_batch_size_per_gpu": 1,
                "policy_loss": {"loss_mode": "vanilla"},
                "optim": {"lr": 1e-5, "weight_decay": 0.0},
                "use_kl_loss": False,
                "entropy_coeff": 0.0,
                "fsdp_config": {"param_offload": True, "optimizer_offload": True},
                "checkpoint": {"save_contents": ["model", "optimizer", "extra", "hf_model"]},
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": 1},
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "tensor_model_parallel_size": 1,
                "n": 1,
                "log_prob_micro_batch_size_per_gpu": 1,
                "temperature": 1.0,
                "gpu_memory_utilization": 0.30,
                "max_model_len": 2048,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 8,
                "enforce_eager": True,
                "multi_turn": {"format": "hermes"},
                "agent": {"num_workers": 1},
                "val_kwargs": {"n": 1, "temperature": 0.7, "do_sample": True},
                "checkpoint_engine": {"update_weights_bucket_megabytes": 2048},
            },
        },
        "critic": {
            "enable": True,
            "ppo_mini_batch_size": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "forward_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-5, "weight_decay": 0.0},
            "model": {
                "path": args.model,
                "tokenizer_path": args.model,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "fsdp_config": {"param_offload": True, "optimizer_offload": True},
            },
            "checkpoint": {"save_contents": ["model", "optimizer", "extra"]},
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "val_before_train": True,
            "logger": ["console", "file"],
            "project_name": "a800-agentlightning",
            "experiment_name": run.name,
            "total_epochs": 4,
            "total_training_steps": args.steps,
            "test_freq": 1,
            "save_freq": 1,
            "max_actor_ckpt_to_keep": 2,
            "max_critic_ckpt_to_keep": 2,
            "default_local_dir": str(run / "checkpoints"),
            "resume_mode": "disable",
        },
        "ray_kwargs": {
            "ray_init": {
                "address": "local",
                "num_cpus": 8,
                "num_gpus": 1,
                "include_dashboard": False,
                "object_store_memory": 2147483648,
            }
        },
        "agentlightning": {
            "agl_base_url": os.environ["AGL_BASE_URL"],
            "agl_key": os.environ["AGL_KEY"],
            "rollout_timeout_seconds": 300,
            "max_ppo_update_times": None,
            "multi_turn_ppo": {
                "enabled": True,
                "whiten_advantages": True,
                "audit_dir": str(run / "ppo-audit"),
            },
            "hooks": str(Path(__file__).with_name("trace_hooks.py")),
            "local": {
                "agent_class": "smoke_agent.SmokeAgent",
                "env_map": {"AGL_TASK": "input"},
            },
            "trace_aggregator": {"level": "transition"},
        },
    }
    cfg = OmegaConf.merge(base, OmegaConf.create(overrides), OmegaConf.from_dotlist(args.overrides))
    from agentlightning.verl.multi_turn_ppo import validate_config

    validate_config(cfg)
    pi = cfg.agentlightning.get("privileged_critic", {})
    if pi.get("enabled", False):
        expected = {
            "SMITH_PRIVILEGED_STATE": "critic",
            "SMITH_PRIVILEGED_MAX_TOKENS": str(pi.max_tokens),
            "SMITH_PRIVILEGED_SAFETY_MARGIN": str(pi.safety_margin),
        }
        mismatches = {name: os.environ.get(name) for name, value in expected.items() if os.environ.get(name) != value}
        if mismatches or pi.state_schema_version != 1:
            raise ValueError(f"Privileged Critic environment/config mismatch: {mismatches}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != len(set(visible)) or len(visible) != cfg.trainer.n_gpus_per_node:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain one unique device per configured GPU")
    # The ephemeral local auth key should not be persisted in the config artifact.
    safe = OmegaConf.to_container(cfg, resolve=True)
    safe["agentlightning"]["agl_key"] = "<runtime key>"
    (run / "resolved-config.json").write_text(json.dumps(safe, indent=2))
    from sampling_config import proxy_overrides

    (run / "proxy-overrides.txt").write_text("\n".join(proxy_overrides(safe["actor_rollout_ref"]["rollout"])) + "\n")
    repo = Path(__file__).resolve().parents[2]
    sources = [
        repo / "agentlightning/verl/multi_turn_ppo.py",
        repo / "agentlightning/verl/distributed_ppo.py",
        repo / "agentlightning/verl/capo_ppo.py",
        repo / "agentlightning/verl/capo_padding.py",
        repo / "agentlightning/verl/privileged_critic.py",
        repo / "agentlightning/verl/entrypoint.py",
        repo / "agentlightning/verl/config.yaml",
        repo / "agentlightning/verl/trainer.py",
        repo / "agentlightning/verl/full_dataset.py",
        repo / "agentlightning/verl/agl_rollout_manager.py",
        repo / "agentlightning/server/routes/events.py",
        repo / "agentlightning/server/routes/proxy.py",
        repo / "agentlightning/server/proxy.py",
        *Path(repo / "agentlightning/privileged_state").glob("*.py"),
        *Path(__file__).parent.glob("*.py"),
        *Path(__file__).parent.glob("*.sh"),
    ]
    if cfg.agentlightning.multi_turn_ppo.get("backend", "agl") == "capo":
        from agentlightning.verl.capo_ppo import verify_vendor

        verify_vendor()
        sources.extend(p for p in (repo / "agentlightning/verl/vendor/capo").iterdir() if p.is_file())
    provenance = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "verl", "vllm", "ray", "transformers", "agentlightning")
        },
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_options": {
            "local_runner_maximum_size": int(os.environ.get("AGL_MAX_LOCAL_AGENTS", "4")),
            "gpu_monitor_enabled": os.environ.get("AGL_GPU_MONITOR", "0") == "1",
            "smith_budgets": {
                name: os.environ.get(name)
                for name in (
                    "SMITH_MAX_TURNS",
                    "SMITH_MAX_TOKENS",
                    "SMITH_CONTEXT",
                    "SMITH_OBS_CHAR_CAP",
                    "SMITH_MODEL_TIMEOUT",
                    "SMITH_MAX_FORMAT_ERRORS",
                    "SMITH_GATEWAY_WAIT_S",
                    "SMITH_CMD_TIMEOUT",
                    "SMITH_EVAL_TIMEOUT",
                    "SMITH_VERIFY_SUBMISSION",
                    "SMITH_ALLOW_REPRO_FILES",
                    "SMITH_CHECK_SYNTAX",
                    "SMITH_CHECKED_EDITOR",
                    "SMITH_PRIVILEGED_STATE",
                    "SMITH_PRIVILEGED_MAX_TOKENS",
                    "SMITH_PRIVILEGED_SAFETY_MARGIN",
                )
            },
        },
    }
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--config-only", action="store_true")
    args, args.overrides = parser.parse_known_args()
    cfg = build_config(args)
    if args.config_only:
        print("CONFIG_OK")
        return
    if args.train_file:
        if not args.val_file:
            raise ValueError("A separate validation file is required")
        train = [json.loads(line) for line in args.train_file.read_text().splitlines() if line.strip()]
        val = [json.loads(line) for line in args.val_file.read_text().splitlines() if line.strip()]
        if {x["data_id"] for x in train} & {x["data_id"] for x in val}:
            raise ValueError("Training and validation overlap")
    else:
        pairs = [
            (127, 39),
            (236, 47),
            (731, 83),
            (418, 67),
            (892, 57),
            (361, 94),
            (753, 28),
            (649, 76),
        ]
        train = [
            {
                "data_source": "synthetic_arithmetic_diagnostic",
                "data_id": f"smoke-train-{i}",
                "question": f"Compute {a} * {b}.",
                "answer": a * b,
            }
            for i, (a, b) in enumerate(pairs)
        ]
        val = [
            {
                "data_source": "synthetic_arithmetic_diagnostic",
                "data_id": f"smoke-val-{i}",
                "question": f"Compute {a} * {b}.",
                "answer": a * b,
            }
            for i, (a, b) in enumerate([(143, 51), (782, 49)])
        ]
    run = Path(os.environ["AGL_RUN_DIR"])
    (run / "datasets.json").write_text(json.dumps({"train": train, "validation": val}, indent=2))
    import ray

    from agentlightning.verl.entrypoint import run_ppo

    try:
        run_ppo(cfg, train_dataset=train, val_dataset=val)
    finally:
        ray.shutdown()  # Only the Ray instance created by this driver.


if __name__ == "__main__":
    main()
