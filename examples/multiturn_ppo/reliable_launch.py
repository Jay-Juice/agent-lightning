"""CPU-only preflight and explicit SWE reliability configuration checks."""

import argparse
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

# Measured full Actor + Critic checkpoint; retain two while writing the third.
CHECKPOINT_GIB = 91.39
CHECKPOINT_KEEP = 2
SAFETY_GIB = 20
PEAK_FREE_GIB = math.ceil((CHECKPOINT_KEEP + 1) * CHECKPOINT_GIB + SAFETY_GIB)


def check_free_space(path: Path, *, minimum_gib: int = PEAK_FREE_GIB) -> int:
    if minimum_gib < PEAK_FREE_GIB:
        raise ValueError(f"Reliability run requires at least {PEAK_FREE_GIB} GiB initial free space")
    free_bytes = shutil.disk_usage(path).free
    if free_bytes < minimum_gib * 1024**3:
        raise ValueError(
            f"Insufficient free space: {free_bytes / 1024**3:.2f} GiB; require {minimum_gib} GiB "
            "for two retained checkpoints, a third being written, and safety reserve. No files were deleted."
        )
    return free_bytes


def validate_reliability_config(config: Mapping, env: Mapping[str, str] | None = None) -> None:
    """Reject split agent/worker/trainer protocols before creating Ray workers."""
    env = os.environ if env is None else env
    agl = config["agentlightning"]
    enabled = bool(agl.get("reliability", {}).get("enabled", False))
    env_map = agl["local"].get("env_map", {})
    for name in ("SMITH_RELIABILITY", "AGL_SWE_RELIABILITY"):
        if (env.get(name, "0") == "1") != enabled:
            raise ValueError(f"{name} and agentlightning.reliability.enabled must agree")
        if enabled and env_map.get(name) != "1":
            raise ValueError(f"local.env_map.{name} must explicitly forward the string '1'")
        if not enabled and str(env_map.get(name)) == "1":
            raise ValueError(f"local.env_map.{name} enables reliability but the trainer does not")
    if not enabled:
        return
    if agl.get("async_rollout", {}).get("enabled", False):
        raise ValueError("Reliability requires synchronous episode collection (async_rollout.enabled=false)")
    multi_turn = agl.get("multi_turn_ppo", {})
    if (
        not multi_turn.get("enabled", False)
        or multi_turn.get("backend", "agl") != "capo"
        or not multi_turn.get("capo_strict_padding", False)
    ):
        raise ValueError("Reliability requires enabled CAPO multi_turn_ppo and capo_strict_padding=true")
    actor_ref = config.get("actor_rollout_ref", {})
    rollout = actor_ref.get("rollout", {})
    # rollout.mode=async is the vLLM transport and remains supported. The
    # independent agentlightning.async_rollout switch controls collection.
    if rollout.get("n", 1) != 1 or rollout.get("val_kwargs", {}).get("n", 1) != 1:
        raise ValueError("Reliability requires rollout.n=1 and rollout.val_kwargs.n=1")
    trainer = config["trainer"]
    # Defaults verified against the installed veRL 0.7.1 ppo_trainer.yaml and
    # actor/critic checkpoint configs: local saves, no deletion on load, sync
    # checkpoint writes, model/optimizer/extra, load_contents=${.save_contents}.
    if trainer.get("default_hdfs_dir") is not None:
        raise ValueError("Reliability requires trainer.default_hdfs_dir=null")
    if trainer.get("del_local_ckpt_after_load", False):
        raise ValueError("Reliability requires trainer.del_local_ckpt_after_load=false to preserve recovery sources")
    required_contents = {"model", "optimizer", "extra"}
    for role, worker in (("actor", actor_ref.get("actor", {})), ("critic", config.get("critic", {}))):
        checkpoint = worker.get("checkpoint", {})
        if checkpoint.get("async_save", False):
            raise ValueError(f"Reliability requires {role}.checkpoint.async_save=false")
        save_contents = checkpoint.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint.get("load_contents", save_contents)
        for field, contents in (("save_contents", save_contents), ("load_contents", load_contents)):
            if not isinstance(contents, (list, tuple)) or not required_contents.issubset(contents):
                raise ValueError(f"Reliability requires {role}.checkpoint.{field} to include model, optimizer, extra")
    timeout = float(agl["rollout_timeout_seconds"])
    if env_map.get("AGL_ROLLOUT_TIMEOUT_SECONDS") != str(env.get("AGL_ROLLOUT_TIMEOUT_SECONDS")):
        raise ValueError("Hard deadline must be explicitly forwarded through local.env_map")
    if timeout != float(env["AGL_ROLLOUT_TIMEOUT_SECONDS"]):
        raise ValueError("Agent and controller hard deadlines disagree")
    for name in ("SMITH_AGENT_WALL_TIMEOUT", "SMITH_EVAL_TIMEOUT"):
        if env_map.get(name) != env.get(name):
            raise ValueError(f"local.env_map.{name} must match the launcher budget")
    needed = float(env["SMITH_AGENT_WALL_TIMEOUT"]) + 2 * (float(env["SMITH_EVAL_TIMEOUT"]) + 120) + 300
    if timeout < needed:
        raise ValueError(f"Hard deadline must be >= {needed:g}s to include one fixed-patch grading retry")
    if trainer["max_actor_ckpt_to_keep"] != 2 or trainer["max_critic_ckpt_to_keep"] != 2:
        raise ValueError("Reliable full run requires two retained Actor and Critic checkpoints")
    if not trainer["val_before_train"] or trainer["test_freq"] != 20:
        raise ValueError("Reliable full run requires initial validation and validation every 20 steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--minimum-gib", type=int, default=PEAK_FREE_GIB)
    args = parser.parse_args()
    free = check_free_space(args.path, minimum_gib=args.minimum_gib)
    print(f"RELIABILITY_DISK_OK free={free / 1024**3:.2f} GiB required={args.minimum_gib} GiB")


if __name__ == "__main__":
    main()
