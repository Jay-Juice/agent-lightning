"""Reject drift from a saved baseline recipe before starting model services."""

import hashlib
import json
from pathlib import Path


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    return result


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_matched_recipe(config, environment, train_file, val_file):
    baseline = Path(environment["AGL_MATCH_BASELINE_RUN"])
    reference = json.loads((baseline / "resolved-config.json").read_text())
    provenance = json.loads((baseline / "provenance.json").read_text())
    expected_data = json.loads((baseline / "datasets.json").read_text())
    identity = {
        "agentlightning.agl_key",
        "agentlightning.agl_base_url",
        "agentlightning.hooks",
        "agentlightning.multi_turn_ppo.audit_dir",
        "actor_rollout_ref.rollout.trace.experiment_name",
        "trainer.experiment_name",
        "trainer.default_local_dir",
        "trainer.resume_mode",
        "trainer.resume_from_path",
        "trainer.total_training_steps",
        "trainer.max_actor_ckpt_to_keep",
        "trainer.max_critic_ckpt_to_keep",
    }
    left, right = flatten(reference), flatten(config)
    differences = [
        key
        for key in left.keys() | right.keys()
        if key not in identity
        and not key.startswith("agentlightning.privileged_critic.")
        and left.get(key) != right.get(key)
    ]
    if differences:
        raise ValueError(f"Baseline training recipe differs: {sorted(differences)}")
    assert config["trainer"]["resume_mode"] == "disable", "Matched PI must start from original weights"
    assert config["trainer"].get("resume_from_path") is None
    steps = int(environment["AGL_PI_TRAINING_STEPS"])
    assert 1 <= steps <= 784 and config["trainer"]["total_training_steps"] == steps
    budgets = provenance["runtime_options"]["smith_budgets"]
    mismatches = [
        key
        for key, value in budgets.items()
        if not key.startswith("SMITH_PRIVILEGED_") and (value is None or environment.get(key) != value)
    ]
    if mismatches:
        raise ValueError(f"Baseline Agent environment differs or is missing: {mismatches}")
    assert int(environment["AGL_MAX_LOCAL_AGENTS"]) == provenance["runtime_options"]["local_runner_maximum_size"]
    assert environment.get("AGL_GPU_MONITOR") == "1"
    hashes = {}
    for split, path in (("train", train_file), ("validation", val_file)):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        hashes[split] = digest(rows)
        assert hashes[split] == digest(expected_data[split]), f"Baseline data differs: {split}"
    return {
        "matched": True,
        "baseline_run": str(baseline),
        "training_steps": steps,
        "data_sha256": hashes,
        "agent_environment": {key: environment.get(key) for key in budgets},
        "allowed_differences": sorted(identity),
        "privileged_critic": config["agentlightning"]["privileged_critic"],
    }
