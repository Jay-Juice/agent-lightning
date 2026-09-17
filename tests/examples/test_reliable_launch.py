"""No-GPU checks for reliability launch safety and protocol wiring."""

import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "examples/multiturn_ppo/reliable_launch.py"
SPEC = importlib.util.spec_from_file_location("reliable_launch", SOURCE)
assert SPEC is not None and SPEC.loader is not None
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class ReliabilityLaunchTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            "SMITH_RELIABILITY": "1",
            "AGL_SWE_RELIABILITY": "1",
            "SMITH_AGENT_WALL_TIMEOUT": "3600",
            "SMITH_EVAL_TIMEOUT": "600",
            "AGL_ROLLOUT_TIMEOUT_SECONDS": "8400",
        }
        self.config = {
            "agentlightning": {
                "reliability": {"enabled": True},
                "multi_turn_ppo": {"enabled": True, "backend": "capo", "capo_strict_padding": True},
                "rollout_timeout_seconds": 8400,
                "local": {"env_map": dict(self.env)},
            },
            "trainer": {
                "max_actor_ckpt_to_keep": 2,
                "max_critic_ckpt_to_keep": 2,
                "val_before_train": True,
                "test_freq": 20,
            },
        }

    def test_current_181_gib_rejected_before_training(self):
        with (patch.object(launch.shutil, "disk_usage", return_value=SimpleNamespace(free=181 * 1024**3)),
              self.assertRaisesRegex(ValueError, "Insufficient free space")):
            launch.check_free_space(Path("."))

    def test_peak_space_includes_third_checkpoint(self):
        self.assertEqual(launch.PEAK_FREE_GIB, 295)
        with patch.object(launch.shutil, "disk_usage", return_value=SimpleNamespace(free=295 * 1024**3)):
            self.assertEqual(launch.check_free_space(Path(".")), 295 * 1024**3)
        with self.assertRaisesRegex(ValueError, "at least 295"):
            launch.check_free_space(Path("."), minimum_gib=182)

    def test_enabled_config_and_explicit_forwarding_agree(self):
        launch.validate_reliability_config(self.config, self.env)

    def test_worker_or_agent_protocol_mismatch_rejected(self):
        for name in ("SMITH_RELIABILITY", "AGL_SWE_RELIABILITY"):
            with self.subTest(name=name):
                self.env[name] = "0"
                with self.assertRaisesRegex(ValueError, name):
                    launch.validate_reliability_config(self.config, self.env)
                self.env[name] = "1"
                self.config["agentlightning"]["local"]["env_map"].pop(name)
                with self.assertRaisesRegex(ValueError, name):
                    launch.validate_reliability_config(self.config, self.env)
                self.config["agentlightning"]["local"]["env_map"][name] = "1"

    def test_hard_deadline_reserves_fixed_patch_retry(self):
        self.env["AGL_ROLLOUT_TIMEOUT_SECONDS"] = "5200"
        self.config["agentlightning"]["rollout_timeout_seconds"] = 5200
        self.config["agentlightning"]["local"]["env_map"]["AGL_ROLLOUT_TIMEOUT_SECONDS"] = "5200"
        with self.assertRaisesRegex(ValueError, "8220s"):
            launch.validate_reliability_config(self.config, self.env)

    def test_retention_and_validation_required(self):
        for key, value in (
            ("max_actor_ckpt_to_keep", 3),
            ("max_critic_ckpt_to_keep", 3),
            ("val_before_train", False),
            ("test_freq", 40),
        ):
            old = self.config["trainer"][key]
            self.config["trainer"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.validate_reliability_config(self.config, self.env)
            self.config["trainer"][key] = old

    def test_disabled_legacy_config_unchanged(self):
        legacy = {"agentlightning": {"local": {"env_map": {"AGL_TASK": "input"}}}}
        launch.validate_reliability_config(legacy, {})

    def test_sync_collection_capable_backend_and_single_sample_required(self):
        changes = (
            ("agentlightning", {"async_rollout": {"enabled": True}}),
            ("agentlightning", {"multi_turn_ppo": {"enabled": False, "backend": "capo"}}),
            ("agentlightning", {"multi_turn_ppo": {"enabled": True, "backend": "agl"}}),
            ("agentlightning", {"multi_turn_ppo": {"enabled": True, "backend": "capo"}}),
            ("actor_rollout_ref", {"rollout": {"n": 2}}),
            ("actor_rollout_ref", {"rollout": {"val_kwargs": {"n": 2}}}),
        )
        for section, values in changes:
            config = deepcopy(self.config)
            config.setdefault(section, {}).update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                launch.validate_reliability_config(config, self.env)

    def test_async_vllm_transport_is_not_async_collection(self):
        self.config["actor_rollout_ref"] = {"rollout": {"mode": "async"}}
        launch.validate_reliability_config(self.config, self.env)

    def test_local_checkpoint_and_recovery_source_required(self):
        for field, value in (("default_hdfs_dir", "hdfs://checkpoints"), ("del_local_ckpt_after_load", True)):
            config = deepcopy(self.config)
            config["trainer"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                launch.validate_reliability_config(config, self.env)

    def test_both_roles_require_synchronous_complete_save_and_load(self):
        for role in ("actor", "critic"):
            for field, value in (
                ("async_save", True),
                ("save_contents", ["model", "extra"]),
                ("load_contents", ["model", "optimizer"]),
                ("save_contents", None),
                ("load_contents", None),
                ("load_contents", "model optimizer extra"),
            ):
                config = deepcopy(self.config)
                section = config.setdefault("actor_rollout_ref", {}) if role == "actor" else config
                section[role] = {"checkpoint": {field: value}}
                with self.subTest(role=role, field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    launch.validate_reliability_config(config, self.env)

    def test_verified_upstream_checkpoint_defaults_and_complete_explicit_contents(self):
        # Omitting these fields uses the installed defaults, not invented
        # requirements for redundant explicit overrides in every launcher.
        launch.validate_reliability_config(self.config, self.env)
        self.config["actor_rollout_ref"] = {
            "actor": {"checkpoint": {"save_contents": ["model", "optimizer", "extra", "hf_model"]}}
        }
        self.config["critic"] = {
            "checkpoint": {
                "save_contents": ["model", "optimizer", "extra"],
                "load_contents": ["extra", "model", "optimizer"],
            }
        }
        launch.validate_reliability_config(self.config, self.env)

    def test_disabled_reliability_does_not_restrict_legacy_resume_options(self):
        self.config["agentlightning"]["reliability"]["enabled"] = False
        self.config["agentlightning"]["local"]["env_map"] = {}
        self.config["trainer"]["del_local_ckpt_after_load"] = True
        self.config["trainer"]["default_hdfs_dir"] = "hdfs://checkpoints"
        self.config["critic"] = {"checkpoint": {"load_contents": ["model"], "async_save": True}}
        launch.validate_reliability_config(self.config, {})


if __name__ == "__main__":
    unittest.main()
