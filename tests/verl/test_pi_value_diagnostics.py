"""CPU regressions for PI diagnostics; these must never affect PPO tensors."""

from __future__ import annotations

import hashlib
import json
import math
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

from verl import DataProto  # noqa: E402

from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet  # noqa: E402
from agentlightning.verl.rollout_adapter import RolloutAdapter, _privileged_coverage_metrics  # noqa: E402
from agentlightning.verl.value_diagnostics import compute_value_diagnostics  # noqa: E402


def _batch() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "values": torch.tensor([[1.0, 90.0, 3.0], [2.0, 4.0, 90.0], [5.0, 90.0, 90.0], [99.0, 99.0, 99.0]]),
            "returns": torch.zeros(4, 3),
            "response_mask": torch.tensor([[1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]),
        },
        non_tensors={
            "is_pad": np.array([False, False, False, True]),
            "pi_semantic_content": np.array([True, False, True, True]),
            "pi_truncated": np.array([False, False, True, False]),
            "pi_changed_lines_total": np.array([4, 0, 10, 4]),
            "pi_changed_lines_included": np.array([4, 0, 2, 4]),
        },
    )


def test_pi_value_groups_exclude_padding_and_response_holes_without_mutating_batch():
    batch = _batch()
    before = {key: value.clone() for key, value in batch.batch.items()}
    metadata_before = {key: value.copy() for key, value in batch.non_tensor_batch.items()}
    metrics = compute_value_diagnostics(batch)
    assert metrics["value/n_tokens"] == 5
    assert metrics["value/mse_all"] == pytest.approx(11)
    assert metrics["value/mse_first"] == pytest.approx(10)
    assert metrics["value/mse_last"] == pytest.approx(50 / 3)
    assert metrics["value/n_tokens_with_semantic_pi"] == 3
    assert metrics["value/mse_with_semantic_pi"] == pytest.approx(35 / 3)
    assert metrics["value/mae_with_semantic_pi"] == pytest.approx(3)
    assert metrics["value/bias_with_semantic_pi"] == pytest.approx(3)
    assert metrics["value/mse_no_semantic_pi"] == pytest.approx(10)
    assert metrics["value/mse_truncated_pi"] == pytest.approx(25)
    assert metrics["value/n_tokens_full_pi"] == 2
    assert metrics["value/mse_full_pi"] == pytest.approx(5)
    for key, tensor in before.items():
        assert torch.equal(batch.batch[key], tensor)
    for key, array in metadata_before.items():
        assert np.array_equal(batch.non_tensor_batch[key], array)


def test_unchanged_state_is_no_semantic_content_and_not_full_pi():
    batch = _batch()
    batch.non_tensor_batch["pi_semantic_content"][:] = False
    batch.non_tensor_batch["pi_truncated"][:] = False
    batch.non_tensor_batch["pi_changed_lines_total"][:] = 0
    metrics = compute_value_diagnostics(batch)
    assert metrics["value/n_tokens_no_semantic_pi"] == 5
    for group in ("with_semantic_pi", "full_pi", "truncated_pi"):
        assert metrics[f"value/available_{group}"] == 0
        assert metrics[f"value/n_tokens_{group}"] == 0
        assert math.isnan(metrics[f"value/mse_{group}"])


def test_baseline_without_pi_metadata_marks_groups_unavailable():
    batch = _batch()
    batch.non_tensor_batch = {"is_pad": batch.non_tensor_batch["is_pad"]}
    metrics = compute_value_diagnostics(batch)
    assert metrics["value/pi_groups_available"] == 0
    assert metrics["value/mse_all"] == pytest.approx(11)
    assert math.isnan(metrics["value/mse_no_semantic_pi"])


def test_all_padding_has_no_valid_value_samples():
    batch = _batch()
    batch.non_tensor_batch["is_pad"][:] = True
    metrics = compute_value_diagnostics(batch)
    assert metrics["value/n_tokens"] == 0
    for group in ("all", "first", "last", "full_pi", "with_semantic_pi"):
        assert math.isnan(metrics[f"value/mse_{group}"])


def test_coverage_excludes_zero_change_and_reports_weighted_coverage():
    counts = {
        "hunks_total": [0, 1, 3],
        "hunks_included": [0, 1, 1],
        "hunk_chunks_total": [0, 2, 8],
        "hunk_chunks_included": [0, 2, 2],
        "changed_lines_total": [0, 2, 18],
        "changed_lines_included": [0, 2, 2],
    }
    metrics = _privileged_coverage_metrics(counts)
    assert metrics["privilege/hunk_coverage_mean"] == pytest.approx(2 / 3)
    assert metrics["privilege/hunk_coverage_weighted"] == pytest.approx(0.5)
    assert metrics["privilege/changed_line_coverage_mean"] == pytest.approx(5 / 9)
    assert metrics["privilege/changed_line_coverage_weighted"] == pytest.approx(0.2)
    assert metrics["privilege/hunk_chunks_total"] == 10
    assert metrics["privilege/hunk_chunks_included"] == 4
    assert metrics["privilege/n_changed_rows"] == 2
    assert metrics["privilege/n_zero_changed_line_rows"] == 1
    empty = _privileged_coverage_metrics({key: [0, 0] for key in counts})
    assert math.isnan(empty["privilege/changed_line_coverage_mean"])
    assert math.isnan(empty["privilege/changed_line_coverage_weighted"])
    assert empty["privilege/changed_line_coverage_n_rows"] == 0


class Tokenizer:
    all_special_ids: ClassVar[list[int]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return [ord(c) for c in "|".join(message["content"] for message in messages)]


def test_adapter_keeps_per_row_pi_flags_and_actor_tensors_identical(monkeypatch):
    from agentlightning.verl import rollout_adapter

    monkeypatch.setattr(rollout_adapter, "_upload_trace_merge_mismatches_to_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_adapter, "_upload_compact_rollout_trajectories_to_wandb", lambda *args, **kwargs: None)
    tokenizer = Tokenizer()
    rollouts = []
    for idx, (total, included, truncated) in enumerate(((0, 0, False), (4, 4, False), (10, 2, True))):
        messages = [{"role": "user", "content": "fix"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        pi = "patch" if included else ""
        critic_messages = [
            {
                "role": "user",
                "content": "fix" + ("\n\n### Privileged State (critic only; hidden from actor)\n" + pi if pi else ""),
            }
        ]
        critic_prompt = tokenizer.apply_chat_template(critic_messages, tokenize=True, add_generation_prompt=True)

        def digest(ids):
            return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()

        state = {
            "mode": "critic",
            "text": pi,
            "truncated": truncated,
            "actor_prompt_tokens": len(prompt),
            "actor_prompt_hash": digest(prompt),
            "critic_prompt_tokens": len(critic_prompt),
            "critic_prompt_hash": digest(critic_prompt),
            "changed_lines_total": total,
            "changed_lines_included": included,
            "hunks_total": int(total > 0),
            "hunks_included": int(included > 0),
            "hunks_fully_included": int(total > 0 and total == included),
            "hunk_chunks_total": total,
            "hunk_chunks_included": included,
        }
        rollouts.append(
            CompletedRollout(
                rollout_id=f"r{idx}",
                data_id=f"d{idx}",
                step=1,
                sample_idx_in_step=idx,
                enqueue_time=0,
                final_reward=float(idx > 0),
                triplets=[
                    Triplet(
                        prompt={"token_ids": prompt},
                        response={"token_ids": [7, 8], "log_probs": None},
                        metadata={
                            "logical_call_id": f"c{idx}",
                            "privileged_state": state,
                            "actor_messages": messages,
                            "chat_template_kwargs": {},
                        },
                    )
                ],
            )
        )
    kwargs = dict(
        max_prompt_length=100, max_response_length=4, device=torch.device("cpu"), pad_token_id=0, tokenizer=tokenizer
    )
    pi_batch, metrics = RolloutAdapter(**kwargs, privileged_critic_enabled=True).get_train_data_batch(rollouts)
    baseline_batch, _ = RolloutAdapter(**kwargs).get_train_data_batch(rollouts)
    for key, tensor in baseline_batch.batch.items():
        assert torch.equal(tensor, pi_batch.batch[key])
    assert pi_batch.non_tensor_batch["pi_semantic_content"].tolist() == [False, True, True]
    assert pi_batch.non_tensor_batch["pi_truncated"].tolist() == [False, False, True]
    assert pi_batch.non_tensor_batch["pi_changed_lines_included"].tolist() == [0, 4, 2]
    assert metrics["privilege/hunk_coverage_weighted"] == pytest.approx(0.5)
    assert metrics["privilege/n_full_pi_rows"] == 1
    assert metrics["privilege/changed_line_coverage_mean"] == pytest.approx(0.6)
