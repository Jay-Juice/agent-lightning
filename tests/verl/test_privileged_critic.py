from __future__ import annotations

# ruff: noqa: E402
import hashlib
import json
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

from agentlightning.schemas import RolloutState
from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet
from agentlightning.verl.privileged_critic import critic_batch_view
from agentlightning.verl.rollout_adapter import RolloutAdapter


class Tokenizer:
    pad_token_id = 0
    all_special_ids: ClassVar[list[int]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize and add_generation_prompt
        text = "|".join(f"{row['role']}:{row['content']}" for row in messages) + "|assistant:"
        return [ord(char) for char in text]


def _hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def test_adapter_builds_independent_critic_view_with_same_response():
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "fix it"}]
    actor_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    pi = "[PRIVILEGED SANDBOX STATE — CRITIC ONLY]\nfiles_changed=1"
    critic_messages = [
        {
            "role": "user",
            "content": messages[0]["content"]
            + "\n\n### Privileged State (critic only; hidden from actor)\n"
            + pi,
        }
    ]
    critic_prompt = tokenizer.apply_chat_template(critic_messages, tokenize=True, add_generation_prompt=True)
    state = {
        "mode": "critic",
        "text": pi,
        "state_hash": "abc",
        "snapshot_duration_s": 0.25,
        "serialized_tokens": 12,
        "truncated": False,
        "actor_prompt_tokens": len(actor_prompt),
        "actor_prompt_hash": _hash(actor_prompt),
        "critic_prompt_tokens": len(critic_prompt),
        "critic_prompt_hash": _hash(critic_prompt),
    }
    rollout = CompletedRollout(
        rollout_id="r1",
        data_id="d1",
        step=1,
        sample_idx_in_step=0,
        enqueue_time=0,
        rollout_state=RolloutState.SUCCEEDED,
        final_reward=1.0,
        triplets=[
            Triplet(
                prompt={"token_ids": actor_prompt},
                response={"token_ids": [7, 8], "log_probs": None},
                metadata={
                    "logical_call_id": "call-1",
                    "privileged_state": state,
                    "actor_messages": messages,
                    "chat_template_kwargs": {},
                },
            )
        ],
    )
    adapter = RolloutAdapter(
        max_prompt_length=len(critic_prompt) + 2,
        max_response_length=4,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level="transition",
        tokenizer=tokenizer,
        privileged_critic_enabled=True,
    )
    batch, metrics = adapter.get_train_data_batch([rollout])
    actor_before = batch.batch["input_ids"].clone()
    view = critic_batch_view(batch)
    assert torch.equal(batch.batch["input_ids"], actor_before)
    assert not torch.equal(view.batch["input_ids"], batch.batch["input_ids"])
    assert torch.equal(view.batch["input_ids"][:, -4:], batch.batch["responses"])
    assert torch.equal(view.batch["attention_mask"][:, -4:], batch.batch["attention_mask"][:, -4:])
    assert batch.non_tensor_batch["logical_call_id_list"].tolist() == ["call-1"]
    assert metrics["privilege/snapshot_time_s_mean"] == pytest.approx(0.25)
    assert metrics["privilege/critic_total_num_tokens"] == len(critic_prompt) + 2


def test_adapter_rejects_actor_prompt_hash_mismatch():
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "fix it"}]
    actor_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    pi = "[PRIVILEGED SANDBOX STATE — CRITIC ONLY]\nfiles_changed=0"
    critic_messages = [
        {
            "role": "user",
            "content": messages[0]["content"]
            + "\n\n### Privileged State (critic only; hidden from actor)\n"
            + pi,
        }
    ]
    critic_prompt = tokenizer.apply_chat_template(critic_messages, tokenize=True, add_generation_prompt=True)
    state = {
        "mode": "critic",
        "text": pi,
        "state_hash": "abc",
        "snapshot_duration_s": 0.0,
        "serialized_tokens": 0,
        "truncated": False,
        "actor_prompt_tokens": len(actor_prompt),
        "actor_prompt_hash": "tampered",
        "critic_prompt_tokens": len(critic_prompt),
        "critic_prompt_hash": _hash(critic_prompt),
    }
    rollout = CompletedRollout(
        rollout_id="r1",
        data_id="d1",
        step=1,
        sample_idx_in_step=0,
        enqueue_time=0,
        rollout_state=RolloutState.SUCCEEDED,
        final_reward=0.0,
        triplets=[
            Triplet(
                prompt={"token_ids": actor_prompt},
                response={"token_ids": [7], "log_probs": None},
                metadata={
                    "logical_call_id": "call-1",
                    "privileged_state": state,
                    "actor_messages": messages,
                    "chat_template_kwargs": {},
                },
            )
        ],
    )
    adapter = RolloutAdapter(
        max_prompt_length=len(critic_prompt) + 1,
        max_response_length=2,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level="transition",
        tokenizer=tokenizer,
        privileged_critic_enabled=True,
    )
    with pytest.raises(ValueError, match="Actor prompt changed after capture"):
        adapter.get_train_data_batch([rollout])


def test_adapter_rejects_missing_pi():
    adapter = RolloutAdapter(
        max_prompt_length=8,
        max_response_length=4,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level="transition",
        tokenizer=Tokenizer(),
        privileged_critic_enabled=True,
    )
    rollout = CompletedRollout(
        rollout_id="r1",
        data_id="d1",
        step=1,
        sample_idx_in_step=0,
        enqueue_time=0,
        final_reward=0,
        triplets=[Triplet(prompt={"token_ids": [1]}, response={"token_ids": [2], "log_probs": None})],
    )
    with pytest.raises(ValueError, match="Missing Critic PI"):
        adapter.get_train_data_batch([rollout])


def test_critic_view_rejects_partial_tensor_set():
    from verl import DataProto

    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1]]),
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "position_ids": torch.tensor([[0, 1]]),
            "responses": torch.tensor([[2]]),
            "critic_prompts": torch.tensor([[3]]),
        }
    )
    with pytest.raises(ValueError, match="Incomplete privileged Critic tensors"):
        critic_batch_view(batch)
