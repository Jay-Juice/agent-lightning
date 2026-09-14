"""Construct a Critic-only tensor view without mutating Actor tensors."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from verl import DataProto

CRITIC_KEYS = {
    "critic_prompts": "prompts",
    "critic_input_ids": "input_ids",
    "critic_attention_mask": "attention_mask",
    "critic_position_ids": "position_ids",
}


def critic_batch_view(batch: DataProto) -> DataProto:
    present = set(CRITIC_KEYS).intersection(batch.batch.keys())
    if not present:
        return batch
    if present != set(CRITIC_KEYS):
        raise ValueError(f"Incomplete privileged Critic tensors: missing {set(CRITIC_KEYS) - present}")
    tensors = {
        key: value
        for key, value in batch.batch.items()
        if key not in CRITIC_KEYS and key not in CRITIC_KEYS.values()
    }
    tensors.update({target: batch.batch[source] for source, target in CRITIC_KEYS.items()})
    width = batch.batch["responses"].shape[-1]
    if not torch.equal(tensors["input_ids"][:, -width:], batch.batch["responses"]):
        raise ValueError("Critic response IDs differ from Actor response IDs")
    if not torch.equal(
        tensors["attention_mask"][:, -width:], batch.batch["attention_mask"][:, -width:]
    ):
        raise ValueError("Critic response attention differs from Actor response attention")
    return DataProto(
        batch=TensorDict(tensors, batch_size=batch.batch.batch_size),
        non_tensor_batch=batch.non_tensor_batch.copy(),
        meta_info=batch.meta_info.copy(),
    )
