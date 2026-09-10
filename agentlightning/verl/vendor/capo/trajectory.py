# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Selected CAPO trainer functions; only local imports are redirected."""

import math
from functools import reduce
from typing import Optional

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.ray_trainer import compute_response_mask

from .verl_core_algos import AdvantageEstimator

def get_valid_data(data: DataProto) -> tuple[DataProto, torch.Tensor]:
    """Extract valid (non-padded) data from a DataProto object.

    Args:
        data (DataProto): The data potentially containing padded samples.

    Returns:
        tuple[DataProto, torch.Tensor]: A tuple containing the valid data and a boolean mask
            of valid indices.
    """
    is_pad = data.non_tensor_batch.get("is_pad", None)
    if is_pad is not None:
        valid_mask = torch.from_numpy(~is_pad).to(data.batch.device)
        valid_data = data.select_idxs(valid_mask)
    else:
        valid_mask = torch.ones(len(data), dtype=torch.bool, device=data.batch.device)
        valid_data = data
    return valid_data, valid_mask


def _agent_adv_estimator_key(adv_estimator: AdvantageEstimator | str) -> str:
    """Normalize Hydra / enum to `algorithm.adv_estimator` string for ARFT routing."""
    if isinstance(adv_estimator, AdvantageEstimator):
        return adv_estimator.value
    return str(adv_estimator)


def _critic_vf_loss_response_mask(response_mask: torch.Tensor, adv_key: str) -> torch.Tensor:
    """
    ``response_mask`` to pass into ``dp_critic.update_critic`` (VF loss uses this as ``loss_mask``).

    - ``token_gae``: clone of full mask — train V on every LLM token (aligns with
      ``arft.core_algos.compute_token_gae_advantage_return``).
    - ``gae`` (and any other adv): only ``[:, 0]`` — step-level scalar V per agent step (aligns with
      ``arft.core_algos.compute_gae_advantage_return`` using ``values[:, 0]``).

    Repo audit (``zeros_like(response_mask)`` + ``[:, 0] = 1`` for critic): **only this helper**
    implements that shrink for ARFT agent PPO. ``_compute_values`` / ``dp_critic.compute_values`` still
    use the batch's real ``response_mask``; no second site needs changing for ``token_gae``.
    """
    if adv_key == "token_gae":
        return response_mask.clone()
    value_mask = torch.zeros_like(response_mask)
    value_mask[:, 0] = 1
    return value_mask


def _scatter_step_gae_diagnostics(
    data: DataProto,
    valid_mask: torch.Tensor,
    step_diagnostics: dict[str, list[float]],
) -> None:
    """Write per-row step GAE diagnostics into ``data.non_tensor_batch``."""
    batch_size = len(data)
    valid_indices = torch.where(valid_mask)[0].cpu().numpy()
    for key, values in step_diagnostics.items():
        arr = np.full(batch_size, None, dtype=object)
        for batch_idx, value in zip(valid_indices, values):
            arr[batch_idx] = value
        data.non_tensor_batch[key] = arr


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator | str,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    gigpo_step_advantage_w: float = 1.0,
    gigpo_mode: str = "mean_std_norm",
    gigpo_enable_similarity: bool = False,
    gigpo_similarity_thresh: float = 0.95,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    # TODO: 重写所有 core_algos 中的 advantage 函数，适配新型的 agent flow 数据结构
    # 多行 data 对应一条完整轨迹，通过 non_tensor_batch["trajectory_uids"] 来区分不同轨迹，每条轨迹包含多行 data。
    # 通过 non_tensor_batch["step_indices"] 来区分同一条轨迹内的不同 step 的顺序。
    """Compute advantage estimates for policy optimization.

    Uses **only** ``arft.core_algos``: ``gae`` → ``compute_gae_advantage_return``,
    ``token_gae`` → ``compute_token_gae_advantage_return``, ``grpo`` → ``compute_grpo_outcome_advantage``,
    ``reinforce_plus_plus`` → ``compute_reinforce_plus_plus_outcome_advantage``,
    ``rloo`` → ``compute_rloo_outcome_advantage``, ``gigpo`` → ``compute_gigpo_outcome_advantage``.
    Dispatch is by string key from ``_agent_adv_estimator_key`` (Hydra often passes plain ``str``).

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: ``AdvantageEstimator`` member or equivalent string (e.g. ``"token_gae"``).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    advantages = torch.zeros_like(data.batch["token_level_rewards"])
    returns = torch.zeros_like(data.batch["token_level_rewards"])

    valid_data, valid_mask = get_valid_data(data)

    adv_key = _agent_adv_estimator_key(adv_estimator)

    if adv_key == "gae":
        from .arft_core_algos import compute_gae_advantage_return

        valid_advantages, valid_returns, step_diagnostics = compute_gae_advantage_return(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            values=valid_data.batch["values"],
            response_mask=valid_data.batch["response_mask"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            step_indices=valid_data.non_tensor_batch["step_indices"],
            gamma=gamma,
            lam=lam,
            return_step_diagnostics=True,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
        _scatter_step_gae_diagnostics(data, valid_mask, step_diagnostics)
    elif adv_key == "token_gae":
        from .arft_core_algos import compute_token_gae_advantage_return

        valid_advantages, valid_returns = compute_token_gae_advantage_return(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            values=valid_data.batch["values"],
            response_mask=valid_data.batch["response_mask"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            step_indices=valid_data.non_tensor_batch["step_indices"],
            gamma=gamma,
            lam=lam,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_key == "grpo":
        from .arft_core_algos import compute_grpo_outcome_advantage

        valid_advantages, valid_returns = compute_grpo_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            index=valid_data.non_tensor_batch["uid"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_key == "reinforce_plus_plus":
        from .arft_core_algos import compute_reinforce_plus_plus_outcome_advantage

        valid_advantages, valid_returns = compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            gamma=gamma,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_key == "reinforce_plus_plus_baseline":
        from .arft_core_algos import compute_reinforce_plus_plus_baseline_outcome_advantage

        valid_advantages, valid_returns = compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            index=valid_data.non_tensor_batch["uid"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_key == "rloo":
        from .arft_core_algos import compute_rloo_outcome_advantage

        valid_advantages, valid_returns = compute_rloo_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            index=valid_data.non_tensor_batch["uid"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_key == "gigpo":
        from .arft_core_algos import compute_gigpo_outcome_advantage, compute_step_discounted_returns

        if "anchor_obs" not in valid_data.non_tensor_batch:
            raise KeyError(
                "algorithm.adv_estimator='gigpo' requires non_tensor_batch['anchor_obs']. "
                "Set step.extra_fields['anchor_obs'] in the agent flow before using GiGPO."
            )
        step_rewards = compute_step_discounted_returns(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            step_indices=valid_data.non_tensor_batch["step_indices"],
            gamma=gamma,
        )
        valid_advantages, valid_returns = compute_gigpo_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            step_rewards=step_rewards,
            response_mask=valid_data.batch["response_mask"],
            anchor_obs=valid_data.non_tensor_batch["anchor_obs"],
            index=valid_data.non_tensor_batch["uid"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            step_advantage_w=gigpo_step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    else:
        raise ValueError(
            f"RayAgentTrainer.compute_advantage: unsupported adv_estimator={adv_estimator!r} (key={adv_key!r}). "
            "Supported: 'gae', 'token_gae', 'grpo', 'reinforce_plus_plus', "
            "'reinforce_plus_plus_baseline', 'rloo', 'gigpo' → arft.core_algos.*"
        )

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def _pad_dataproto_to_world_size(self, batch):
    world_sizes = []
    if self.use_critic and self.critic_wg.world_size != 0:
        world_sizes.append(self.critic_wg.world_size)
    if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
        world_sizes.append(self.ref_policy_wg.world_size)
    if self.hybrid_engine:
        if self.actor_rollout_wg.world_size != 0:
            world_sizes.append(self.actor_rollout_wg.world_size)
    else:
        if self.actor_wg.world_size != 0:
            world_sizes.append(self.actor_wg.world_size)
        if self.rollout_wg.world_size != 0:
            world_sizes.append(self.rollout_wg.world_size)
    if not world_sizes:
        return batch

    world_size = reduce(math.lcm, world_sizes)

    original_batch_size = batch.batch["prompts"].shape[0]
    batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
    batch.non_tensor_batch["is_pad"] = np.array([False] * original_batch_size + [True] * pad_size)

    return batch
