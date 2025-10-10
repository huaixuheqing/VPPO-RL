# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig

from torch.distributions import Categorical


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            if getattr(self.config, 'use_vppo_on_entropy', False):
                dist = Categorical(logits=logits_rmpad)
                entropy = dist.entropy()

                if self.config.ulysses_size > 1:
                    entropy = gather_outputs_and_unpad(entropy, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                full_entropy = pad_input(
                    hidden_states=entropy.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
            else:
                entropy = torch.zeros_like(log_probs)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

            if getattr(self.config, 'use_vppo_on_entropy', False):
                dist = Categorical(logits=logits)
                entropy = dist.entropy()
            else:
                entropy = torch.zeros_like(log_probs)

        # Use vppo based on entropy: Return a dictionary
        return {"log_probs": log_probs, "entropy": entropy}

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)['log_probs']
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        if self.config.use_vppo_on_perception:
            select_keys.append("aug_log_probs")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                # Pre-compute sensitivity score statistics over the entire mini-batch for stable normalization.
                global_min_score, global_max_score = None, None
                
                # Only pre-compute if advantage shaping is enabled.
                if self.config.use_advantage_shaping and self.config.use_vppo_on_perception:
                    with torch.no_grad():
                        # Pre-calculate sensitivity scores for all samples in the mini-batch.
                        mb_old_log_probs = mini_batch.batch["old_log_probs"]
                        mb_aug_log_probs = mini_batch.batch["aug_log_probs"]
                        mb_response_mask = mini_batch.batch["response_mask"]
                        
                        # Apply the same stable KL computation as in the micro-batch loop.
                        mb_log_probs_diff = (mb_aug_log_probs - mb_old_log_probs).clamp(-20.0, 20.0)
                        mb_low_var_kl = (mb_log_probs_diff.exp() - mb_log_probs_diff - 1).contiguous()
                        mb_low_var_kl = torch.clamp(mb_low_var_kl, min=0.0, max=10.0)
                        
                        mb_num_valid_tokens = mb_response_mask.sum(dim=1)
                        mb_num_valid_tokens_safe = torch.clamp(mb_num_valid_tokens, min=1)
                        
                        mb_masked_low_var_kl = mb_low_var_kl * mb_response_mask
                        mb_sum_low_var_kl = mb_masked_low_var_kl.sum(dim=1)
                        
                        all_sensitivity_scores = mb_sum_low_var_kl / mb_num_valid_tokens_safe
                        
                        # Determine the min/max scores across the mini-batch to use as a global normalization range.
                        valid_scores_mask = mb_num_valid_tokens > 0
                        valid_scores = all_sensitivity_scores[valid_scores_mask]
                        
                        if valid_scores.numel() > 1:
                            global_min_score = torch.quantile(valid_scores, 0.0) # 使用0%分位数作为下限
                            global_max_score = torch.quantile(valid_scores, 1.0) # 使用100%分位数作为上限
                            
                            # Optional: sync min/max across all devices for a true global range in distributed training.
                            # dist.all_reduce(global_min_score, op=dist.ReduceOp.MIN)
                            # dist.all_reduce(global_max_score, op=dist.ReduceOp.MAX)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # Outputs from the forward pass are shaped (bsz, response_length).
                    output = self._forward_micro_batch(model_inputs, temperature=temperature)
                    log_probs = output['log_probs']
                    entropy = output['entropy']

                    loss_token_mask = None # Default to None

                    if self.config.use_vppo_on_entropy:
                        # Use vppo based on entropy at the response level.
                        # For each response, select the top p% of tokens with the highest entropy.
                        top_p = self.config.top_p_entropy_tokens
                        
                        # Calculate the number of tokens to keep for each response in the batch.
                        # 'k' will be a tensor of shape (mini_batch_size,).
                        num_valid_tokens = response_mask.sum(dim=1)
                        k = torch.ceil(num_valid_tokens * top_p).int()

                        # To ensure padded tokens are not selected, we mask their entropy to a very low value.
                        masked_entropy = entropy.clone()
                        masked_entropy[~response_mask.bool()] = -float('inf')

                        # Sort entropies in descending order to find the top-k values and their original indices.
                        sorted_entropy_vals, sorted_indices = torch.sort(masked_entropy, dim=1, descending=True)
                        
                        # Create a tensor representing the rank of each token within its response.
                        range_tensor = torch.arange(entropy.size(1), device=entropy.device).expand_as(entropy)

                        # Create a mask to identify the top k ranked tokens for each response.
                        # k is unsqueezed to (mini_batch_size, 1) to enable broadcasting.
                        rank_mask = range_tensor < k.unsqueeze(1)

                        # Scatter the rank_mask back to the original token order to create the final mask.
                        top_p_mask = torch.zeros_like(entropy, dtype=torch.bool)
                        top_p_mask.scatter_(1, sorted_indices, rank_mask)
                        
                        loss_token_mask = top_p_mask.to(entropy.dtype)
                        
                        # The subsequent logging code expects a single `threshold` value.
                        # We will calculate the threshold for each response (the k-th largest entropy)
                        # and then average them for logging purposes.
                        
                        # To avoid index errors for responses where k=0, clamp k at 1 for indexing.
                        k_safe_for_indexing = k.clone().clamp(min=1)
                        threshold_indices = (k_safe_for_indexing - 1).unsqueeze(1)
                        
                        # Gather the threshold value for each response from the sorted entropy values.
                        threshold_per_response = torch.gather(sorted_entropy_vals, 1, threshold_indices.long()).squeeze(1)
                        
                        # For responses where k was originally 0, the threshold is not meaningful, so we ignore them.
                        valid_thresholds = threshold_per_response[k > 0]
                        if valid_thresholds.numel() > 0:
                            threshold = valid_thresholds.mean()
                        else:
                            # Fallback if no tokens are selected in any response.
                            threshold = torch.tensor(0.0, device=entropy.device)

                        # Add logging.
                        with torch.no_grad():
                            num_total_valid_tokens = response_mask.sum()
                            num_selected_tokens = top_p_mask.sum()
                            
                            if num_total_valid_tokens > 0:
                                # Log the actual fraction of tokens used.
                                actual_token_fraction = (num_selected_tokens / num_total_valid_tokens).item()
                                metrics["actor/entropy_token_fraction"].append(actual_token_fraction)
                                
                                # Log the entropy threshold to observe its changes during training.
                                metrics["actor/entropy_threshold"].append(threshold.item())

                                # Log mean entropy for selected tokens.
                                selected_entropies = torch.masked_select(entropy, top_p_mask.bool())
                                if selected_entropies.numel() > 0:
                                    metrics["actor/entropy_mean_selected"].append(selected_entropies.mean().item())
                                    
                                # Log mean entropy for rejected tokens for comparison.
                                rejected_mask = response_mask.bool() & ~top_p_mask.bool()
                                rejected_entropies = torch.masked_select(entropy, rejected_mask)
                                if rejected_entropies.numel() > 0:
                                    metrics["actor/entropy_mean_rejected"].append(rejected_entropies.mean().item())

                    if self.config.use_vppo_on_perception:
                        # Use vppo based on perception: Calculate the top p tokens based on perception.
                        top_p = self.config.top_p_perception_tokens
                        
                        aug_log_probs = model_inputs["aug_log_probs"]
                        log_probs_diff = (aug_log_probs - old_log_probs).clamp(-20.0, 20.0)
                        low_var_kl = (log_probs_diff.exp() - log_probs_diff - 1).contiguous()
                        low_var_kl = torch.clamp(low_var_kl, min=0.0, max=10.0)

                        low_var_kl_for_sort = low_var_kl.clone()
                        invalid_mask = ~response_mask.bool()
                        low_var_kl_for_sort[invalid_mask] = -torch.inf

                        # Calculate the number of tokens to keep for each response.
                        num_valid_tokens = response_mask.sum(dim=1)
                        k = torch.ceil(num_valid_tokens * top_p).int()

                        # Sort the perception differences in descending order to get values and original indices.
                        sorted_vals, sorted_indices = torch.sort(low_var_kl_for_sort, dim=1, descending=True)
                        
                        # Create a rank mask to identify the top k positions in each response.
                        range_tensor = torch.arange(low_var_kl_for_sort.size(1), device=low_var_kl_for_sort.device).expand_as(low_var_kl_for_sort)
                        rank_mask = range_tensor < k.unsqueeze(1)

                        # Use scatter to map the rank mask back to the original token order, creating the final perception mask.
                        top_p_mask = torch.zeros_like(low_var_kl_for_sort, dtype=torch.bool)
                        top_p_mask.scatter_(1, sorted_indices, rank_mask)
                        
                        top_p_mask = (top_p_mask.bool() & response_mask.bool()).to(log_probs.dtype)

                        if loss_token_mask is not None:
                            loss_token_mask = (loss_token_mask.bool() | top_p_mask.bool()).to(log_probs.dtype)
                        else:
                            loss_token_mask = top_p_mask

                        # Calculate an average threshold for logging purposes.
                        # This is the mean of the k-th largest perception difference for each response.
                        k_safe_for_indexing = k.clone().clamp(min=1)
                        threshold_indices = (k_safe_for_indexing - 1).unsqueeze(1)
                        
                        threshold_per_response = torch.gather(sorted_vals, 1, threshold_indices.long()).squeeze(1)
                        
                        valid_thresholds = threshold_per_response[k > 0]
                        if valid_thresholds.numel() > 0:
                            threshold = valid_thresholds.mean()
                        else:
                            threshold = torch.tensor(0.0, device=low_var_kl_for_sort.device)

                        # Add logging.
                        with torch.no_grad():
                            num_total_valid_tokens = response_mask.sum()
                            num_selected_tokens = top_p_mask.sum()
                            
                            if num_total_valid_tokens > 0:
                                actual_token_fraction = (num_selected_tokens / num_total_valid_tokens).item()
                                metrics["actor/perception_token_fraction"].append(actual_token_fraction)
                                metrics["actor/low_var_kl_threshold"].append(threshold.item())

                                selected_low_var_kl = torch.masked_select(low_var_kl, top_p_mask.bool())
                                if selected_low_var_kl.numel() > 0:
                                    metrics["actor/low_var_kl_mean_selected"].append(selected_low_var_kl.mean().item())

                                rejected_mask = response_mask.bool() & ~top_p_mask.bool()
                                rejected_low_var_kl = torch.masked_select(low_var_kl, rejected_mask)
                                if rejected_low_var_kl.numel() > 0:
                                    metrics["actor/low_var_kl_mean_rejected"].append(rejected_low_var_kl.mean().item())

                    if self.config.use_vppo_on_entropy and self.config.use_vppo_on_perception:
                        # Add combined logging.
                        metrics["actor/combined_token_fraction"].append(
                            (loss_token_mask.sum() / response_mask.sum()).item()
                        )
                        metrics["actor/combined_entropy_mean_selected"].append(
                            torch.masked_select(entropy, loss_token_mask.bool()).mean().item()
                        )
                        metrics["actor/combined_entropy_mean_rejected"].append(
                            torch.masked_select(entropy, response_mask.bool() & ~loss_token_mask.bool()).mean().item()
                        )
                        metrics["actor/combined_perception_mean_selected"].append(
                            torch.masked_select(low_var_kl, loss_token_mask.bool()).mean().item()
                        )
                        metrics["actor/combined_perception_mean_rejected"].append(
                            torch.masked_select(low_var_kl, response_mask.bool() & ~loss_token_mask.bool()).mean().item()
                        )
                            
                    # Apply Advantage Shaping if enabled.
                    if self.config.use_advantage_shaping:
                        if 'low_var_kl' in locals() or 'low_var_kl' in globals():
                            with torch.no_grad():
                                # Calculate sensitivity scores for the current micro-batch.
                                num_valid_tokens = response_mask.sum(dim=1)
                                num_valid_tokens_safe = torch.clamp(num_valid_tokens, min=1)
                                
                                masked_low_var_kl = low_var_kl * response_mask
                                sum_low_var_kl = masked_low_var_kl.sum(dim=1)
                                
                                sensitivity_score = sum_low_var_kl / num_valid_tokens_safe

                                # Apply scaling using pre-computed global statistics.
                                scaling_factor = torch.ones_like(sensitivity_score)
                                
                                if global_min_score is not None and global_max_score is not None and (global_max_score - global_min_score) > 1e-6:
                                    valid_scores_mask_micro = num_valid_tokens > 0
                                    valid_scores_micro = sensitivity_score[valid_scores_mask_micro]
                                    
                                    # Normalize scores using the global min/max and clamp to [0, 1] for robustness.
                                    normalized_scores = (valid_scores_micro - global_min_score) / (global_max_score - global_min_score)
                                    normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)

                                    target_min = self.config.advantage_scaling_min
                                    # Calculate the mean of normalized scores for the valid samples in the micro-batch
                                    mu_norm = normalized_scores.mean()

                                    # Add a small epsilon for numerical stability in case mu_norm is zero
                                    epsilon = 1e-8
                                    # Dynamically calculate target_max
                                    target_max = target_min + (1.0 - target_min) / (mu_norm + epsilon)

                                    # Log this dynamic value to see how it changes
                                    metrics["actor/dynamic_scaling_max"].append(target_max.item())

                                    # Map normalized scores to the DYNAMIC range [target_min, target_max].
                                    target_range = target_max - target_min
                                    mapped_scores = target_min + normalized_scores * target_range

                                    scaling_factor[valid_scores_mask_micro] = mapped_scores

                                
                                # Log metrics, including the global normalization range.
                                if (num_valid_tokens > 0).any():
                                    metrics["actor/sensitivity_score_mean"].append(sensitivity_score[num_valid_tokens > 0].mean().item())
                                if global_min_score is not None:
                                    metrics["actor/global_sensitivity_score_min"].append(global_min_score.item())
                                    metrics["actor/global_sensitivity_score_max"].append(global_max_score.item())
                                metrics["actor/scaling_factor_mean"].append(scaling_factor.mean().item())

                            # Apply the final scaling factor to the advantages.
                            advantages = advantages * scaling_factor.unsqueeze(1)


                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                        loss_token_mask=loss_token_mask
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    if self.config.use_entropy_penalty:
                        # Use entropy penalty for training
                        entropy_loss = -VF.masked_mean(log_probs, response_mask)
                        loss = loss + entropy_loss * self.config.entropy_penalty_coef
                        metrics["actor/entropy_penalty_coef"] = self.config.entropy_penalty_coef

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
