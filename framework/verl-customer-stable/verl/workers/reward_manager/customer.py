# Copyright 2024 PRIME team and/or its affiliates
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

from typing import Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("customer")
class CustomerRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        sequences_str = data.non_tensor_batch["messages"]
        scores = data.non_tensor_batch["reward_scores"]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        # Convert numpy arrays to lists to avoid serialization issues
        if hasattr(sequences_str, 'tolist'):
            sequences_str = sequences_str.tolist()
        elif not isinstance(sequences_str, list):
            sequences_str = list(sequences_str)
            
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = list(scores)
            
        if hasattr(data_sources, 'tolist'):
            data_sources = data_sources.tolist()
        elif not isinstance(data_sources, list):
            data_sources = list(data_sources)
            
        if extra_info is not None:
            if hasattr(extra_info, 'tolist'):
                extra_info = extra_info.tolist()
            elif not isinstance(extra_info, list):
                extra_info = list(extra_info)

        # Compute scores synchronously (no async/multiprocessing)
        computed_scores = []
        try:
            for completion, reference, task, ei in zip(sequences_str, scores, data_sources, 
                                                        extra_info if extra_info else [None] * len(sequences_str),
                                                        strict=True):
                try:
                    score = self.compute_score(
                        data_source=task,
                        solution_str=completion,
                        ground_truth=reference,
                        extra_info=ei,
                    )
                    
                    if isinstance(score, dict):
                        computed_scores.append(score.get("score", 0.0))
                    else:
                        computed_scores.append(float(score) if score is not None else 0.0)
                except Exception as e:
                    print(f"[Error] Failed to compute score: {e}")
                    computed_scores.append(0.0)
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            computed_scores = [0.0 for _ in range(len(sequences_str))]
        
        data.batch["acc"] = torch.tensor(computed_scores, dtype=torch.float32, device=prompt_ids.device)
        return computed_scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
