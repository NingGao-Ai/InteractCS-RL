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

from collections import defaultdict
from typing import Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("customer_batch_debug")
class CustomerBatchRewardManager:
    """
    Batch reward manager for customer service multi-turn dynamic interactions.
    Combines the batch processing efficiency with customer-specific data format.

    This reward manager is designed for customer service scenarios with:
    1. Multi-turn dialogue data in `data.non_tensor_batch["messages"]`
    2. Pre-computed reward scores in `data.non_tensor_batch["reward_scores"]`
    3. Batch processing for efficiency
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        **reward_kwargs,
    ) -> None:
        """
        Initialize the CustomerBatchRewardManager instance.

        Args:
            tokenizer: The tokenizer used for decoding responses.
            num_examine: The number of responses to examine and print for debugging.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data.
            **reward_kwargs: Additional keyword arguments to pass to the compute_score function.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data: DataProto):
        """
        Verify the batch and compute scores in batch mode.
        Saves the computed scores as "acc" tensor in the batch.

        Args:
            data: The DataProto containing the batch data.

        Returns:
            List of computed scores.
        """
        # Get data from non_tensor_batch (customer-specific format)
        sequences_str = data.non_tensor_batch.get("messages", [])
        reward_scores = data.non_tensor_batch.get("reward_scores", [])
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, [])
        extra_infos = data.non_tensor_batch.get("extra_info", [None] * len(data))

        # Convert numpy arrays to lists if needed (for serialization compatibility)
        if hasattr(sequences_str, 'tolist'):
            sequences_str = sequences_str.tolist()
        elif not isinstance(sequences_str, list):
            sequences_str = list(sequences_str)

        if hasattr(reward_scores, 'tolist'):
            reward_scores = reward_scores.tolist()
        elif not isinstance(reward_scores, list):
            reward_scores = list(reward_scores)

        if hasattr(data_sources, 'tolist'):
            data_sources = data_sources.tolist()
        elif not isinstance(data_sources, list):
            data_sources = list(data_sources)

        # Prepare extra_infos as list
        if extra_infos is not None:
            if hasattr(extra_infos, 'tolist'):
                extra_infos = extra_infos.tolist()
            elif not isinstance(extra_infos, list):
                extra_infos = list(extra_infos)
        else:
            extra_infos = [None] * len(sequences_str)

        # Check if compute_score supports batch mode
        # Try to call with batch parameters first, fall back to loop if not supported
        try:
            # Try batch call (assuming compute_score supports batch parameters)
            scores = self.compute_score(
                data_sources=data_sources,
                solution_strs=sequences_str,
                ground_truths=reward_scores,
                extra_infos=extra_infos,
                **self.reward_kwargs,
            )

            # If batch call succeeded, process the results
            computed_scores = []
            for score in scores:
                if isinstance(score, dict):
                    computed_scores.append(score.get("score", 0.0))
                else:
                    computed_scores.append(float(score) if score is not None else 0.0)

        except (TypeError, ValueError):
            # Fall back to loop if batch call fails (compute_score doesn't support batch)
            computed_scores = []
            for completion, reference, task, ei in zip(
                sequences_str, reward_scores, data_sources, extra_infos, strict=True
            ):
                try:
                    score = self.compute_score(
                        data_source=task,
                        solution_str=completion,
                        ground_truth=reference,
                        extra_info=ei,
                        **self.reward_kwargs,
                    )

                    if isinstance(score, dict):
                        computed_scores.append(score.get("score", 0.0))
                    else:
                        computed_scores.append(float(score) if score is not None else 0.0)
                except Exception as e:
                    print(f"[Error] Failed to compute score: {e}")
                    computed_scores.append(0.0)

        # Save scores as tensor
        prompt_ids = data.batch["prompts"]
        data.batch["acc"] = torch.tensor(computed_scores, dtype=torch.float32, device=prompt_ids.device)

        return computed_scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Compute rewards for the batch.

        Args:
            data: The DataProto containing the batch data.
            return_dict: Whether to return a dictionary with additional information.

        Returns:
            Either reward tensor or dictionary containing reward tensor and extra info.
        """
        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Get batch information
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Get data sources for printing
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, [])
        if hasattr(data_sources, 'tolist'):
            data_sources = data_sources.tolist()
        elif not isinstance(data_sources, list):
            data_sources = list(data_sources)

        # Compute scores in batch mode
        scores = self.verify(data)

        # Track printed data sources for debugging
        already_printed = {}

        # Assign scores to reward tensor
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            # Handle score that might be a dictionary
            if isinstance(score, dict):
                reward = score.get("score", 0.0)
                # Store extra information if available
                for key, value in score.items():
                    if key != "score":  # Don't store score in extra info
                        reward_extra_info[key].append(value)
            else:
                reward = float(score) if score is not None else 0.0

            # Assign reward to the last token of the response
            reward_tensor[i, length - 1] = reward

            # Print debugging information
            data_source = data_sources[i] if i < len(data_sources) else "unknown"
            if already_printed.get(data_source, 0) < self.num_examine:
                # Decode for printing
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:length],
                    skip_special_tokens=True
                )
                prompt_str = self.tokenizer.decode(
                    data.batch["prompts"][i],
                    skip_special_tokens=True
                )

                # Get messages if available (customer-specific format)
                messages = data.non_tensor_batch.get("messages", [])
                if i < len(messages):
                    message_content = messages[i]
                else:
                    message_content = "N/A"

                print(f"[Data Source] {data_source}")
                print(f"[Prompt] {prompt_str[:200]}...")
                print(f"[Response] {response_str[:200]}...")
                print(f"[Messages] {str(message_content)[:200]}...")
                print(f"[Score] {score}")
                print("-" * 50)

                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
