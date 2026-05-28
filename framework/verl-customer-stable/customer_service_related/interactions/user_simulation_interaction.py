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

import logging
import os
import re
import threading
from collections import deque
from typing import Any, Optional
from uuid import uuid4

import ray

from verl.interactions.base import BaseInteraction
from .utils.user_simulator import UserSimulator
from .shared_stats_manager import get_or_create_shared_stats_manager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class UserSimulationInteraction(BaseInteraction):

    def __init__(self, config: dict):
        super().__init__(config)
        logger.info(f"Initializing UserSimulationInteraction with config: {config}")
        
        # Instance management
        self._instance_dict = {}
        
        # Basic configuration
        self.max_turns = self.config.get("max_turns", 20)
        
        # No local batch cache needed; each interaction is submitted directly to the global manager

        # Initialize the global stats manager (Ray Actor)
        # This manager is shared across all processes to address stats inconsistency in distributed training
        self.shared_stats_manager = get_or_create_shared_stats_manager(config)
        
        # Get current cost coefficients from the global manager
        costs = ray.get(self.shared_stats_manager.get_current_costs.remote())
        self.transfer_cost_lambda = costs["transfer_cost_lambda"]
        self.voucher_cost_lambda = costs["voucher_cost_lambda"]
        
        # Thread lock: protects concurrent access to local batch_interactions
        self._local_lock = threading.Lock()
        
        # Load reward configuration
        self._load_reward_config()
        
        # Initialize user simulator
        self._init_user_simulator()
    
    def _load_reward_config(self):
        """Load reward-related configuration parameters"""
        reward_config = self.config.get("reward", {})
        
        # Dynamic penalty configuration - target rates
        self.target_transfer_rate = reward_config.get("target_transfer_rate", 0.2)  # Target transfer-to-human rate
        self.target_voucher_rate = reward_config.get("target_voucher_rate", 0.3)  # Target voucher rate

        # Length limit configuration
        self.max_user_response_length = reward_config.get("max_user_response_length", 2000)  # Max user response length
        self.max_assistant_response_length = reward_config.get("max_assistant_response_length", 5000)  # Max assistant response length
        
        logger.info(
            f"Reward config loaded - Target rates: transfer={self.target_transfer_rate}, "
            f"assistant={self.max_assistant_response_length}"
        )
    
    def _init_user_simulator(self):
        """Initialize user simulator"""
        user_simulator_config = self.config.get("user_simulator")
        if not user_simulator_config:
            raise ValueError("user_simulator config is required")
        self.user_simulator = UserSimulator(user_simulator_config)

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize instance state
        self._instance_dict[instance_id] = {
            "turn_count": 0,
            "score": 0,
            "user_satisfaction": None,
            "user_profile": kwargs.get("user_profile", {}),
            "core_demand": kwargs.get("core_demand",{}),
            "system_signal": kwargs.get("system_signal", {}),
            "user_end_signal": False,  # Flag: whether the user has signaled end
            "has_voucher": False,  # Flag: whether voucher action has been used
            "has_transfer": False,  # Flag: whether transfer action has been used
            "has_format_error": False,  # Flag: whether a format error has occurred
            "split": kwargs.get("split", "train"),  # Flag: train set or validation set
        }
        return instance_id

    def _get_last_assistant_message(self, messages: list[dict[str, Any]]) -> Optional[dict]:
        """
        Get the last message with role 'assistant'

        Args:
            messages: List of messages

        Returns:
            dict: The last assistant message, or None if not found
        """
        if not messages:
            return None

        # Traverse from the end to find the first assistant message
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return message
        
        return None

    def _check_message_format(self, message_content: str) -> bool:
        """
        Check whether the assistant message format is correct

        Args:
            message_content: Content of the assistant message

        Returns:
            bool: True if format is correct, False otherwise
        """
        # Check format: must contain <think></think>, <response></response>, <action></action>
        # Use loose regex patterns, allowing spaces and newlines between tags
        think_pattern = r'<think>.*?</think>'
        response_pattern = r'<response>.*?</response>'
        action_pattern = r'<action>\s*(chat|voucher|transfer)\s*</action>'
        
        # Use DOTALL flag so that . matches all characters including newlines
        has_think = re.search(think_pattern, message_content, re.DOTALL | re.IGNORECASE)
        has_response = re.search(response_pattern, message_content, re.DOTALL | re.IGNORECASE)
        has_action = re.search(action_pattern, message_content, re.DOTALL | re.IGNORECASE)

        # If any format check fails, return False
        return bool(has_think and has_response and has_action)

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        instance = self._instance_dict[instance_id]
        instance["turn_count"] += 1
        should_terminate = False
        additional_data = {}
        
        # Get the last assistant message
        assistant_message = self._get_last_assistant_message(messages)
        
        if assistant_message:
            message_content = assistant_message.get("content", "")
            
            # Check if assistant response length is abnormal
            assistant_length = len(message_content)
            if assistant_length > self.max_assistant_response_length:
                logger.warning(
                    f"Assistant response too long in instance {instance_id}: "
                    f"{assistant_length} > {self.max_assistant_response_length}, terminating with score 0"
                )
                instance["has_format_error"] = True
                should_terminate = True
                user_response = ""
                turn_score = 0.0
                additional_data["reason"] = "assistant_too_long"
                additional_data["assistant_length"] = assistant_length
                return should_terminate, user_response, turn_score, additional_data
            
            # Check message format
            if not self._check_message_format(message_content):
                instance["has_format_error"] = True
                logger.warning(f"Format error in instance {instance_id}, returning score 0")
                should_terminate = True
                user_response = ""
                turn_score = 0.0
                additional_data["reason"] = "assistant_too_long"
                additional_data["assistant_length"] = assistant_length
                return should_terminate, user_response, turn_score, additional_data
            
            action_match = re.search(r'<action>\s*(chat|voucher|transfer)\s*</action>', message_content, re.DOTALL | re.IGNORECASE)
            if action_match:
                action_type = action_match.group(1).lower()
            # Check if voucher action was used
                if action_type == "voucher":
                    instance["has_voucher"] = True

            # Check if transfer action was used
                if action_type == 'transfer':
                    instance["has_transfer"] = True
                    should_terminate = True
                    user_response = ""
                    turn_score = await self.calculate_score(instance_id, is_end=True, end_reason="transfer", messages=messages)
                    additional_data["reason"] = "transfer"
                    return should_terminate, user_response, turn_score, additional_data

        # Check if max turns reached
        if instance["turn_count"] >= self.max_turns:
            should_terminate = True
            user_response = ""
            turn_score = await self.calculate_score(instance_id, is_end=True, end_reason="max_turn", messages=messages)
            additional_data["reason"] = "max_turn"
            return should_terminate, user_response, turn_score, additional_data
        
        # Check if user has signaled end
        if instance.get("user_end_signal"):
            # User signaled end in the previous turn; now actually terminate
            should_terminate = True
            turn_score = await self.calculate_score(
                instance_id, is_end=True, messages=messages, end_reason='user', 
                satisfaction=instance.get("end_satisfaction"), **kwargs
            )
            additional_data["reason"] = "user"
            return should_terminate, "", turn_score, additional_data
        
        # Generate user response
        user_response, user_metadata = self.user_simulator.generate_user_response(
            messages=messages,
            user_profile=instance.get("user_profile"),
            core_demand=instance.get("core_demand"),
            system_signal=instance.get("system_signal")
        )
        
        # Check if user response length is abnormal
        user_response_length = len(user_response)
        if user_response_length > self.max_user_response_length:
            logger.warning(
                f"User response too long in instance {instance_id}: "
                f"{user_response_length} > {self.max_user_response_length}, terminating with score 0"
            )
            instance["has_format_error"] = True
            should_terminate = True
            turn_score = 1.0
            additional_data["reason"] = "user_too_long"
            additional_data["user_length"] = user_response_length
            return should_terminate, user_response, turn_score, additional_data
        
        # Check if user signaled end in current turn
        if user_metadata.get("is_end"):
            # Mark user as signaling end, but don't terminate yet; let the assistant respond once more
            should_terminate = False
            instance["user_end_signal"] = True
            instance["end_satisfaction"] = user_metadata.get("satisfaction")
            additional_data["end_value"] = user_metadata.get("satisfaction")
            turn_score = 0  # No final score calculated for this turn
            additional_data["reason"] = "user_end_signal"
            return should_terminate, user_response, turn_score, additional_data

        return should_terminate, user_response, 0, additional_data
            

    async def _submit_interaction_to_global_manager(self, interaction_record: dict):
        """
        Submit a single interaction record to the global manager (async version)

        This method will:
        1. Submit the interaction record to the global manager
        2. The global manager accumulates records until batch_size before updating cost coefficients
        3. If cost coefficients are updated, fetch the latest values and update the local cache

        Note: Uses async to avoid blocking the event loop
        """
        split = interaction_record.get("split", "train")

        if split == "train":
            # Submit training stats (may trigger cost update)
            # Use async to wait for result without blocking the event loop
            result_ref = self.shared_stats_manager.submit_interaction.remote(
                interaction_record, split="train"
            )
            # Convert ObjectRef to awaitable
            result = await result_ref

            # If costs were updated, sync to local cache
            if result.get("cost_updated", False):
                self.transfer_cost_lambda = result["transfer_cost_lambda"]
                self.voucher_cost_lambda = result["voucher_cost_lambda"]
        else:
            # Submit validation stats (fire-and-forget, no need to wait for result)
            # Don't wait for result, return immediately to avoid blocking
            self.shared_stats_manager.submit_interaction.remote(
                interaction_record, split="test"
            )
    
    
    def get_metrics_for_logging(self) -> dict:
        """
        Get metrics from the global manager for logging

        Returns:
            dict: Dictionary of formatted metrics, with keys in "category/metric_name" format
        """
        # Get metrics from the global manager
        metrics = ray.get(self.shared_stats_manager.get_metrics_for_logging.remote())
        return metrics

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """
        Calculate the final reward score, considering base satisfaction and dynamic penalties

        Args:
            instance_id: Instance ID
            **kwargs: Additional parameters

        Returns:
            float: Final reward score
        """
        is_end = kwargs.get("is_end")
        if not is_end:
            return 0
        
        instance = self._instance_dict[instance_id]
        # If a format error occurred, return 0 directly
        if instance.get("has_format_error"):
            logger.warning(f"Format error in instance {instance_id}, returning score 0")
            return -10.0
        
        # Get base satisfaction score
        base_score = 0.0
        end_reason = kwargs.get("end_reason")
        
        if end_reason == 'user':
            # User ended the session; use user satisfaction score
            base_score = kwargs.get('satisfaction', 0)
        elif end_reason == 'transfer':
            # Transfer to human agent; assign a fixed base score
            base_score = 3
        elif end_reason == 'max_turn':
            # Max turns reached; assign a low score
            base_score = 0
        
        # Save base_score to instance for later statistics
        instance["base_score"] = base_score

        # Validation set does not apply dynamic costs
        if instance.get("split") == "test":
            return base_score
        
        final_score = base_score
        
        # Deduct transfer cost
        if instance.get("has_transfer"):
            final_score -= self.transfer_cost_lambda
        
        # Deduct voucher cost
        if instance.get("has_voucher"):
            final_score -= self.voucher_cost_lambda
        
        return final_score

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Finalize the interaction and immediately submit stats to the global manager

        Args:
            instance_id: Instance ID
            **kwargs: Additional parameters
        """
        instance = self._instance_dict[instance_id]
        
        # Collect interaction record
        interaction_record = {
            "has_transfer": instance.get("has_transfer", False),
            "has_voucher": instance.get("has_voucher", False),
            "base_score": instance.get("base_score", 0.0),  # Include base_score for statistics
            "split": instance.get("split", "train"),
        }
        
        # Submit to global manager immediately (async)
        await self._submit_interaction_to_global_manager(interaction_record)
        
        # Clean up resources
        del self._instance_dict[instance_id]
