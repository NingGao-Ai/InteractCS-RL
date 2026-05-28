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

"""
Global Stats Manager - For sharing statistics and dynamic cost coefficients in distributed training

This module implements a global state manager based on Ray Actor, solving the problem of
high training variance caused by each process maintaining statistics independently
in distributed training.

Main features:
1. Global statistics collection (transfer-to-human rate, voucher rate, etc.)
2. Dynamic cost adjustment based on PID controller
3. Independent statistics for training and validation sets
4. Thread-safe concurrent access
"""

import logging
import threading
from collections import deque
from typing import Dict, List

import ray

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class SharedStatsManager:
    """
    Global Stats Manager - Ray Actor implementation

    This Actor has only one instance throughout the training process. All worker processes use it to:
    1. Submit batch statistics
    2. Get the current dynamic cost coefficients
    3. Query global statistical metrics

    Design notes:
    - Uses Ray Actor to ensure global uniqueness
    - Thread-safe state updates
    - Supports independent statistics for training and validation sets
    - PID controller for smooth cost adjustment
    """

    def __init__(self, config: dict):
        """
        Initialize the global stats manager

        Args:
            config: Configuration dictionary containing reward-related settings
        """
        self.config = config
        reward_config = config.get("reward", {})
        
        # Dynamic cost coefficients (Lambda) - adjusted via PID controller
        self.transfer_cost_lambda = reward_config.get("initial_transfer_cost", 0.9)
        self.voucher_cost_lambda = reward_config.get("initial_voucher_cost", 0.5)
        
        # Max cost limit
        self.max_cost = reward_config.get("max_cost", 5.0)
        
        # PID controller parameters
        self.kp = reward_config.get("kp", 0.2)
        self.kd = reward_config.get("kd", 0.05)
        # EMA smoothing parameter
        self.cost_ema_alpha = reward_config.get("cost_ema_alpha", 0.7)
        
        # Record previous error for computing D term
        self.prev_transfer_error = 0.0
        self.prev_voucher_error = 0.0

        # Target rates
        self.target_transfer_rate = reward_config.get("target_transfer_rate", 0.2)
        self.target_voucher_rate = reward_config.get("target_voucher_rate", 0.3)
        
        # EMA-smoothed observed rates (initialized to target values)
        self.smoothed_transfer_rate = self.target_transfer_rate
        self.smoothed_voucher_rate = self.target_voucher_rate
        
        # Global statistics - training set
        self.total_interactions = 0
        self.batch_count = 0
        # Sliding window size should accommodate batches from multiple workers
        # Given N workers, each with batch_size B, window_size should be at least N*B*K (K as multiplier, e.g., 4-8)
        # Default 512 can hold: 8 workers * 32 samples/worker * 2x = 512
        self.recent_train_interactions = deque(maxlen=reward_config.get("train_window_size", 128))
        
        # Validation set statistics
        self.test_window_size = reward_config.get("test_window_size", 256)
        self.recent_test_interactions = deque(maxlen=self.test_window_size)
        self.test_total_interactions = 0
        
        # Batch update configuration
        # Accumulate this many interactions before updating cost coefficients
        self.update_batch_size = reward_config.get("update_batch_size", 128)
        self.pending_train_interactions = []  # Pending training interactions

        # Thread lock
        self._lock = threading.Lock()
        
    
    def submit_interaction(self, interaction_record: Dict, split: str = "train") -> Dict:
        """
        Submit a single interaction record (called frequently)

        Args:
            interaction_record: Statistics for a single interaction
            split: "train" or "test"

        Returns:
            dict: Contains current cost coefficients and an update flag
        """
        with self._lock:
            cost_updated = False
            
            if split == "train":
                # Add to pending list
                self.pending_train_interactions.append(interaction_record)
                self.total_interactions += 1
                
                # Check if enough interactions have accumulated for a batch
                if len(self.pending_train_interactions) >= self.update_batch_size:
                    # Batch update
                    self.recent_train_interactions.extend(self.pending_train_interactions)
                    self._update_cost_lambda()
                    self.batch_count += 1
                    cost_updated = True
                    # Clear pending list
                    self.pending_train_interactions = []

            elif split == "test":
                # Validation set: add directly to sliding window
                self.recent_test_interactions.append(interaction_record)
                self.test_total_interactions += 1
            
            # Return current state
            return {
                "transfer_cost_lambda": self.transfer_cost_lambda,
                "voucher_cost_lambda": self.voucher_cost_lambda,
                "total_interactions": self.total_interactions,
                "batch_count": self.batch_count,
                "cost_updated": cost_updated,  # Flag indicating whether costs were updated
            }
    
    def submit_batch_stats(self, batch_interactions: List[Dict], split: str = "train") -> Dict:
        """
        Submit statistics for an entire batch (kept for backward compatibility)

        Args:
            batch_interactions: List of statistics for all interactions in the batch
            split: "train" or "test"

        Returns:
            dict: Dictionary containing current cost coefficients and statistics
        """
        with self._lock:
            if split == "train":
                # Update training set statistics
                self.recent_train_interactions.extend(batch_interactions)
                self.total_interactions += len(batch_interactions)
                self.batch_count += 1
                
                # Update cost coefficients based on sliding window
                self._update_cost_lambda()

            elif split == "test":
                # Update validation set statistics
                self.recent_test_interactions.extend(batch_interactions)
                self.test_total_interactions += len(batch_interactions)
            
            # Return current state
            return {
                "transfer_cost_lambda": self.transfer_cost_lambda,
                "voucher_cost_lambda": self.voucher_cost_lambda,
                "total_interactions": self.total_interactions,
                "batch_count": self.batch_count,
            }
    
    def _update_cost_lambda(self):
        """
        Update cost coefficients based on global statistics in the sliding window (internal method, already locked)

        Strategy:
        1. Compute current transfer and voucher rates from all samples in the sliding window
        2. Apply EMA smoothing to observed rates (filter out sporadic fluctuations)
        3. Compute error using the smoothed rates
        4. PID controller directly integrates to update costs

        Note: EMA is applied to the observed rates, not the costs themselves.
        This filters out sporadic batch fluctuations and updates costs based on longer-term stable trends.
        """
        # Compute statistics from all samples in the sliding window
        if len(self.recent_train_interactions) == 0:
            return
        
        # 1. Compute current statistics from the sliding window
        transfer_count = sum(1 for item in self.recent_train_interactions if item.get("has_transfer"))
        voucher_count = sum(1 for item in self.recent_train_interactions if item.get("has_voucher"))
        window_size = len(self.recent_train_interactions)
        
        current_transfer_rate = transfer_count / window_size
        current_voucher_rate = voucher_count / window_size
        
        # 2. Apply EMA smoothing to observed rates (core correction)
        # This filters out sporadic batch fluctuations, updating based on longer-term stable trends
        self.smoothed_transfer_rate = (
            (1 - self.cost_ema_alpha) * self.smoothed_transfer_rate + 
            self.cost_ema_alpha * current_transfer_rate
        )
        self.smoothed_voucher_rate = (
            (1 - self.cost_ema_alpha) * self.smoothed_voucher_rate + 
            self.cost_ema_alpha * current_voucher_rate
        )
        
        # 3. Compute error using smoothed rates
        transfer_error = self.smoothed_transfer_rate - self.target_transfer_rate
        voucher_error = self.smoothed_voucher_rate - self.target_voucher_rate
        
        # 4. Compute error rate of change (D term input)
        transfer_d_term = transfer_error - self.prev_transfer_error
        voucher_d_term = voucher_error - self.prev_voucher_error


        # # 4. PID controller direct integration update (do not apply EMA to Cost again)
        # self.transfer_cost_lambda += transfer_error * self.kp
        # self.voucher_cost_lambda += voucher_error * self.kp

       
        
        # # 5. Boundary limits
        # self.transfer_cost_lambda = max(0.0, min(self.max_cost, self.transfer_cost_lambda))
        # self.voucher_cost_lambda = max(0.0, min(self.max_cost, self.voucher_cost_lambda))

        transfer_adjustment = (transfer_error * self.kp) + (transfer_d_term * self.kd)
        voucher_adjustment = (voucher_error * self.kp) + (voucher_d_term * self.kd)

        self.transfer_cost_lambda += transfer_adjustment
        self.voucher_cost_lambda += voucher_adjustment
        
        # 6. Update historical error state
        self.prev_transfer_error = transfer_error
        self.prev_voucher_error = voucher_error
        
        # 7. Boundary limits
        self.transfer_cost_lambda = max(0.0, min(self.max_cost, self.transfer_cost_lambda))
        self.voucher_cost_lambda = max(0.0, min(self.max_cost, self.voucher_cost_lambda))

    def get_current_costs(self) -> Dict[str, float]:
        """
        Get the current cost coefficients

        Returns:
            dict: Contains transfer_cost_lambda and voucher_cost_lambda
        """
        with self._lock:
            return {
                "transfer_cost_lambda": self.transfer_cost_lambda,
                "voucher_cost_lambda": self.voucher_cost_lambda,
            }
    
    def get_metrics_for_logging(self) -> Dict:
        """
        Get metrics for logging

        Returns:
            dict: Dictionary containing all statistical metrics
        """
        with self._lock:
            metrics = {
                "interaction/batch_count": self.batch_count,
                "interaction/transfer_cost": self.transfer_cost_lambda,
                "interaction/voucher_cost": self.voucher_cost_lambda,
            }
            
            # Training set statistics (based on sliding window)
            if len(self.recent_train_interactions) > 0:
                train_transfer_count = sum(
                    1 for item in self.recent_train_interactions if item.get("has_transfer")
                )
                train_voucher_count = sum(
                    1 for item in self.recent_train_interactions if item.get("has_voucher")
                )
                # Compute average base_score
                train_base_scores = [item.get("base_score", 0.0) for item in self.recent_train_interactions]
                train_avg_base_score = sum(train_base_scores) / len(train_base_scores) if train_base_scores else 0.0
                
                window_size = len(self.recent_train_interactions)
                
                metrics.update({
                    "interaction/train_transfer_rate": train_transfer_count / window_size,
                    "interaction/train_voucher_rate": train_voucher_count / window_size,
                    "interaction/smoothed_train_transfer_rate": self.smoothed_transfer_rate,
                    "interaction/smoothed_train_voucher_rate": self.smoothed_voucher_rate,
                    "interaction/train_avg_base_score": train_avg_base_score,
                })
            
            # Validation set statistics
            if len(self.recent_test_interactions) > 0:
                test_transfer_count = sum(
                    1 for item in self.recent_test_interactions if item.get("has_transfer")
                )
                test_voucher_count = sum(
                    1 for item in self.recent_test_interactions if item.get("has_voucher")
                )
                test_window_size = len(self.recent_test_interactions)
                
                metrics.update({
                    "val-interaction/total_interactions": self.test_total_interactions,
                    "val-interaction/test_transfer_rate": test_transfer_count / test_window_size,
                    "val-interaction/test_voucher_rate": test_voucher_count / test_window_size,
                    "val-interaction/test_window_size": test_window_size,
                })
            
            return metrics
    


def get_or_create_shared_stats_manager(config: dict, force_create: bool = False) -> ray.actor.ActorHandle:
    """
    Get or create the global stats manager

    This function ensures only one SharedStatsManager instance exists throughout the training process.

    Args:
        config: Configuration dictionary
        force_create: Whether to force creation of a new instance (for testing)

    Returns:
        ray.actor.ActorHandle: Actor handle for SharedStatsManager
    """
    actor_name = "shared_stats_manager"
    
    if not force_create:
        try:
            # Try to get existing actor
            manager = ray.get_actor(actor_name)
            logger.info(f"Retrieved existing SharedStatsManager: {actor_name}")
            return manager
        except ValueError:
            # Actor does not exist, create a new one
            pass
    
    # Create new actor
    manager = SharedStatsManager.options(
        name=actor_name,
        lifetime="detached",  # Ensure actor survives after its creating task ends
        max_concurrency=100,  # Allow concurrent access
    ).remote(config)
    
    logger.info(f"Created new SharedStatsManager: {actor_name}")
    return manager
