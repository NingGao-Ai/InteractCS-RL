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
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CustomerServiceTool(BaseTool):
    """Customer service tool for handling voucher and chat actions.

    This tool is triggered when LLM outputs:
    <tool_call>
    {"name": "customer_service", "arguments": {"action": "voucher"}}
    </tool_call>

    or:
    <tool_call>
    {"name": "customer_service", "arguments": {"action": "chat"}}
    </tool_call>

    The tool will return "voucher success" or "chat success" accordingly.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "customer_service",
                "description": "A tool for customer service actions like voucher distribution or chat handling",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The action to perform: 'voucher' for voucher distribution, 'chat' for chat handling",
                            "enum": ["voucher", "chat"]
                        },
                    },
                    "required": ["action"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Initialize instance state
        self._instance_dict[instance_id] = {
            "action": None,
            "executed": False,
            "reward": 0.0,
        }

        logger.error(f"Created customer service tool instance: {instance_id}")
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        """Execute the customer service tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The parameters containing the action to perform.

        Returns:
            tuple: (tool_response, tool_reward_score, tool_metrics)
                - tool_response: "voucher success" or "chat success"
                - tool_reward_score: Always 0.0 (no step reward)
                - tool_metrics: Empty dict
        """
        action = parameters.get("action", "").lower()

        if not isinstance(action, str):
            action = str(action)

        # Validate action
        if action not in ["voucher", "chat"]:
            error_msg = f"Invalid action: {action}. Must be 'voucher' or 'chat'."
            logger.warning(error_msg)
            return error_msg, 0.0, {"error": "invalid_action"}

        # Update instance state
        self._instance_dict[instance_id]["action"] = action
        self._instance_dict[instance_id]["executed"] = True

        logger.error(f"Customer service tool executed: instance_id={instance_id}, action={action}")

        # Return appropriate response
        if action == "voucher":
            return "voucher success", 0.0, {}
        else:  # action == "chat"
            return "chat success", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        For customer service tool, we can give a small positive reward
        for successfully executing an action.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (0.1 for successful execution).
        """
        instance = self._instance_dict.get(instance_id)
        if instance and instance.get("executed", False):
            return 0.0  # Small positive reward for using the tool
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.error(f"Released customer service tool instance: {instance_id}")
