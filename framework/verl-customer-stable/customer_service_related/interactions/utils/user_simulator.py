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
import re
from typing import Dict, List, Any, Optional, Tuple
import os
from .llm_client import create_llm_client

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def build_system_state_string(actions: List[str]) -> str:
    is_refunded = "refund" in [a.lower().strip() for a in actions]
    is_voucher_issued = "voucher" in [a.lower().strip() for a in actions]

    status_items = []
    status_items.append(f"Refunded: {'Yes' if is_refunded else 'No'}")
    status_items.append(f"Voucher Issued: {'Yes' if is_voucher_issued else 'No'}")
    return "[SYSTEM STATE: " + " | ".join(status_items) + "]"

class UserSimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get("llm", {})
        self.llm_client = create_llm_client(self.llm_config)
        prompt_template_path = config.get("prompt_template")
        if not prompt_template_path or not self.llm_client:
            raise
        try:
            with open(prompt_template_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except Exception as e:
            logger.error(f"Failed to read prompt template from {prompt_template_path}: {e}")
            raise

    def generate_user_response(
        self,
        messages: List[Dict[str, str]],
        user_profile: Optional[Dict[str, Any]] = None,
        core_demand: Optional[Dict[str, Any]] = None,
        system_signal: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.llm_client:
            raise
        system_message = self._build_system_message(user_profile, core_demand, system_signal)
        llm_messages = [{"role": "system", "content": system_message}]
        # Swap assistant and user roles in the incoming messages
        # Important: assistant messages should only contain content within <response>, not <think> or <action>
        swapped_messages = []
        actions = []
        
        for msg in messages:
            if msg.get("role") == "assistant":
                swapped_msg = msg.copy()
                swapped_msg["role"] = "user"
                # Extract <response> content and action from assistant message
                response_content, action = self._extract_response_content(msg.get("content", ""))
                swapped_msg["content"] = response_content
                swapped_messages.append(swapped_msg)
                # Collect action
                if action:
                    actions.append(action)
            elif msg.get("role") == "user":
                swapped_msg = msg.copy()
                swapped_msg["role"] = "assistant"
                swapped_messages.append(swapped_msg)
        
        # Add system state to messages
        if swapped_messages:
            last_msg = swapped_messages[-1]
            if last_msg.get("role") == "user":
                system_state_str = build_system_state_string(actions)
                last_msg["content"] = f"{last_msg['content']}\n\n{system_state_str}"
        
        llm_messages.extend(swapped_messages)
        response_text = self.llm_client.call_chat_completion(
            llm_messages,
            temperature=kwargs.get("temperature", self.llm_config.get("temperature")),
            max_tokens=kwargs.get("max_tokens", self.llm_config.get("max_tokens"))
        )
        parsed_response, metadata = self._parse_user_response(response_text)
        return parsed_response, metadata

    def _build_system_message(
        self,
        user_profile: Optional[Dict[str, Any]] = None,
        core_demand: Optional[Dict[str, Any]] = None,
        system_signal: Optional[Dict[str, Any]] = None
    ) -> str:
        prompt = self.prompt_template
        
        if user_profile:
            category_name = user_profile.get('category_info', {}).get('category_name', '')
            prompt = prompt.replace('{{CATEGORY}}', category_name)
            profile_str = self._format_profile(user_profile)
            prompt = prompt.replace('{{USER_PROFILE}}', profile_str)
        else:
            prompt = prompt.replace('{{CATEGORY}}', '')
            prompt = prompt.replace('{{USER_PROFILE}}', '')
        
        if core_demand:
            core_need_str = core_demand.get('core_need', '')
            prompt = prompt.replace('{{CORE_DEMAND}}', core_need_str)
        else:
            prompt = prompt.replace('{{CORE_DEMAND}}', '')
        
        if system_signal:
            signal_str = self._format_signal(system_signal)
            prompt = prompt.replace('{{SYSTEM_SIGNALS}}', signal_str)
        else:
            prompt = prompt.replace('{{SYSTEM_SIGNALS}}', '')
        
        return prompt

    def _format_profile(self, profile: Dict[str, Any]) -> str:
        if not profile:
            return ""
        
        category_info = profile.get('category_info', {})
        profile_data = profile.get('user_profile', {})
        
        formatted = []
        
        if category_info:
            if category_info.get('category_name'):
                formatted.append(f"Category: {category_info['category_name']}")
            if category_info.get('description'):
                formatted.append(f"Description: {category_info['description']}")
        
        if profile_data.get('summary'):
            formatted.append(f"User Summary: {profile_data['summary']}")
        
        behavioral_patterns = profile_data.get('behavioralPatterns', {})
        if behavioral_patterns:
            formatted.append("Behavioral Patterns:")
            for key, value in behavioral_patterns.items():
                formatted.append(f"  - {key}: {value}")
        
        inferred_attributes = profile_data.get('inferredAttributes', {})
        if inferred_attributes:
            formatted.append("Inferred Attributes:")
            for key, values in inferred_attributes.items():
                if isinstance(values, list):
                    formatted.append(f"  {key}:")
                    for value in values:
                        formatted.append(f"    - {value}")
                else:
                    formatted.append(f"  {key}: {values}")
        
        return '\n'.join(formatted)

    def _extract_response_content(self, content: str) -> Tuple[str, Optional[str]]:
        """
        Extract content within <response> tags and <action> tags from assistant messages

        Args:
            content: Assistant message content, may contain <think>, <response>, <action> tags

        Returns:
            (response_content, action) tuple
            - response_content: Content within <response> tags, or original content if not found
            - action: Content within <action> tags, or None if not found
        """
        # Extract content within <response> tags
        response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL | re.IGNORECASE)
        response_content = response_match.group(1).strip() if response_match else content.strip()
        
        # Extract content within <action> tags
        action_match = re.search(r'<action>\s*(\w+)\s*</action>', content, re.DOTALL | re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None
        
        return response_content, action

    def _format_signal(self, signal: Dict[str, Any]) -> str:
        if not signal:
            return ""
        
        formatted = []
        formatted.append(f"Order 1:")
        formatted.append(f"  Merchant: {signal.get('merchantNameMap', {}).get('en', '')}")
        formatted.append(f"  Food Items: {signal.get('foodInfo', '')}")
        formatted.append(f"  Food Problem: {signal.get('faqType', '')}")
        
        instant_messages = signal.get('instantMessageMap', '')
        if instant_messages:
            formatted.append(f"  Recent Messages with Courier: {instant_messages}")
        
        metadata = signal.get('metadata', {})
        if metadata.get('region'):
            formatted.append(f"  Region: {metadata['region']}")
        
        return '\n'.join(formatted)

    def _parse_user_response(self, response_text: str) -> Tuple[str, Dict[str, Any]]:
        metadata = {
            "is_end": False,
            "satisfaction": None,
            "original_end_tag": None,
            "role": "user"
        }
        
        # Check for [end] tag
        match = re.search(r"\[end(?::(\d+))?\]", response_text)
        if match:
            metadata["is_end"] = True
            metadata["original_end_tag"] = match.group(0)
            if match.group(1) is not None:
                try:
                    satisfaction = int(match.group(1))
                    if 1 <= satisfaction <= 5:
                        metadata["satisfaction"] = satisfaction
                    else:
                        metadata["satisfaction"] = 1
                except ValueError:
                    metadata["satisfaction"] = 1
            # cleaned_response = response_text.replace(match.group(0), "").strip()
            cleaned_response = response_text.strip()
        else:
            cleaned_response = response_text.strip()
        return cleaned_response, metadata
