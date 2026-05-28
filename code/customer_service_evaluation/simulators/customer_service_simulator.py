"""
Customer Service Simulator - Generates agent replies based on prompts and system signals
Supports the response format: <think></think><response></response><action></action>
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from clients.base_llm_client import BaseLLMClient
from core.conversation_manager import ConversationTurn


class CustomerServiceSimulator:
    """Customer service simulator"""

    def __init__(self,
                 llm_client: BaseLLMClient,
                 prompt_file: str):
        """
        Initialize customer service simulator

        Args:
            llm_client: LLM client instance
            prompt_file: Customer service prompt file path
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_file)

        print(f"Customer service simulator initialization complete")

    def _load_prompt(self, prompt_file: str) -> str:
        """Load customer service prompt"""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Failed to load customer service prompt: {e}")

    def format_system_signals_for_prompt(self, system_signals: List[Dict[str, Any]]) -> str:
        """
        Format system signals for prompt

        Args:
            system_signals: System signals dictionary

        Returns:
            Formatted system signals text
        """
        if not system_signals:
            return ""

        signal = system_signals[0]
        formatted = []

        # Format system signals according to prompt requirements
        # if signal.get('orderId'):
        #     formatted.append(f"- orderId: {signal['orderId']}")
        if signal.get('instantMessageMap'):
            formatted.append(f"- instantMessageMap: {signal['instantMessageMap']}")
        if signal.get('abnormalReports'):
            formatted.append(f"- abnormalReports: {signal['abnormalReports']}")
        if signal.get('foodInfo'):
            formatted.append(f"- foodInfo: {signal['foodInfo']}")
        if 'rcTag' in signal:
            formatted.append(f"- rcTag: {signal['rcTag']}")
        if signal.get('merchantNameMap'):
            formatted.append(f"- merchantName: {signal['merchantNameMap']['en']}")

        return '\n'.join(formatted)

    def prepare_conversation_messages(self,
                                    conversation_history: List[ConversationTurn],
                                    system_signals: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Prepare conversation messages

        Args:
            conversation_history: Conversation history
            system_signals: System signals

        Returns:
            List of messages
        """
        # Prepare system message - replace system signals placeholder
        system_message = self.prompt_template.replace(
            '{{SYSTEM_SIGNALS}}',
            self.format_system_signals_for_prompt(system_signals)
        )

        # Build message list
        messages = [{"role": "system", "content": system_message}]

        # Convert conversation history to message format
        for turn in conversation_history:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })

        return messages

    def parse_xml_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse XML-formatted response

        Format: <think></think><response></response><action></action>

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed response data
        """
        try:
            # Extract think section
            think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else ""

            # Extract response section
            response_match = re.search(r'<response>(.*?)</response>', response_text, re.DOTALL)
            response_content = response_match.group(1).strip() if response_match else ""

            # Extract action section
            action_match = re.search(r'<action>(.*?)</action>', response_text, re.DOTALL)
            action_content = action_match.group(1).strip().lower() if action_match else "chat"

            # Validate action value
            if action_content not in ['chat', 'voucher']:
                action_content = 'chat'

            # If action is voucher, append coupon info to response
            if action_content == 'voucher' and response_content:
                response_content += "[SYSTEM: A coupon has been issued to you.]"

            return {
                'think': think_content,
                'response': response_content,
                'action': action_content,
                'full_response': response_text  # Save full response
            }

        except Exception as e:
            print(f"Failed to parse XML response: {e}")
            return self._get_default_response("Response format parsing failed")

    def generate_response(self,
                         conversation_history: List[ConversationTurn],
                         system_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate customer service reply

        Args:
            conversation_history: Conversation history
            system_signals: System signals

        Returns:
            Customer service reply result
        """
        # Prepare messages
        messages = self.prepare_conversation_messages(conversation_history, system_signals)
        # Call LLM
        response_text = self.llm_client.call_chat_completion(messages)

        if not response_text:
            return self._get_default_response("LLM call failed")

        # Parse XML response
        try:
            parsed_response = self.parse_xml_response(response_text)
            return parsed_response

        except Exception as e:
            print(f"Failed to parse customer service reply: {e}")
            return self._get_default_response("Response parsing failed")

    def _get_default_response(self, reason: str) -> Dict[str, Any]:
        """Get default response"""
        return {
            'think': f"System error: {reason}",
            'response': "Sorry, I encountered some technical issues. Please try again later.",
            'action': 'chat',
            'full_response': f"<think>System error: {reason}</think><response>Sorry, I encountered some technical issues. Please try again later.</response><action>chat</action>"
        }
