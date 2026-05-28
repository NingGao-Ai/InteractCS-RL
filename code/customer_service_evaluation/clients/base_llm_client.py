"""
Base LLM client abstract class
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseLLMClient(ABC):
    """Base LLM client abstract class"""

    def __init__(self,
                 model: str = "gpt-4",
                 max_retries: int = 3,
                 timeout: int = 60,
                 temperature: float = 1.0,
                 max_tokens: int = 8000):
        """
        Initialize base LLM client

        Args:
            model: Model name to use
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            temperature: Generation temperature
            max_tokens: Maximum token count
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def call_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Call chat completion API

        Args:
            messages: List of messages
            **kwargs: Additional parameters

        Returns:
            LLM-generated response text
        """
        pass

    def call_llm(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Call LLM for text generation (backward-compatible interface)

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            LLM-generated response text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.call_chat_completion(messages, **kwargs)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON-formatted response"""
        try:
            # Try to parse JSON directly
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract the JSON portion
            try:
                # Find JSON start and end positions
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    raise ValueError("No valid JSON content found")
            except Exception:
                # If still fails, return default response
                return {
                    "responsibility": "OTHERS",
                    "cot": "Response format parsing failed",
                    "chat": "Sorry, I encountered some technical issues.",
                    "action": "COMFORT"
                }

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry mechanism with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(4 ** attempt)  # Exponential backoff
                else:
                    print(f"API call ultimately failed: {e}")
                    raise
