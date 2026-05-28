"""
VLLM client - Supports locally deployed vLLM services
"""

import requests
from typing import Dict, List, Any, Optional
from clients.base_llm_client import BaseLLMClient


class VLLMClient(BaseLLMClient):
    """VLLM client - Supports locally deployed vLLM services"""

    def __init__(self,
                 api_url: str = "http://localhost:8000/v1",
                 model: str = None,
                 max_retries: int = 3,
                 timeout: int = 180,
                 temperature: float = 0.7,
                 max_tokens: int = 8000):
        """
        Initialize VLLM client

        Args:
            api_url: VLLM API service address
            model: Model name to use
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            temperature: Generation temperature
            max_tokens: Maximum token count
        """
        super().__init__(model, max_retries, timeout, temperature, max_tokens)
        self.api_url = api_url

        # Create session
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })

    def call_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Call VLLM chat completion API

        Args:
            messages: List of messages
            **kwargs: Additional parameters

        Returns:
            LLM-generated response text
        """
        # Merge parameters
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', 0.95)

        def _call_api():
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": False
            }

            response = self.session.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        try:
            return self._retry_with_backoff(_call_api)
        except Exception as e:
            print(f"VLLM API call failed: {e}")
            return None
