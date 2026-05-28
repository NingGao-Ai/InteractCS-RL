"""
OpenAI API client
"""

import requests
from typing import Dict, List, Any, Optional
from clients.base_llm_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self,
                 api_url: str = None,
                 api_key: str = None,
                 model: str = "gpt-4",
                 max_retries: int = 3,
                 timeout: int = 180,
                 temperature: float = 1.0,
                 max_tokens: int = 8000):
        """
        Initialize OpenAI client

        Args:
            api_url: API endpoint URL
            api_key: API key
            model: Model name to use
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            temperature: Generation temperature
            max_tokens: Maximum token count
        """
        super().__init__(model, max_retries, timeout, temperature, max_tokens)

        self.api_url = api_url or "<your_api_url>"
        self.api_key = api_key or "<your_api_key>"

        # Create session
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })

    def call_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Call OpenAI chat completion API

        Args:
            messages: List of messages
            **kwargs: Additional parameters

        Returns:
            LLM-generated response text
        """
        # Merge parameters
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        def _call_api():

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if "gemini-2.5-pro" in self.model:
                payload.update("reasoning_effort","low")
            response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        try:
            return self._retry_with_backoff(_call_api)
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return None
