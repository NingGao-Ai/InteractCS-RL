import requests
from typing import Dict, List, Optional
from BaseLLMClient import BaseLLMClient
import logging

logger = logging.getLogger(__name__)


class VLLMClient(BaseLLMClient):
    def __init__(self, api_url: str, model: str, temperature: float = 1.0, max_tokens: int = 8000,
                 timeout: int = 60, max_retries: int = 3, **kwargs):
        super().__init__(model, temperature, max_tokens, timeout, max_retries, **kwargs)
        self.api_url = api_url + "/chat/completions"
    
    def call_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        def _call():
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            payload.update(self.extra_config)
            payload.update(kwargs)
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code}, {response.text}")
            
            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                raise Exception("API response format error")
            
            return result["choices"][0]["message"]["content"]
        
        return self._retry_with_backoff(_call)
