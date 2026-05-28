import time
import json
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    def __init__(self, model: str = "gpt-4.1", temperature: float = 1.0, max_tokens: int = 8000,
                 timeout: int = 60, max_retries: int = 3, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs
    
    @abstractmethod
    def call_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        pass

    def call_llm(self, prompt: str, **kwargs) -> Optional[str]:
        messages = [{"role": "user", "content": prompt}]
        return self.call_chat_completion(messages, **kwargs)
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 ** attempt)
                else:
                    logger.error(f"API call ultimately failed: {e}")
                    raise Exception(f"LLM call failed: {e}")
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    raise ValueError("No valid JSON content found")
            except Exception as e:
                logger.error(f"JSON parsing failed: {e}")
                raise Exception(f"Response format parsing failed: {e}")
