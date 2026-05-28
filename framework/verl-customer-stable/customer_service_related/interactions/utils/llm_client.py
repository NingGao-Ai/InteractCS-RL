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
import time
import json
import logging
import requests
from typing import Dict, List, Any, Optional
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BaseLLMClient(ABC):
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 1.0,
        max_tokens: int = 8000,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
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
                if attempt < self.max_retries - 1:
                    wait_time = 5 ** attempt
                    time.sleep(wait_time)
    
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
                    return ""
            except Exception as e:
                return ""


class OpenAIClient(BaseLLMClient):
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 1.0,
        max_tokens: int = 8000,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, timeout, max_retries, **kwargs)
        self.api_url = api_url
        self.api_key = api_key
    
    def call_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        def _call():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            payload.update(self.extra_config)
            payload.update(kwargs)
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code}, {response.text}")
            
            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                raise Exception("Invalid API response format")
            
            return result["choices"][0]["message"]["content"]
        
        return self._retry_with_backoff(_call)


class VLLMClient(BaseLLMClient):
    
    def __init__(
        self,
        api_url: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 8000,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, timeout, max_retries, **kwargs)
        if not api_url.endswith("/chat/completions"):
            if api_url.endswith("/"):
                self.api_url = api_url + "chat/completions"
            else:
                self.api_url = api_url + "/chat/completions"
        else:
            self.api_url = api_url
    
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
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code}, {response.text}")
            
            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                raise Exception("Invalid API response format")
            
            return result["choices"][0]["message"]["content"]
        
        return self._retry_with_backoff(_call)


def create_llm_client(llm_config: Dict[str, Any]) -> BaseLLMClient:
    client_type = llm_config.get("client_type", "openai").lower()
    model = llm_config.get("model")
    api_url = llm_config.get("api_url")
    max_retries = llm_config.get("max_retries", 3)
    kwargs = llm_config.get("kwargs", {})
    
    if not model:
        raise ValueError("Model name is required in llm_config")
    
    if not api_url:
        raise ValueError("API URL is required in llm_config")
    
    if client_type == "openai":
        api_key = llm_config.get("api_key")
        if not api_key:
            raise ValueError("API key is required for OpenAI client")
        return OpenAIClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            **kwargs
        )
    elif client_type == "vllm":
        return VLLMClient(
            api_url=api_url,
            model=model,
            max_retries=max_retries,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported LLM client type: {client_type}. Supported types: 'openai', 'vllm'")
