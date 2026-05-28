from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from core.types import Context, Response
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "llm"))
from llm import OpenAIClient, VLLMClient, BaseLLMClient

logger = logging.getLogger(__name__)


def create_llm_client(llm_config: Dict[str, Any]) -> Any:
    client_type = llm_config.get("client_type", "openai")
    model = llm_config.get("model", "gpt-4.1")
    max_retries = llm_config.get("max_retries", 3)
    kwargs = llm_config.get("kwargs", {})
    
    if client_type == "openai":
        api_url = llm_config.get("api_url")
        api_key = llm_config.get("api_key")
        if not api_url or not api_key:
            raise ValueError("OpenAI client requires api_url and api_key configuration")
        return OpenAIClient(api_url=api_url, api_key=api_key, model=model, max_retries=max_retries, **kwargs)
    elif client_type == "vllm":
        api_url = llm_config.get("api_url")
        if not api_url:
            raise ValueError("VLLM client requires api_url configuration")
        return VLLMClient(api_url=api_url, model=model, max_retries=max_retries, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM client type: {client_type}")


class GenericAgent(ABC):
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.name = name
        self.component_type = component_type
        
        self.llm_client: BaseLLMClient = None
        try:
            self.llm_client = create_llm_client(self.config["llm"])
            logger.debug(f"{component_type}:{name} LLM client initialized")
        except Exception as e:
            logger.error(f"{component_type}:{name} LLM client initialization failed: {e}")
            raise
        
        self._initialize_agent()
        logger.debug(f"{component_type}:{name} initialization complete")
    
    @abstractmethod
    def _initialize_agent(self):
        pass
        
    
    def _load_prompt(self, prompt_file: str) -> str:
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise Exception(f"Failed to load prompt: {e}")
    
    def generate_response(self, system_message: Optional[str], context: Context) -> Response:
        try:
            context = self._pre_process(context)
            if not self.llm_client:
                raise Exception("LLM client not set")
            
            messages = [{"role": "system", "content": system_message}]
            for msg in context.messages:
                role = "user" if msg.metadata.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.content})

            llm_response = self.llm_client.call_chat_completion(messages)
            parsed_response = self._parse_response(llm_response)
            return self._post_process(parsed_response)
        
        except Exception as e:
            logger.error(f"Agent {self.name} failed to generate response: {e}")
            return Response(content="", success=False, metadata={"agent": self.name, "error": str(e)})
    
    def _pre_process(self, context: Context) -> Context:
        return context
    
    def _post_process(self, parsed_response: Response) -> Response:
        return parsed_response

    @abstractmethod
    def _parse_response(self, llm_response: str) -> Response:
        pass
    
    