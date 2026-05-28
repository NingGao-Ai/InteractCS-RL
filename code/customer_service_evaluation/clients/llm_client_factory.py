"""
LLM client factory class
"""

from typing import Dict, Any
from clients.base_llm_client import BaseLLMClient
from clients.openai_client import OpenAIClient
from clients.vllm_client import VLLMClient


class LLMClientFactory:
    """LLM client factory class"""

    @staticmethod
    def create_client(client_type: str = "openai", **kwargs) -> BaseLLMClient:
        """
        Create an LLM client instance

        Args:
            client_type: Client type ("openai", "vllm")
            **kwargs: Client initialization parameters

        Returns:
            LLM client instance
        """
        if client_type.lower() == "openai":
            return OpenAIClient(**kwargs)
        elif client_type.lower() == "vllm":
            return VLLMClient(**kwargs)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

    @staticmethod
    def create_openai_client(api_url: str = None, api_key: str = None,
                           model: str = None, **kwargs) -> OpenAIClient:
        """
        Convenience method for creating an OpenAI client

        Args:
            api_url: API endpoint URL
            api_key: API key
            model: Model name
            **kwargs: Additional parameters

        Returns:
            OpenAI client instance
        """
        return OpenAIClient(api_url=api_url, api_key=api_key, model=model, **kwargs)

    @staticmethod
    def create_vllm_client(api_url: str = "http://localhost:8000/v1",
                         model: str = None, **kwargs) -> VLLMClient:
        """
        Convenience method for creating a VLLM client

        Args:
            api_url: VLLM API service address
            model: Model name
            **kwargs: Additional parameters

        Returns:
            VLLM client instance
        """
        return VLLMClient(api_url=api_url, model=model, **kwargs)
