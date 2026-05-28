"""
Customer Service Evaluation Framework

An engineered user-agent dialogue simulation system supporting parallel conversation simulation and multiple LLM backends.
"""

__version__ = "1.0.0"
__author__ = "Customer Service Evaluation Team"

from .core.conversation_manager import ConversationManager
from .core.parallel_dialogue_manager import ParallelDialogueManager
from .simulators.user_simulator import UserSimulator
from .simulators.customer_service_simulator import CustomerServiceSimulator
from .clients.llm_client_factory import LLMClientFactory
from .config.config_manager import ConfigManager

__all__ = [
    'ConversationManager',
    'ParallelDialogueManager',
    'UserSimulator',
    'CustomerServiceSimulator',
    'LLMClientFactory',
    'ConfigManager'
]
