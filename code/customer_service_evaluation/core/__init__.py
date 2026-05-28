"""
Core module - Conversation management and parallel processing
"""

from .conversation_manager import ConversationManager, ConversationTurn, ConversationState
from .parallel_dialogue_manager import ParallelDialogueManager

__all__ = [
    'ConversationManager',
    'ConversationTurn',
    'ConversationState',
    'ParallelDialogueManager'
]
