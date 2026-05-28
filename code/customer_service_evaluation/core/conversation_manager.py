"""
Conversation Manager - Manages the lifecycle and state of a single conversation
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum


class ConversationState(Enum):
    """Conversation state enumeration"""
    INITIALIZED = "initialized"
    USER_TURN = "user_turn"
    CUSTOMER_TURN = "customer_turn"
    COMPLETED = "completed"
    TRANSFERRED = "transferred"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """Conversation turn"""
    role: str  # 'user' or 'assistant'
    content: str  # Full content (agent perspective)
    user_view_content: Optional[str] = None  # User perspective content (response part only)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Conversation Manager - Manages the lifecycle of a single conversation"""

    def __init__(self,
                 conversation_id: str,
                 user_simulator,
                 customer_service_simulator,
                 user_profile: Dict[str, Any],
                 system_signals: Dict[str, Any],
                 max_turns: int = 10,
                 core_demand: Dict[str, Any] = None):
        """
        Initialize conversation manager

        Args:
            conversation_id: Unique conversation identifier
            user_simulator: User simulator instance
            customer_service_simulator: Customer service simulator instance
            user_profile: User profile
            system_signals: System signals
            max_turns: Maximum conversation turns
            core_demand: Core demand
        """
        self.conversation_id = conversation_id
        self.user_simulator = user_simulator
        self.customer_service_simulator = customer_service_simulator
        self.user_profile = user_profile
        self.system_signals = system_signals
        self.max_turns = max_turns
        self.core_demand = core_demand

        # Conversation state
        self.state = ConversationState.INITIALIZED
        self.history: List[ConversationTurn] = []
        self.current_turn = 0
        self.user_requested_end = False  # Flag for whether user requested to end the conversation

        # Statistics
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def initialize_conversation(self) -> bool:
        """Initialize conversation - User initiates the first turn"""
        try:
            # User initiates conversation
            user_response = self.user_simulator.generate_response(
                conversation_history=[],  # Empty conversation history
                user_profile=self.user_profile,
                system_signals=self.system_signals,
                core_demand=self.core_demand
            )

            if user_response:
                self.history.append(ConversationTurn(
                    role='user',
                    content=user_response
                ))
                self.state = ConversationState.CUSTOMER_TURN
                self.current_turn += 1
                return True
            else:
                self.state = ConversationState.ERROR
                return False

        except Exception as e:
            print(f"Conversation initialization failed: {e}")
            self.state = ConversationState.ERROR
            return False

    def next_turn(self) -> bool:
        """Execute the next conversation turn"""
        if self.state in [ConversationState.COMPLETED, ConversationState.TRANSFERRED, ConversationState.ERROR]:
            return False

        if self.current_turn >= self.max_turns:
            self.state = ConversationState.COMPLETED
            self.end_time = time.time()
            return False

        try:
            if self.state == ConversationState.CUSTOMER_TURN:
                # Agent reply - filter [end] markers from user messages
                filtered_history = self._filter_end_markers_from_history(self.history)
                customer_response = self.customer_service_simulator.generate_response(
                    conversation_history=filtered_history,
                    system_signals=self.system_signals
                )

                if customer_response and 'response' in customer_response:
                    # Save full response in metadata
                    metadata = {
                        'think': customer_response.get('think', ''),
                        'action': customer_response.get('action', ''),
                        'full_response': customer_response.get('full_response', '')  # Save full response
                    }

                    # Create conversation turn, save full content and user-view content
                    turn = ConversationTurn(
                        role='assistant',
                        content=customer_response.get('full_response', customer_response['response']),  # Agent sees full content
                        user_view_content=customer_response['response'],  # User can only see the response part
                        metadata=metadata
                    )

                    self.history.append(turn)

                    # Check if conversation should end (user previously requested end)
                    if self.user_requested_end:
                        self.state = ConversationState.COMPLETED
                        self.end_time = time.time()
                        return False

                    self.state = ConversationState.USER_TURN
                    self.current_turn += 1
                    return True
                else:
                    self.state = ConversationState.ERROR
                    return False

            elif self.state == ConversationState.USER_TURN:
                # User reply
                user_response = self.user_simulator.generate_response(
                    conversation_history=self.history,  # Pass conversation history
                    user_profile=self.user_profile,
                    system_signals=self.system_signals,
                    core_demand=self.core_demand
                )

                if user_response:
                    self.history.append(ConversationTurn(
                        role='user',
                        content=user_response
                    ))

                    # Check if user wants to end the conversation
                    if '[end]' in user_response.lower():
                        self.user_requested_end = True
                        # Don't end immediately; let the agent reply one last time

                    self.state = ConversationState.CUSTOMER_TURN
                    self.current_turn += 1
                    return True
                else:
                    self.state = ConversationState.ERROR
                    return False

        except Exception as e:
            print(f"Conversation turn execution failed: {e}")
            self.state = ConversationState.ERROR
            return False

    def _format_context(self) -> str:
        """Format conversation context (agent perspective - sees full content)"""
        context_lines = []
        for turn in self.history:
            if turn.role == 'user':
                context_lines.append(f"user: {turn.content}")
            else:  # assistant
                context_lines.append(f"assistant: {turn.content}")
        return '\n'.join(context_lines)

    def _filter_end_markers_from_history(self, history: List[ConversationTurn]) -> List[ConversationTurn]:
        """
        Filter [end] markers from conversation history so the agent cannot see them

        Args:
            history: Original conversation history

        Returns:
            Filtered conversation history (with [end] markers removed from user messages)
        """
        filtered_history = []
        for turn in history:
            if turn.role == 'user':
                # Remove [end] markers from user messages
                filtered_content = turn.content.replace('[end]', '').replace('[END]', '').strip()
                # If filtered content is empty, keep original content (avoid empty messages)
                if not filtered_content:
                    filtered_content = turn.content
                filtered_turn = ConversationTurn(
                    role=turn.role,
                    content=filtered_content,
                    user_view_content=turn.user_view_content,
                    timestamp=turn.timestamp,
                    metadata=turn.metadata
                )
                filtered_history.append(filtered_turn)
            else:
                # Agent messages remain unchanged
                filtered_history.append(turn)
        return filtered_history

    def _format_context_for_user(self) -> str:
        """Format conversation context (user perspective - sees only the response part)"""
        context_lines = []
        for turn in self.history:
            if turn.role == 'user':
                context_lines.append(f"user: {turn.content}")
            else:  # assistant - user can only see the response part
                user_view = turn.user_view_content if turn.user_view_content is not None else turn.content
                context_lines.append(f"assistant: {user_view}")
        return '\n'.join(context_lines)

    def run_to_completion(self) -> Dict[str, Any]:
        """Run conversation to completion"""
        # Initialize conversation
        if not self.initialize_conversation():
            return self._get_result()

        # Execute conversation turns
        while self.next_turn():
            pass

        return self._get_result()

    def _get_result(self) -> Dict[str, Any]:
        """Get conversation result"""
        if not self.end_time:
            self.end_time = time.time()

        # Build returned conversation history containing both user-view and full-view
        conversation_history = []
        for turn in self.history:
            history_item = {
                'role': turn.role,
                'content': turn.content,  # Full content (agent perspective)
                'timestamp': turn.timestamp,
                'metadata': turn.metadata
            }

            # If it's an agent reply, add user-view content
            if turn.role == 'assistant':
                history_item['user_view_content'] = turn.user_view_content

            conversation_history.append(history_item)

        return {
            'conversation_id': self.conversation_id,
            'user_profile': self.user_profile,
            'system_signals': self.system_signals,
            'core_demand': self.core_demand,
            'conversation_history': conversation_history,
            'total_turns': len(self.history),
            'state': self.state.value,
            'duration': self.end_time - self.start_time,
            'success': self.state not in [ConversationState.ERROR]
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get conversation progress"""
        return {
            'conversation_id': self.conversation_id,
            'current_turn': self.current_turn,
            'max_turns': self.max_turns,
            'state': self.state.value,
            'progress': f"{self.current_turn}/{self.max_turns}"
        }
