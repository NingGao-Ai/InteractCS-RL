"""
User Simulator - Generates personalized user replies based on user profiles and system signals
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from clients.base_llm_client import BaseLLMClient
from core.conversation_manager import ConversationTurn


class UserSimulator:
    """User simulator"""

    def __init__(self,
                 llm_client: BaseLLMClient,
                 prompt_file: str,
                 user_profiles_file: str,
                 system_signals_file: str,
                 core_need_file: str = None):
        """
        Initialize user simulator

        Args:
            llm_client: LLM client instance
            prompt_file: User simulation prompt file path
            user_profiles_file: User profiles file path
            system_signals_file: System signals pool file path
            core_need_file: Core demand file path
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template(prompt_file)

        # Load data
        self.user_profiles = self._load_user_profiles(user_profiles_file)
        self.system_signals = self._load_system_signals(system_signals_file)
        self.core_needs = self._load_core_needs(core_need_file) if core_need_file else []

        print(f"User simulator initialization complete")
        print(f"Loaded {len(self.user_profiles)} user profile categories")
        print(f"Loaded {len(self.system_signals)} system signals")
        print(f"Loaded {len(self.core_needs)} core demand types")

    def _load_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template"""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Failed to load prompt file: {e}")

    def _load_user_profiles(self, file_path: str) -> Dict[str, Any]:
        """Load user profile data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('user_profiles', {})
        except Exception as e:
            print(f"Failed to load user profiles: {e}")
            return {}

    def _load_core_needs(self, file_path: str) -> List[Dict[str, Any]]:
        """Load core demand data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Failed to load core demands: {e}")
            return []

    def _load_system_signals(self, file_path: str) -> List[Dict[str, Any]]:
        """Load system signal data"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            signal = json.loads(line)
                            if self._is_signal_valid(signal):
                                data.append(signal)
                        except json.JSONDecodeError as e:
                            print(f"JSON parse error (line {line_num}): {e}")
                            continue
            return data
        except Exception as e:
            print(f"Failed to load system signals: {e}")
            return []

    def _is_signal_valid(self, signal: Dict[str, Any]) -> bool:
        """Check if a system signal is valid"""
        # Check top-level fields
        for key in ['orderId', 'merchantNameMap', 'foodInfo', 'rcTag']:
            if key not in signal or not signal[key]:
                return False
            # For dict types, check if all values are non-empty
            if isinstance(signal[key], dict):
                if not all(signal[key].values()):
                    return False
        return True

    def get_random_user_profile(self, category: str = None) -> Dict[str, Any]:
        """
        Get a random user profile

        Args:
            category: User profile category (1-5); if None, randomly selected

        Returns:
            User profile data
        """
        if category:
            category_data = self.user_profiles.get(category)
        else:
            # Randomly select a category
            available_categories = list(self.user_profiles.keys())
            if not available_categories:
                return {}
            category = random.choice(available_categories)
            category_data = self.user_profiles[category]

        if not category_data:
            return {}

        # Randomly select one from this category's user profiles
        user_profiles = category_data.get('user_profiles', [])
        if not user_profiles:
            return {}

        selected_profile = random.choice(user_profiles)
        return {
            'category_info': category_data.get('category_info', {}),
            'user_profile': selected_profile
        }

    def get_random_core_demand(self, user_category: str) -> Dict[str, Any]:
        """
        Randomly select a core demand based on user profile category

        Args:
            user_category: User profile category (1-5)

        Returns:
            Core demand data
        """
        if not self.core_needs:
            return {}

        # Convert user category to index (1-5 -> 0-4)
        try:
            category_index = int(user_category) - 1
            if category_index < 0 or category_index >= 5:
                category_index = 0  # Default to first category
        except (ValueError, TypeError):
            category_index = 0

        # Calculate probability for each core demand type
        probabilities = []
        for core_need in self.core_needs:
            distribution = core_need.get('distribution', [])
            if len(distribution) > category_index:
                prob = distribution[category_index]
            else:
                prob = 0.0
            probabilities.append(prob)

        # If all probabilities are 0, use uniform distribution
        if sum(probabilities) == 0:
            probabilities = [1.0 / len(self.core_needs)] * len(self.core_needs)

        # Randomly select based on probabilities
        selected_core_need = random.choices(self.core_needs, weights=probabilities, k=1)[0]
        return selected_core_need

    def get_random_system_signals(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get random system signals

        Args:
            count: Number of system signals needed

        Returns:
            List of system signals
        """
        if not self.system_signals:
            return []

        return random.sample(self.system_signals, min(count, len(self.system_signals)))

    def format_user_profile_for_prompt(self, user_profile: Dict[str, Any]) -> str:
        """Format user profile for prompt"""
        if not user_profile:
            return ""

        category_info = user_profile.get('category_info', {})
        profile_data = user_profile.get('user_profile', {})

        formatted = []

        # Category information
        if category_info:
            formatted.append(f"Category: {category_info.get('category_name', '')}")
            formatted.append(f"Description: {category_info.get('description', '')}")

        # User summary
        if profile_data.get('summary'):
            formatted.append(f"User Summary: {profile_data['summary']}")

        # Behavioral patterns
        behavioral_patterns = profile_data.get('behavioralPatterns', {})
        if behavioral_patterns:
            formatted.append("Behavioral Patterns:")
            for key, value in behavioral_patterns.items():
                formatted.append(f"  - {key}: {value}")

        # Inferred attributes
        inferred_attributes = profile_data.get('inferredAttributes', {})
        if inferred_attributes:
            formatted.append("Inferred Attributes:")
            for key, values in inferred_attributes.items():
                if isinstance(values, list):
                    formatted.append(f"  {key}:")
                    for value in values:
                        formatted.append(f"    - {value}")
                else:
                    formatted.append(f"  {key}: {values}")

        return '\n'.join(formatted)

    def format_system_signals_for_prompt(self, system_signals: List[Dict[str, Any]]) -> str:
        """Format system signals for prompt"""
        if not system_signals:
            return ""

        formatted = []
        for i, signal in enumerate(system_signals, 1):
            formatted.append(f"Order {i}:")
            # formatted.append(f"  Order ID: {signal.get('orderId', '')}")
            formatted.append(f"  Merchant: {signal.get('merchantNameMap', {}).get('en', '')}")
            # formatted.append(f"  Order Status: {signal.get('orderStatus', '')}")
            formatted.append(f"  Food Items: {signal.get('foodInfo', '')}")
            formatted.append(f"  Food Problem: {signal.get('faqType', '')}")
            # Handle instant messages
            instant_messages = signal.get('instantMessageMap', '')
            if instant_messages:
                formatted.append(f"  Recent Messages with Courier: {instant_messages}")

            # Region information
            metadata = signal.get('metadata', {})
            if metadata.get('region'):
                formatted.append(f"  Region: {metadata['region']}")

        return '\n'.join(formatted)

    def generate_response(self,
                         conversation_history: List[ConversationTurn],
                         user_profile: Dict[str, Any] = None,
                         system_signals: List[Dict[str, Any]] = None,
                         core_demand: Dict[str, Any] = None) -> Optional[str]:
        """
        Generate user reply

        Args:
            conversation_history: Conversation history
            user_profile: User profile; if None, randomly selected
            system_signals: System signals; if None, randomly selected
            core_demand: Core demand; if None, randomly selected based on user profile

        Returns:
            User reply text
        """
        # Raise error if not provided
        if user_profile is None:
            raise ValueError("User profile is empty")
        if system_signals is None:
            raise ValueError("System signals are empty")

        # Raise error if core demand not provided
        if core_demand is None and self.core_needs:
            raise ValueError("Core demand is empty")

        # Format data
        formatted_user_profile = self.format_user_profile_for_prompt(user_profile)
        formatted_system_signals = self.format_system_signals_for_prompt(system_signals)

        # Format conversation history
        formatted_history = self._format_conversation_history(conversation_history)

        # Prepare prompt
        prompt = self.prompt_template
        prompt = prompt.replace('{{CATEGORY}}', user_profile.get('category_info', {}).get('category_name', ''))
        prompt = prompt.replace('{{USER_PROFILE}}', formatted_user_profile)
        prompt = prompt.replace('{{SYSTEM_SIGNALS}}', formatted_system_signals)
        prompt = prompt.replace('{{CONTEXT}}', formatted_history)

        # Replace core demand placeholder
        if core_demand:
            core_need_text = core_demand.get('core_need', '')
            prompt = prompt.replace('{{CORE_DEMAND}}', core_need_text)
        else:
            prompt = prompt.replace('{{CORE_DEMAND}}', 'No specific core demand')

        # Call LLM
        response = self.llm_client.call_llm(prompt)
        return response

    def generate_batch_responses(self,
                               requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate user replies in batch

        Args:
            requests: List of requests, each containing conversation_history, user_profile, system_signals

        Returns:
            List of reply results
        """
        results = []

        for i, request in enumerate(requests):
            try:
                response = self.generate_response(
                    conversation_history=request.get('conversation_history', []),
                    user_profile=request.get('user_profile'),
                    system_signals=request.get('system_signals'),
                    core_demand=request.get('core_demand')
                )

                results.append({
                    'request_id': i + 1,
                    'conversation_history': request.get('conversation_history', []),
                    'user_profile': request.get('user_profile'),
                    'system_signals': request.get('system_signals'),
                    'core_demand': request.get('core_demand'),
                    'simulated_response': response,
                    'success': response is not None
                })
            except Exception as e:
                results.append({
                    'request_id': i + 1,
                    'conversation_history': request.get('conversation_history', []),
                    'user_profile': request.get('user_profile'),
                    'system_signals': request.get('system_signals'),
                    'core_demand': request.get('core_demand'),
                    'simulated_response': None,
                    'success': False,
                    'error': str(e)
                })

        return results

    def _format_conversation_history(self, conversation_history: List[ConversationTurn]) -> str:
        """
        Format conversation history for user simulator

        Args:
            conversation_history: Conversation history

        Returns:
            Formatted conversation history text
        """
        if not conversation_history:
            return "Now you start your first Message:"

        formatted_lines = []
        for turn in conversation_history:
            if turn.role == 'user':
                formatted_lines.append(f"user: {turn.content}")
            else:  # assistant
                # User can only see the response part
                user_view = turn.user_view_content if turn.user_view_content is not None else turn.content
                formatted_lines.append(f"assistant: {user_view}")

        return "Dialogue:\n"'\n'.join(formatted_lines)
