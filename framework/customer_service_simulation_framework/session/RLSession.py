import json
import random
import logging
import re
from typing import Dict, Any, List, Optional
from session.base import BaseSession
from agents.base import GenericAgent
from core.types import ConversationResult, Context, Response
from core.registry import register_component
from utils.formatters import format_user_system_message, format_assistant_system_message
from core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

@register_component("session", "RLSimulation")
class RLSession(BaseSession):
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type, name, config)
        
    def initialize(self, num_conversation: int, registry: ComponentRegistry):
        self.user_agent = registry.get("agent","user")
        self.assistant_agent = registry.get("agent","customer")
        self.num_conversationts = num_conversation
        self.system_signals_pool = []
        self.assigned_system_signals = []
        self.assigned_user_profiles = []
        self.assigned_core_needs = []
        self.max_turns = self.config.get("max_turns", 20)
        self.user_system_messages = self.prepare_user_system_message(user_agent=self.user_agent, num_conversationts=self.num_conversationts)
        self.assistant_system_messages = self.prepare_assistant_system_message(assistant_agent=self.assistant_agent, num_conversationts=self.num_conversationts)
    
    
    def _add_message_to_contexts(self, response: Response, user_context: Context, assistant_context: Context, is_assistant: bool = False):
        if is_assistant:
            user_context.add_message(Response(content=response.content, metadata=response.metadata))
            assistant_context.add_message(Response(content=response.metadata.get("full_response", response.content), metadata=response.metadata))
        else:
            user_context.add_message(Response(content=response.content, metadata=response.metadata))
            assistant_context.add_message(Response(content=response.content, metadata=response.metadata))

    def start_conversation(self, index: int) -> ConversationResult:
        
        results = ConversationResult()
        user_context = Context()
        assistant_context = Context()
        user_system_mesage = self.user_system_messages[index]
        assistant_system_message = self.assistant_system_messages[index]
        user_requested_end = False

        try:
            # User initiates the first conversation turn
            user_response = self.user_agent.generate_response(user_system_mesage, user_context)
            results.add_result(user_response)
            if not user_response:
                return results
            
            self._add_message_to_contexts(user_response, user_context, assistant_context, is_assistant=False)
            
            # Conversation loop
            for turn in range(self.max_turns):
                # Customer service agent responds
                assistant_response = self.assistant_agent.generate_response(assistant_system_message, assistant_context)
                results.add_result(assistant_response)
                if not assistant_response:
                    break
                
                self._add_message_to_contexts(assistant_response, user_context, assistant_context, is_assistant=True)
                
                # Check if user previously requested end
                if user_requested_end or assistant_response.metadata.get("is_end"):
                    break
                
                # User responds
                user_response = self.user_agent.generate_response(user_system_mesage, user_context)
                
                # Check if user requested to end conversation
                if user_response.metadata.get("is_end"):
                    user_requested_end = True
                    
                results.add_result(user_response)
                if not user_response:
                    break
                
                self._add_message_to_contexts(user_response, user_context, assistant_context, is_assistant=False)
                
                
            
        except Exception as e:
            logger.error(f"Conversation {index} execution failed: {e}", exc_info=True)
        
        return results

    def custom_result(self, conversation_result: ConversationResult, index: int = None) -> Dict[str, Any]:
        """Custom result storage format"""
        total_turns = len(conversation_result.results)
        user_turns = sum(1 for r in conversation_result.results if r.metadata.get("role") == "user")
        assistant_turns = sum(1 for r in conversation_result.results if r.metadata.get("role") == "assistant")
        all_success = all(r.success for r in conversation_result.results)
        
        end_reason = "max_turns_reached"
        if conversation_result.results:
            last_response = conversation_result.results[-1]
            if len(conversation_result.results) >= 2:
                second_last_response = conversation_result.results[-2]
                if last_response.metadata.get("is_end") and last_response.metadata.get("role") == "assistant":
                    end_reason = "assistant_ended"
                elif second_last_response.metadata.get("is_end") and second_last_response.metadata.get("role") == "user":
                    end_reason = "user_ended"
            elif last_response.metadata.get("is_end"):
                end_reason = "user_ended" if last_response.metadata.get("role") == "user" else "assistant_ended"
        
        # Extract user satisfaction (typically appears only once at conversation end)
        satisfaction = None
        for response in conversation_result.results:
            if "satisfaction" in response.metadata:
                satisfaction = response.metadata["satisfaction"]
        
        satisfaction_stats = {
            "has_satisfaction": satisfaction is not None,
            "satisfaction": satisfaction
        }
        
        dialogue_history = []
        for idx, response in enumerate(conversation_result.results):
            agent_type = response.metadata.get("role", "unknown")
            role = "user" if agent_type == "user" else "customer_service"
            
            turn_data = {
                "turn": idx + 1,
                "role": role,
                "agent_type": agent_type,
                "content": response.content,
                "success": response.success,
                "metadata": response.metadata
            }
            
            if "error" in response.metadata:
                turn_data["metadata"]["error"] = response.metadata["error"]
            
            dialogue_history.append(turn_data)
        
        user_profile = None
        system_signals = None
        core_need = None
        
        if index is not None:
            if index < len(self.assigned_user_profiles):
                user_profile = self.assigned_user_profiles[index]
            if index < len(self.assigned_system_signals):
                system_signals = self.assigned_system_signals[index]
            if index < len(self.assigned_core_needs):
                core_need = self.assigned_core_needs[index]
        
        return {
            "summary": {
                "total_turns": total_turns,
                "user_turns": user_turns,
                "assistant_turns": assistant_turns,
                "all_success": all_success,
                "end_reason": end_reason,
                "satisfaction": satisfaction_stats
            },
            "user_profile": user_profile,
            "system_signals": system_signals,
            "core_need": core_need,
            "dialogue": dialogue_history,
        }
        

    def _load_system_signals_file(self, config: Dict[str, Any]):
        """Load system signals file"""
        system_signals_file = config.get("system_signals_file")
        self.system_signals_pool = []
        
        if not system_signals_file:
            logger.warning("System signals file not configured")
            return
        
        try:
            with open(system_signals_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.system_signals_pool.append(json.loads(line))
            logger.debug(f"Loaded system signals pool: {len(self.system_signals_pool)} entries")
        except Exception as e:
            logger.error(f"Failed to load system signals file: {e}")

    
    def _select_user_profile_by_category(self, user_agent: GenericAgent, category: str) -> Dict[str, Any]:
        """
        Select a user profile based on category

        Args:
            user_agent: User agent
            category: User category

        Returns:
            Selected user profile dictionary
        """
        if category in user_agent.user_profiles and user_agent.user_profiles[category]:
            category_data = user_agent.user_profiles[category]
            profiles_list = category_data.get('user_profiles', [])
            if profiles_list:
                selected_profile = random.choice(profiles_list)
                return {
                    "category_info": category_data.get('category_info', {}),
                    "user_profile": selected_profile
                }
            else:
                logger.warning(f"User profile list for category {category} is empty, using empty profile")
        else:
            logger.warning(f"No user profiles for category {category}, using empty profile")
        
        return {"category_info": {"category_name": category}}

    def prepare_user_system_message(self, user_agent: GenericAgent, num_conversationts: int) -> List[str]:
        """
        Prepare user agent system_messages

        Assigns user profiles, system signals, and core needs for each conversation
        based on the user category distribution defined in config.
        User profile distribution must match the user_category_distribution in config.
        Core need distribution uses the distribution field from core_need.json.
        """
        self.user_prompt_template = user_agent.prompt_template
        system_messages = []
        user_category_distribution = self.config.get("user_category_distribution", {})
        
        # Generate conversation assignment list (based on distribution in config)
        conversation_assignments = []
        for category, count in user_category_distribution.items():
            conversation_assignments.extend([category] * count)
        
        # Check if assignment count matches num_conversations
        total_assigned = len(conversation_assignments)
        if total_assigned != num_conversationts:
            raise ValueError(
                f"User category assignment count ({total_assigned}) does not match conversation count ({num_conversationts})! "
                f"Please check the user_category_distribution in config."
            )
        
        # Pre-assign system_signals (one per conversation, shared between user and assistant)
        self._load_system_signals_file(self.config)
        self.assigned_system_signals = [
            [random.choice(self.system_signals_pool)] if self.system_signals_pool else []
            for _ in range(num_conversationts)
        ]
        logger.debug(f"Pre-assigned {len(self.assigned_system_signals)} system_signals")

        # Generate system_message for each conversation
        for i, category in enumerate(conversation_assignments):
            user_profile = self._select_user_profile_by_category(user_agent, category)
            core_demand = self._select_core_need_by_distribution(category)
            
            # Save user profile and core need to list (for later result storage)
            self.assigned_user_profiles.append(user_profile)
            self.assigned_core_needs.append(core_demand)
            
            # Format system_message (using pre-assigned system_signals)
            system_message = format_user_system_message(
                user_profile, 
                self.assigned_system_signals[i], 
                core_demand,
                self.user_prompt_template
            )
            system_messages.append(system_message)
            
        logger.debug(f"Generated {len(system_messages)} user system_messages")
        return system_messages
    
    def _select_core_need_by_distribution(self, category: str) -> Dict[str, Any]:
        """
        Select a core need based on its distribution

        Args:
            category: User category (1-5)

        Returns:
            Selected core need dictionary
        """
        # Get core needs list from user_agent
        core_needs = self.user_agent.core_need_file

        if not core_needs:
            return {"core_need": ""}

        # Get category index (category is string "1"-"5", corresponding to index 0-4)
        try:
            category_idx = int(category) - 1
            if category_idx < 0 or category_idx >= 5:
                logger.warning(f"Category {category} out of range, using random core need")
                return random.choice(core_needs)
        except (ValueError, TypeError):
            logger.warning(f"Invalid category {category}, using random core need")
            return random.choice(core_needs)

        # Extract distribution probability for each core need in this category
        distributions = []
        for core_need in core_needs:
            dist_list = core_need.get("distribution", [])
            if category_idx < len(dist_list):
                distributions.append(dist_list[category_idx])
            else:
                distributions.append(0.0)

        total = sum(distributions)
        if total != 1:
            logger.error(f"Core need distribution for category {category} does not sum to 1, please check the core_need file")
            raise Exception("Core need distribution error")
        
        # Use random.choices to select core need by weighted distribution
        selected = random.choices(core_needs, weights=distributions, k=1)[0]
        logger.debug(f"Category {category} selected core need type {selected.get('core_need_type_name')} (probability: {distributions[core_needs.index(selected)]:.2f})")
        return selected
    
    def prepare_assistant_system_message(self, assistant_agent:GenericAgent, num_conversationts: int) -> List[str]:
        """
        Prepare customer service agent system_messages

        The customer service system_message needs to include system signals (shared with user)
        """
        system_messages = []
        self.assistant_prompt_template = assistant_agent.prompt_template
        for i in range(num_conversationts):
            # Use pre-assigned system_signals (shared with user)
            system_signals = self.assigned_system_signals[i]
            
            # Format customer service system_message
            system_message = format_assistant_system_message(
                system_signals,
                self.assistant_prompt_template
            )
            system_messages.append(system_message)
        
        logger.debug(f"Generated {len(system_messages)} customer service system_messages")
        return system_messages