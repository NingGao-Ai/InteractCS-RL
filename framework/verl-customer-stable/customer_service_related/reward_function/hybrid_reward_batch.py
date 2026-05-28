#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone reward function - All dependencies merged into a single file
For loading by the verl framework
"""

import os
import sys
import json
import re
import requests
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class RewardAPIConfig:
    """Reward API Configuration"""
    api_url: str
    api_key: str
    model: str
    max_workers: int = 32
    timeout: int = 60
    max_retries: int = 3
    prompt_template_path: str = None
    persona_principle_path: str = None
    
    @classmethod
    def from_friday(cls, api_key: str = None, model: str = "gpt-4.1", api_url: str = None):
        """Create Friday API configuration"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return cls(
            api_url=api_url or "<your_api_url>",
            api_key=api_key or "<your_api_key>",
            model=model,
            prompt_template_path=os.getenv("REWARD_PROMPT_TEMPLATE_PATH", "prompt_for_verify/reward_dialogue_level_v1.txt"),
            persona_principle_path=os.getenv("REWARD_PERSONA_PRINCIPLE_PATH", "prompt_for_verify/user_specific_principle.json")
        )

# ============================================================================
# Prompt Builder
# ============================================================================

class PromptBuilder:
    """Prompt Builder"""
    
    def __init__(self, template_path: str, persona_principle_path: str):
        self.template = self._load_template(template_path)
        self.persona_principles = self._load_persona_principles(persona_principle_path)
    
    def _load_template(self, template_path: str) -> str:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _load_persona_principles(self, principle_path: str) -> Dict:
        with open(principle_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("persona_guidelines", {})
    
    def _match_persona_principle(self, category_name: str) -> str:
        for key, value in self.persona_principles.items():
            if key.lower() in category_name.lower():
                good_case = value.get("good", "")
                bad_case = value.get("bad", "")
                return f"""    * **Good Case (100%):** {good_case}
    * **Bad Case (0%):** {bad_case}"""
        return """    * **Good Case (100%):** The response employs an appropriate strategy for the given user persona.
    * **Bad Case (0%):** The response fails to adapt to the user persona."""
    
    def _format_conversation_history(self, conversation_history, current_turn_index: Optional[int] = None) -> str:
        """
        Format conversation history (supports multiple input formats)

        Args:
            conversation_history: Can be one of the following formats:
                1. List[Dict] - Each dict contains 'role' and 'content' keys
                2. Dict with 'messages' key - Contains a list of Message objects
                3. List[Message] - List of Message objects
            current_turn_index: Truncate to this turn number

        Returns:
            Formatted conversation string
        """
        formatted_lines = []

        # Handle different input formats
        messages = []

        if isinstance(conversation_history, dict) and 'messages' in conversation_history:
            # Format: {'messages': [Message, Message, ...]}
            messages = conversation_history['messages']
        elif isinstance(conversation_history, list):
            # Format: [Dict, ...] or [Message, ...]
            messages = conversation_history
        else:
            # Unknown format, return empty
            return ""

        # Truncate history
        if current_turn_index is not None:
            messages = messages[:current_turn_index]

        for turn in messages:
            # Extract role and content
            role = ""
            turn_content = ""

            if hasattr(turn, 'role') and hasattr(turn, 'content'):
                # Message object
                role = getattr(turn, 'role', '')
                turn_content = getattr(turn, 'content', '')
            elif isinstance(turn, dict):
                # Dictionary format
                role = turn.get('role', '')
                turn_content = turn.get('content', '')
            else:
                # Unknown format, skip
                continue

            # Process assistant messages
            if role == "assistant":
                response_match = re.search(r'<response>(.*?)</response>', turn_content, re.DOTALL)
                if response_match:
                    content = response_match.group(1).strip()
                else:
                    content = "null"
                
                # action_match = re.search(r'<action>\s*{\s*"name"\s*:\s*"customer_service"\s*,\s*"arguments"\s*:\s*{\s*"action"\s*:\s*"(chat|voucher)"\s*}\s*}\s*</action>', turn_content, re.DOTALL | re.IGNORECASE)
                action_match = re.search(r'<action>(.*?)</action>', turn_content, re.DOTALL)
                if action_match:
                    action = action_match.group(1).strip()
                else:
                    action = "null"
                
                content = f"response: {content} action: {action}"
            else:
                content = turn_content
            
            # Format output
            if role == "user":
                formatted_lines.append(f"User: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")
            # elif role == "system":
            #     formatted_lines.append(f"System: {content}")
            # Can add handling for other roles here
        
        return "\n".join(formatted_lines)


    def _format_conversation_history_initial(self, conversation_history: List[Dict], current_turn_index: Optional[int] = None) -> str:
        formatted_lines = []
        history_to_format = conversation_history[:current_turn_index] if current_turn_index else conversation_history
        
        for turn in history_to_format:
            role = turn.get("role", "")
            turn_content = turn.get("content", "")
            
            if role == "assistant":
                response_match = re.search(r'<response>(.*?)</response>', turn_content, re.DOTALL)
                if response_match:
                    content = response_match.group(1).strip()
                else:
                    content = "null"

                action_match = re.search(r'<action>(.*?)</action>', turn_content, re.DOTALL)
                if action_match:
                    action = action_match.group(1).strip()
                else:
                    action = "null"
                content = "response: " + content + " action: " + action
                    
            
            if role == "user":
                formatted_lines.append(f"User: {turn_content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")
        
        return "\n".join(formatted_lines)
    
    def build_prompt_from_extra_info(self, extra_info: Dict[str, Any], solution_str: List[str]) -> str:
        # Safely extract data, ensuring correct types
        interaction_kwargs = extra_info.get("interaction_kwargs", {})
        if not isinstance(interaction_kwargs, dict):
            interaction_kwargs = {}

        user_profile = interaction_kwargs.get("user_profile", {})
        if not isinstance(user_profile, dict):
            user_profile = {}

        category_info = user_profile.get("category_info", {})
        if not isinstance(category_info, dict):
            category_info = {}

        category_name = str(category_info.get("category_name", "Unknown"))
        description = str(category_info.get("description", ""))

        conversation_history = solution_str
        
        '''
        save_dir = "framework/verl-customer-stable/customer_service_related/output/genrm_conv_his_output"
        filename_prefix = "conv_history"
        try:
            # Create directory (if it does not exist)
            os.makedirs(save_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.txt"
            filepath = os.path.join(save_dir, filename)

            # Save as text file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(conversation_history))
                f.write("\n\n")

            print(f"[RewardFunction] Saved history to: {filepath}")

        except Exception as e:
            print(f"[RewardFunction] Error saving history: {e}")
        '''
        # if not isinstance(conversation_history, list):
        #     conversation_history = []
        
        # Format conversation history
        formatted_history = self._format_conversation_history(conversation_history)

        # Get persona principles
        persona_principle = self._match_persona_principle(category_name)

        # Build prompt, ensuring all variables are strings
        prompt = str(self.template)
        prompt = prompt.replace("{category_name}", str(category_name))
        prompt = prompt.replace("{description}", str(description))
        prompt = prompt.replace("{conversation_history}", str(formatted_history))
        prompt = prompt.replace("{Persona-Specific-principle}", str(persona_principle))
        
        return prompt

# ============================================================================
# API Client
# ============================================================================

class RewardAPIClient:
    """Reward API Client"""
    
    def __init__(self, config: RewardAPIConfig):
        self.config = config
    
    def get_reward_score(self, prompt: str) -> tuple:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.api_url,
                    headers=headers,
                    json=data,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                detailed_result = json.loads(content)
                
                scores = detailed_result.get("scores", {})
                # Ensure score is a numeric type
                total_score = 0
                total_weight = 0

                # Define scoring item weights
                # The last item (action_selection) has double weight
                score_keys = list(scores.keys())

                for i, (key, info) in enumerate(scores.items()):
                    score_value = info.get("score", 0)
                    # Convert to float, handling possible string types
                    try:
                        score_float = float(score_value)
                    except (ValueError, TypeError):
                        score_float = 0

                    # Set weights: last item (action_selection) has weight 2, others have weight 1
                    weight = 2.0 if i == len(score_keys) - 1 else 1.0
                
                    total_score += score_float * weight
                    total_weight += weight

                final_score = total_score / (total_weight * 100) if scores and total_weight > 0 else 0.0

                return final_score, detailed_result
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    continue
                else:
                    print(f"API call failed: {e}")
                    return 0.0, None
        
        return 0.0, None
    
    def get_reward_scores_batch(self, prompts: List[str]) -> List[tuple]:
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self.get_reward_score, prompt) for prompt in prompts]
            results = [future.result() for future in futures]
        return results

# ============================================================================
# Global Variables and Initialization
# ============================================================================

_prompt_builder = None
_api_client = None
_config = None

def _initialize_components(config: RewardAPIConfig = None):
    global _prompt_builder, _api_client, _config
    
    if _prompt_builder is None or _api_client is None:
        if config is None:
            reward_backend = os.getenv("REWARD_BACKEND", "friday").lower()
            reward_model = os.getenv("REWARD_MODEL", "gpt-4.1")
            reward_api_key = os.getenv("REWARD_API_KEY", "<your_api_key>")
            reward_api_url = os.getenv("REWARD_API_URL")
            reward_max_workers = int(os.getenv("REWARD_MAX_WORKERS", "32"))
            
            if reward_api_url is None:
                reward_api_url = "<your_api_url>"
            
            config = RewardAPIConfig.from_friday(
                api_key=reward_api_key,
                model=reward_model,
                api_url=reward_api_url
            )
            config.max_workers = reward_max_workers
        
        _config = config
        _prompt_builder = PromptBuilder(
            template_path=config.prompt_template_path,
            persona_principle_path=config.persona_principle_path
        )
        _api_client = RewardAPIClient(config)
        
        print(f"[RewardFunction] Initialization complete")
        print(f"  API URL: {config.api_url}")
        print(f"  Model: {config.model}")
        print(f"  Max Workers: {config.max_workers}")


def save_prompts_to_json(prompts, solution_strs, ground_truths, extra_infos, 
                         save_dir="/path/to/save/prompts", filename_prefix="prompts"):
    """
    Save prompts and related information as a JSON file

    Args:
        prompts: List of prompts
        solution_strs: List of solution strings
        ground_truths: List of ground truth values
        extra_infos: List of extra information
        save_dir: Save directory
        filename_prefix: Filename prefix
    """
    try:
        # Create directory
        os.makedirs(save_dir, exist_ok=True)

        # Prepare data
        data_to_save = []
        for i, (prompt, solution, ground_truth, extra_info) in enumerate(
            zip(prompts, solution_strs, ground_truths, extra_infos)
        ):
            data_to_save.append({
                "index": i,
                "prompt": prompt,
                "solution_str": solution,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"[RewardFunction] Saved {len(prompts)} prompts to JSON file: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"[RewardFunction] Error saving JSON file: {e}")
        return None

# ============================================================================
# Verl Interface Functions
# ============================================================================


def compute_hybrid_score_batch(data_sources: List[str], solution_strs: List[List[str]],
                       ground_truths: List[Any], extra_infos: List[Dict[str, Any]]) -> List[Dict[str, float]]:

    _initialize_components()

    try:
        # Build all prompts
        prompts = []
        for idx, (solution_str, extra_info) in enumerate(zip(solution_strs, extra_infos)):
            try:
                prompt = _prompt_builder.build_prompt_from_extra_info(extra_info, solution_str)
                prompts.append(prompt)
            except Exception as e:
                import traceback
                print(f"[RewardFunction] Error building prompt {idx}: {e}")
                print(f"[RewardFunction] Error details:")
                traceback.print_exc()
                print(f"[RewardFunction] extra_info type: {type(extra_info)}")
                print(f"[RewardFunction] solution_str type: {type(solution_str)}")
                # Use empty prompt as fallback
                prompts.append("")

        print(f"[RewardFunction] GenRM prompt example")
        print(f"[RewardFunction] {prompts[0]}")
        # save_path = "framework/verl-customer-stable/customer_service_related/output/genrm_prompt_output"
        # save_prompts_to_json(prompts, solution_strs, ground_truths, extra_infos, save_path, "hybrid_reward_data")   
        # Batch get generative rewards
        gen_results = _api_client.get_reward_scores_batch(prompts)
        gen_rewards = [score for score, _ in gen_results]
        # gen_rewards = [0.0] * len(solution_strs)

        user_turn_rewards = [float(ground_truth.get('user_turn_rewards', [])[-1]) for ground_truth in ground_truths]

        # Conditionally combine all rewards
        results = []
        for gen_reward, user_turn_reward in zip(gen_rewards, user_turn_rewards):
            gen_reward_applied = 0.0
            if user_turn_reward > 0.0:
                # Apply generative reward
                total_score = 2.0 * gen_reward + user_turn_reward
                gen_reward_applied = 1.0
            else:
                # Format incorrect, only give format reward
                total_score = user_turn_reward

            results.append({
                "score": total_score,
                "gen_reward": gen_reward,
                "user_turn_reward": user_turn_reward,
                "gen_reward_applied": gen_reward_applied
            })

        return results

    except Exception as e:
        import traceback
        print(f"[RewardFunction] Error computing batch reward scores: {e}")
        traceback.print_exc()

        # Return default values on error
        default_result = {
            "score": 0.0,
            "gen_reward": 0.0,
            "user_turn_reward": 0.0,
            "gen_reward_applied": 0.0
        }
        return [default_result] * len(solution_strs)