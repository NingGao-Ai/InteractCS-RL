"""
Compensation Evaluator - Uses evaluation-compensation.txt for evaluation
"""

import json
import os
from typing import Dict, List, Any, Optional
from core.evaluation_result import CompensationEvaluationResult
from clients.llm_client_factory import LLMClientFactory


class CompensationEvaluator:
    """Compensation evaluator"""

    def __init__(self, llm_config, prompt_file: str):
        """
        Initialize compensation evaluator

        Args:
            llm_config: LLM configuration object
            prompt_file: Compensation evaluation prompt file path
        """
        self.llm_client = LLMClientFactory.create_client(**llm_config.__dict__)
        self.prompt_file = prompt_file
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to load compensation evaluation prompt file: {e}")

    def _format_dialogue_for_compensation_evaluation(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Format dialogue for compensation evaluation

        Compensation evaluation uses the complete user and assistant dialogue
        """
        dialogue_lines = []

        for turn in conversation_history:
            if turn['role'] == 'user':
                dialogue_lines.append(f"user: {turn['content']}")
            elif turn['role'] == 'assistant':
                # Compensation evaluation uses full content (agent perspective)
                dialogue_lines.append(f"assistant: {turn['content']}")

        return '\n'.join(dialogue_lines)

    def has_voucher(self, conversation_history: List[Dict[str, Any]]) -> bool:
        """
        Check if the conversation contains a voucher

        Args:
            conversation_history: Conversation history

        Returns:
            Whether the conversation contains a voucher
        """
        for turn in conversation_history:
            if turn.get('role') == 'assistant' and '<action>voucher</action>' in turn.get('content', ''):
                return True
        return False

    def evaluate(self, conversation_history: List[Dict[str, Any]]) -> Optional[CompensationEvaluationResult]:
        """
        Evaluate dialogue compensation quality

        Args:
            conversation_history: Conversation history

        Returns:
            CompensationEvaluationResult: Compensation evaluation result, or None if no compensation
        """
        # Check if there is compensation
        if not self.has_voucher(conversation_history):
            return None

        try:
            # Format dialogue (using full dialogue)
            dialogue_text = self._format_dialogue_for_compensation_evaluation(conversation_history)

            # Build prompt
            prompt = self.prompt_template.replace("{{DIALOGUE}}", dialogue_text)

            # Call LLM for evaluation
            response = self.llm_client.call_chat_completion(
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse evaluation result
            evaluation_data = self._parse_evaluation_response(response)

            # Extract reason
            compensation_data = evaluation_data.get('compensation_strategy_adaptation', {})
            reason = ""
            if isinstance(compensation_data, dict):
                reason_value = compensation_data.pop('reason', '')  # Extract and remove reason field
                # Ensure reason is string type
                reason = str(reason_value) if reason_value else ''

            # Create evaluation result object
            return CompensationEvaluationResult(
                compensation_strategy_adaptation=compensation_data,
                reason=reason
            )

        except Exception as e:
            print(f"Compensation evaluation failed: {e}")
            # Return default evaluation result (all zeros)
            return CompensationEvaluationResult()

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse evaluation response

        Args:
            response: LLM response text

        Returns:
            Parsed evaluation data
        """
        try:
            # Strip whitespace from response
            response = response.strip()
            # If it starts with ```json, remove the prefix and possible trailing ```
            if response.startswith("```json"):
                response = response[len("```json"):].strip()
                if response.endswith("```"):
                    response = response[:-3].strip()
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                evaluation_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response directly
                evaluation_data = json.loads(response)

            # Ensure all score values are integers
            return self._convert_scores_to_int(evaluation_data)

        except json.JSONDecodeError as e:
            print(f"Failed to parse compensation evaluation response JSON: {e}")
            print(f"Raw response: {response}")
            # Return empty evaluation result
            return {}
        except Exception as e:
            print(f"Failed to parse compensation evaluation response: {e}")
            return {}

    def _convert_scores_to_int(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert all score values in evaluation data to integers

        Args:
            evaluation_data: Raw evaluation data

        Returns:
            Converted evaluation data
        """
        converted_data = {}

        for category, scores in evaluation_data.items():
            if isinstance(scores, dict):
                converted_scores = {}
                for criterion, score in scores.items():
                    if criterion == 'reason':
                        converted_scores[criterion] = score
                        continue
                    # Convert score values to integers
                    if isinstance(score, (int, float)):
                        converted_scores[criterion] = int(score)
                    elif isinstance(score, str) and score.isdigit():
                        converted_scores[criterion] = int(score)
                    else:
                        # If conversion fails, use default value 0
                        converted_scores[criterion] = 0
                converted_data[category] = converted_scores
            else:
                converted_data[category] = scores

        return converted_data
