"""
Format Correctness Checker - Checks if assistant reply format meets requirements
"""

import re
from typing import Dict, List, Any, Tuple


class FormatEvaluator:
    """Format correctness checker"""

    def __init__(self):
        """Initialize format checker"""
        # Define required tags
        self.required_tags = ['think', 'response', 'action']

        # Define allowed values for chat tag
        self.allowed_chat_values = ['chat', 'voucher']

    def evaluate_conversation(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate format correctness of a conversation

        Args:
            conversation_history: Conversation history

        Returns:
            Format evaluation result
        """
        results = {
            'format_correct': True,
            'format_errors': [],
            'voucher_count': 0,
            'voucher_positions': [],
            'total_assistant_turns': 0
        }

        # Check each assistant reply
        for i, turn in enumerate(conversation_history):
            if turn.get('role') == 'assistant':
                results['total_assistant_turns'] += 1

                content = turn.get('content', '')

                # Check format correctness
                format_result = self._check_format_correctness(content)
                if not format_result['correct']:
                    results['format_correct'] = False
                    results['format_errors'].append({
                        'turn': i + 1,
                        'errors': format_result['errors']
                    })

                # Count voucher occurrences
                voucher_result = self._count_voucher_actions(content)
                if voucher_result['count'] > 0:
                    results['voucher_count'] += voucher_result['count']
                    results['voucher_positions'].extend([
                        {'turn': i + 1, 'position': pos}
                        for pos in voucher_result['positions']
                    ])

        # Check for multiple vouchers (excluding content within think tags)
        results['multiple_vouchers'] = results['voucher_count'] > 1

        return results

    def _check_format_correctness(self, content: str) -> Dict[str, Any]:
        """
        Check format correctness of a single assistant reply

        Args:
            content: Assistant reply content

        Returns:
            Format check result
        """
        result = {
            'correct': True,
            'errors': []
        }

        # Check if all required tags are present
        for tag in self.required_tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            matches = re.findall(pattern, content, re.DOTALL)

            if not matches:
                result['correct'] = False
                result['errors'].append(f'Missing <{tag}> tag')
            elif tag == 'chat':
                # Check chat tag content
                chat_content = matches[0].strip()
                if chat_content not in self.allowed_chat_values:
                    result['correct'] = False
                    result['errors'].append(f'<action> tag content "{chat_content}" is invalid, only "chat" or "voucher" allowed')

        # Check tag order (optional but recommended)
        think_pattern = r'<think>.*?</think>'
        response_pattern = r'<response>.*?</response>'
        chat_pattern = r'<chat>.*?</chat>'

        think_match = re.search(think_pattern, content)
        response_match = re.search(response_pattern, content)
        chat_match = re.search(chat_pattern, content)

        if think_match and response_match and chat_match:
            think_end = think_match.end()
            response_start = response_match.start()
            response_end = response_match.end()
            chat_start = chat_match.start()

            if think_end > response_start:
                result['correct'] = False
                result['errors'].append('<think> tag must come before <response> tag')

            if response_end > chat_start:
                result['correct'] = False
                result['errors'].append('<response> tag must come before <chat> tag')

        return result

    def _count_voucher_actions(self, content: str) -> Dict[str, Any]:
        """
        Count voucher action occurrences (excluding content within think tags)

        Args:
            content: Assistant reply content

        Returns:
            Voucher count result
        """
        result = {
            'count': 0,
            'positions': []
        }

        # First remove content within think tags
        content_without_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # Find <action>voucher</action> pattern
        voucher_pattern = r'<action>\s*voucher\s*</action>'
        matches = re.finditer(voucher_pattern, content_without_think, re.IGNORECASE)

        for match in matches:
            result['count'] += 1
            result['positions'].append(match.start())

        return result

    def get_summary(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for format evaluation

        Args:
            evaluation_results: Format evaluation results for all conversations

        Returns:
            Summary statistics
        """
        total_conversations = len(evaluation_results)
        format_correct_count = sum(1 for r in evaluation_results if r['format_correct'])
        multiple_vouchers_count = sum(1 for r in evaluation_results if r['multiple_vouchers'])

        return {
            'total_conversations': total_conversations,
            'format_correct_count': format_correct_count,
            'format_correct_rate': format_correct_count / total_conversations if total_conversations > 0 else 0,
            'multiple_vouchers_count': multiple_vouchers_count,
            'multiple_vouchers_rate': multiple_vouchers_count / total_conversations if total_conversations > 0 else 0,
            'average_voucher_count': sum(r['voucher_count'] for r in evaluation_results) / total_conversations if total_conversations > 0 else 0
        }
