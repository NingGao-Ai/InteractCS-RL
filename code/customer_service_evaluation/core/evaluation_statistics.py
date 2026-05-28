"""
Evaluation Statistics - Aggregates and analyzes evaluation results
"""

from typing import Dict, List, Any, Tuple
import json
import os
from .evaluation_result import EvaluationResult


class EvaluationStatistics:
    """Evaluation statistics calculator"""

    def __init__(self, evaluation_results: List[EvaluationResult]):
        """
        Initialize statistics calculator

        Args:
            evaluation_results: List of evaluation results
        """
        self.evaluation_results = evaluation_results

    def calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics"""
        if not self.evaluation_results:
            return {}

        speech_scores = [result.speech_evaluation.total_score for result in self.evaluation_results]
        logic_scores = [result.logic_evaluation.total_score for result in self.evaluation_results]

        # Calculate compensation evaluation (only for conversations with compensation)
        compensation_results = [result for result in self.evaluation_results if result.compensation_evaluation]
        compensation_scores = [result.compensation_evaluation.total_score for result in compensation_results] if compensation_results else []

        stats = {
            'total_dialogues': len(self.evaluation_results),
            'speech_evaluation_score': {
                'average': sum(speech_scores) / len(speech_scores),
                'max': max(speech_scores),
                'min': min(speech_scores),
                'median': self._calculate_median(speech_scores),
                'max_possible': 28  # Max speech evaluation score is 28
            },
            'logic_evaluation_score': {
                'average': sum(logic_scores) / len(logic_scores),
                'max': max(logic_scores),
                'min': min(logic_scores),
                'median': self._calculate_median(logic_scores),
                'max_possible': 12  # Max logic evaluation score is 12
            }
        }

        # If compensation evaluations exist, add compensation statistics
        if compensation_scores:
            stats['compensation_evaluation_score'] = {
                'average': sum(compensation_scores) / len(compensation_scores),
                'max': max(compensation_scores),
                'min': min(compensation_scores),
                'median': self._calculate_median(compensation_scores),
                'max_possible': 6,  # Max compensation evaluation score is 6
                'evaluated_dialogues': len(compensation_results),
                'evaluated_dialogue_ratio': (len(compensation_results) / len(self.evaluation_results)) * 100
            }

        return stats

    def calculate_detailed_statistics(self) -> Dict[str, Any]:
        """Calculate detailed statistics"""
        if not self.evaluation_results:
            return {}

        # Detailed speech evaluation statistics
        speech_categories = [
            'identity_neutrality', 'dialogue_quality', 'language_adaptability',
            'content_quality', 'communication_effectiveness', 'natural_fluency', 'context_adaptability'
        ]

        speech_stats = {}
        for category in speech_categories:
            category_scores = []
            for result in self.evaluation_results:
                category_data = getattr(result.speech_evaluation, category)
                category_score = sum(category_data.values())
                category_scores.append(category_score)

            speech_stats[category] = {
                'average': sum(category_scores) / len(category_scores),
                'max': max(category_scores),
                'min': min(category_scores)
            }

        # Detailed logic evaluation statistics
        logic_categories = [
            'user_profile_recognition', 'business_rule_capability'
        ]

        logic_stats = {}
        for category in logic_categories:
            category_scores = []
            for result in self.evaluation_results:
                category_data = getattr(result.logic_evaluation, category)
                category_score = sum(category_data.values())
                category_scores.append(category_score)

            logic_stats[category] = {
                'average': sum(category_scores) / len(category_scores),
                'max': max(category_scores),
                'min': min(category_scores)
            }

        # Detailed compensation evaluation statistics (only for conversations with compensation)
        compensation_results = [result for result in self.evaluation_results if result.compensation_evaluation]
        compensation_stats = {}

        if compensation_results:
            compensation_scores = []
            for result in compensation_results:
                compensation_score = sum(result.compensation_evaluation.compensation_strategy_adaptation.values())
                compensation_scores.append(compensation_score)

            compensation_stats['compensation_strategy_adaptation'] = {
                'average': sum(compensation_scores) / len(compensation_scores),
                'max': max(compensation_scores),
                'min': min(compensation_scores),
                'evaluated_dialogues': len(compensation_results)
            }

        return {
            'speech_evaluation': speech_stats,
            'logic_evaluation': logic_stats,
            'compensation_evaluation': compensation_stats
        }

    def calculate_score_distribution(self) -> Dict[str, Any]:
        """Calculate score distribution"""
        if not self.evaluation_results:
            return {}

        distribution = {}

        # Speech score distribution (max 28)
        speech_scores = [result.speech_evaluation.total_score for result in self.evaluation_results]
        speech_ranges = [(0, 7), (8, 14), (15, 21), (22, 28)]
        distribution['speech_score_distribution'] = {
            'score_ranges': ['0-7', '8-14', '15-21', '22-28'],
            'dialogue_count': self._count_scores_in_ranges(speech_scores, speech_ranges)
        }

        # Logic score distribution (max 12)
        logic_scores = [result.logic_evaluation.total_score for result in self.evaluation_results]
        logic_ranges = [(0, 3), (4, 6), (7, 9), (10, 12)]
        distribution['logic_score_distribution'] = {
            'score_ranges': ['0-3', '4-6', '7-9', '10-12'],
            'dialogue_count': self._count_scores_in_ranges(logic_scores, logic_ranges)
        }

        # Compensation score distribution (max 6, only for conversations with compensation)
        compensation_results = [result for result in self.evaluation_results if result.compensation_evaluation]
        if compensation_results:
            compensation_scores = [result.compensation_evaluation.total_score for result in compensation_results]
            compensation_ranges = [(0, 2), (3, 4), (5, 6)]
            distribution['compensation_score_distribution'] = {
                'score_ranges': ['0-2', '3-4', '5-6'],
                'dialogue_count': self._count_scores_in_ranges(compensation_scores, compensation_ranges),
                'evaluated_dialogues': len(compensation_results)
            }

        return distribution

    def calculate_ood_statistics(self) -> Dict[str, Any]:
        """Calculate OOD issue statistics"""
        if not self.evaluation_results:
            return {}

        # Filter conversations involving OOD issues
        ood_results = [result for result in self.evaluation_results if result.has_ood_issue]

        if not ood_results:
            return {
                'dialogues_with_ood_issues': 0,
                'ood_issues_ratio': 0.0,
                'ood_issues_scores': {
                    'average': 0.0,
                    'max': 0,
                    'min': 0
                }
            }

        # Calculate OOD issue scores
        ood_scores = []
        for result in ood_results:
            ood_score = result.logic_evaluation.ood_issues.get('ood_issue_recognition', 0)
            if ood_score != -1:  # Only count scores for conversations involving OOD
                ood_scores.append(ood_score)

        return {
            'dialogues_with_ood_issues': len(ood_results),
            'ood_issues_ratio': (len(ood_results) / len(self.evaluation_results)) * 100,
            'ood_issues_scores': {
                'average': sum(ood_scores) / len(ood_scores) if ood_scores else 0.0,
                'max': max(ood_scores) if ood_scores else 0,
                'min': min(ood_scores) if ood_scores else 0
            }
        }

    def calculate_voucher_statistics(self) -> Dict[str, Any]:
        """Calculate voucher statistics"""
        if not self.evaluation_results:
            return {}

        # Filter conversations that generated vouchers
        voucher_results = [result for result in self.evaluation_results if result.has_voucher]

        if not voucher_results:
            return {
                'dialogues_with_voucher': 0,
                'voucher_dialogue_ratio': 0.0,
                'compensation_strategy_scores': {
                    'average': 0.0,
                    'max': 0,
                    'min': 0
                }
            }

        # Calculate compensation strategy scores (from compensation_evaluation)
        compensation_scores = []
        for result in voucher_results:
            if result.compensation_evaluation:
                compensation_score = sum(result.compensation_evaluation.compensation_strategy_adaptation.values())
                compensation_scores.append(compensation_score)

        if not compensation_scores:
            return {
                'dialogues_with_voucher': len(voucher_results),
                'voucher_dialogue_ratio': (len(voucher_results) / len(self.evaluation_results)) * 100,
                'compensation_strategy_scores': {
                    'average': 0.0,
                    'max': 0,
                    'min': 0
                }
            }

        return {
            'dialogues_with_voucher': len(voucher_results),
            'voucher_dialogue_ratio': (len(voucher_results) / len(self.evaluation_results)) * 100,
            'compensation_strategy_scores': {
                'average': sum(compensation_scores) / len(compensation_scores),
                'max': max(compensation_scores),
                'min': min(compensation_scores)
            }
        }

    def calculate_format_statistics(self) -> Dict[str, Any]:
        """Calculate format correctness statistics"""
        if not self.evaluation_results:
            return {}

        # Count format correctness
        format_correct_count = sum(1 for result in self.evaluation_results
                                  if result.format_evaluation.format_correct)

        # Count multiple vouchers
        multiple_vouchers_count = sum(1 for result in self.evaluation_results
                                     if result.format_evaluation.multiple_vouchers)

        # Count format error types
        format_errors = {}
        for result in self.evaluation_results:
            for error_list in result.format_evaluation.format_errors:
                for error in error_list:
                    if error not in format_errors:
                        format_errors[error] = 0
                    format_errors[error] += 1

        # Count total vouchers
        total_vouchers = sum(result.format_evaluation.voucher_count
                           for result in self.evaluation_results)

        # Count total assistant turns
        total_assistant_turns = sum(result.format_evaluation.total_assistant_turns
                                  for result in self.evaluation_results)

        return {
            'format_correctness': {
                'format_correct_dialogues': format_correct_count,
                'format_correct_rate': (format_correct_count / len(self.evaluation_results)) * 100,
                'format_error_dialogues': len(self.evaluation_results) - format_correct_count,
                'format_error_rate': ((len(self.evaluation_results) - format_correct_count) / len(self.evaluation_results)) * 100
            },
            'multiple_voucher_check': {
                'multiple_voucher_dialogues': multiple_vouchers_count,
                'multiple_voucher_rate': (multiple_vouchers_count / len(self.evaluation_results)) * 100,
                'normal_voucher_dialogues': len(self.evaluation_results) - multiple_vouchers_count
            },
            'format_error_distribution': format_errors,
            'voucher_stats': {
                'total_vouchers': total_vouchers,
                'avg_vouchers_per_dialogue': total_vouchers / len(self.evaluation_results) if self.evaluation_results else 0,
                'total_assistant_turns': total_assistant_turns,
                'avg_assistant_turns_per_dialogue': total_assistant_turns / len(self.evaluation_results) if self.evaluation_results else 0
            }
        }

    def get_top_performers(self, top_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get top performing conversations (sorted by each of three dimensions)"""
        if not self.evaluation_results:
            return {}

        top_results = {}

        # Sort by speech score
        sorted_by_speech = sorted(
            self.evaluation_results,
            key=lambda x: x.speech_evaluation.total_score,
            reverse=True
        )
        top_results['best_speech_evaluation'] = []
        for i, result in enumerate(sorted_by_speech[:top_n]):
            top_results['best_speech_evaluation'].append({
                'rank': i + 1,
                'dialogue_id': result.conversation_id,
                'speech_score': result.speech_evaluation.total_score,
                'logic_score': result.logic_evaluation.total_score,
                'compensation_score': result.compensation_evaluation.total_score if result.compensation_evaluation else None
            })

        # Sort by logic score
        sorted_by_logic = sorted(
            self.evaluation_results,
            key=lambda x: x.logic_evaluation.total_score,
            reverse=True
        )
        top_results['best_logic_evaluation'] = []
        for i, result in enumerate(sorted_by_logic[:top_n]):
            top_results['best_logic_evaluation'].append({
                'rank': i + 1,
                'dialogue_id': result.conversation_id,
                'speech_score': result.speech_evaluation.total_score,
                'logic_score': result.logic_evaluation.total_score,
                'compensation_score': result.compensation_evaluation.total_score if result.compensation_evaluation else None
            })

        # Sort by compensation score (only for conversations with compensation)
        compensation_results = [result for result in self.evaluation_results if result.compensation_evaluation]
        if compensation_results:
            sorted_by_compensation = sorted(
                compensation_results,
                key=lambda x: x.compensation_evaluation.total_score,
                reverse=True
            )
            top_results['best_compensation_evaluation'] = []
            for i, result in enumerate(sorted_by_compensation[:top_n]):
                top_results['best_compensation_evaluation'].append({
                    'rank': i + 1,
                    'dialogue_id': result.conversation_id,
                    'speech_score': result.speech_evaluation.total_score,
                    'logic_score': result.logic_evaluation.total_score,
                    'compensation_score': result.compensation_evaluation.total_score
                })

        return top_results

    def get_bottom_performers(self, bottom_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get worst performing conversations (sorted by each of three dimensions)"""
        if not self.evaluation_results:
            return {}

        bottom_results = {}

        # Sort by speech score
        sorted_by_speech = sorted(
            self.evaluation_results,
            key=lambda x: x.speech_evaluation.total_score
        )
        bottom_results['worst_speech_evaluation'] = []
        for i, result in enumerate(sorted_by_speech[:bottom_n]):
            bottom_results['worst_speech_evaluation'].append({
                'rank': i + 1,
                'dialogue_id': result.conversation_id,
                'speech_score': result.speech_evaluation.total_score,
                'logic_score': result.logic_evaluation.total_score,
                'compensation_score': result.compensation_evaluation.total_score if result.compensation_evaluation else None
            })

        # Sort by logic score
        sorted_by_logic = sorted(
            self.evaluation_results,
            key=lambda x: x.logic_evaluation.total_score
        )
        bottom_results['worst_logic_evaluation'] = []
        for i, result in enumerate(sorted_by_logic[:bottom_n]):
            bottom_results['worst_logic_evaluation'].append({
                'rank': i + 1,
                'dialogue_id': result.conversation_id,
                'speech_score': result.speech_evaluation.total_score,
                'logic_score': result.logic_evaluation.total_score,
                'compensation_score': result.compensation_evaluation.total_score if result.compensation_evaluation else None
            })

        # Sort by compensation score (only for conversations with compensation)
        compensation_results = [result for result in self.evaluation_results if result.compensation_evaluation]
        if compensation_results:
            sorted_by_compensation = sorted(
                compensation_results,
                key=lambda x: x.compensation_evaluation.total_score
            )
            bottom_results['worst_compensation_evaluation'] = []
            for i, result in enumerate(sorted_by_compensation[:bottom_n]):
                bottom_results['worst_compensation_evaluation'].append({
                    'rank': i + 1,
                    'dialogue_id': result.conversation_id,
                    'speech_score': result.speech_evaluation.total_score,
                    'logic_score': result.logic_evaluation.total_score,
                    'compensation_score': result.compensation_evaluation.total_score
                })

        return bottom_results

    def generate_report(self) -> Dict[str, Any]:
        """Generate complete report"""
        overall_stats = self.calculate_overall_statistics()

        # If overall statistics are empty, return an empty report structure
        if not overall_stats:
            return {
                'overall_statistics': {},
                'detailed_statistics': {},
                'primary_metric_statistics': {},
                'score_distribution': {},
                'ood_statistics': {},
                'voucher_statistics': {},
                'format_correctness_statistics': {},
                'best_performing_dialogues': [],
                'worst_performing_dialogues': [],
                'summary': {}
            }

        return {
            'overall_statistics': overall_stats,
            'detailed_statistics': self.calculate_detailed_statistics(),
            'primary_metric_statistics': self.calculate_hierarchical_statistics(),
            'score_distribution': self.calculate_score_distribution(),
            'ood_statistics': self.calculate_ood_statistics(),
            'voucher_statistics': self.calculate_voucher_statistics(),
            'format_correctness_statistics': self.calculate_format_statistics(),
            'best_performing_dialogues': self.get_top_performers(),
            'worst_performing_dialogues': self.get_bottom_performers(),
            'summary': self._generate_summary()
        }

    def save_report(self, output_file: str):
        """Save report to file"""
        report = self.generate_report()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def _calculate_median(self, scores: List[float]) -> float:
        """Calculate median"""
        if not scores:
            return 0.0

        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        if n % 2 == 0:
            return (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            return sorted_scores[n//2]

    def _count_scores_in_ranges(self, scores: List[int], ranges: List[Tuple[int, int]]) -> List[int]:
        """Count scores in each range"""
        counts = [0] * len(ranges)

        for score in scores:
            for i, (low, high) in enumerate(ranges):
                if low <= score <= high:
                    counts[i] += 1
                    break

        return counts

    def calculate_hierarchical_statistics(self) -> Dict[str, Any]:
        """Calculate detailed statistics for primary and secondary metrics"""
        if not self.evaluation_results:
            return {}

        # Primary metric: Speech evaluation
        speech_evaluation = {}
        speech_categories = [
            'identity_neutrality', 'dialogue_quality', 'language_adaptability',
            'content_quality', 'communication_effectiveness', 'natural_fluency', 'context_adaptability'
        ]

        for category in speech_categories:
            category_stats = {}
            # Get all secondary metrics under this primary metric
            for result in self.evaluation_results:
                category_data = getattr(result.speech_evaluation, category)
                for sub_category, score in category_data.items():
                    if sub_category not in category_stats:
                        category_stats[sub_category] = []
                    category_stats[sub_category].append(score)

            # Calculate statistics for each secondary metric
            sub_category_stats = {}
            for sub_category, scores in category_stats.items():
                sub_category_stats[sub_category] = {
                    'average': sum(scores) / len(scores),
                    'max': max(scores),
                    'min': min(scores),
                    'median': self._calculate_median(scores)
                }

            speech_evaluation[category] = sub_category_stats

        # Primary metric: Logic evaluation
        logic_evaluation = {}
        logic_categories = [
            'user_profile_recognition', 'business_rule_capability', 'ood_issues'
        ]

        for category in logic_categories:
            category_stats = {}
            # Get all secondary metrics under this primary metric
            for result in self.evaluation_results:
                category_data = getattr(result.logic_evaluation, category)
                for sub_category, score in category_data.items():
                    if sub_category not in category_stats:
                        category_stats[sub_category] = []
                    category_stats[sub_category].append(score)

            # Calculate statistics for each secondary metric
            sub_category_stats = {}
            for sub_category, scores in category_stats.items():
                sub_category_stats[sub_category] = {
                    'average': sum(scores) / len(scores),
                    'max': max(scores),
                    'min': min(scores),
                    'median': self._calculate_median(scores)
                }

            logic_evaluation[category] = sub_category_stats

        # Primary metric: Compensation evaluation (only for conversations with compensation)
        compensation_evaluation = {}
        compensation_results = [result for result in self.evaluation_results if result.compensation_evaluation]

        if compensation_results:
            category_stats = {}
            for result in compensation_results:
                category_data = result.compensation_evaluation.compensation_strategy_adaptation
                for sub_category, score in category_data.items():
                    if sub_category not in category_stats:
                        category_stats[sub_category] = []
                    category_stats[sub_category].append(score)

            # Calculate statistics for each secondary metric
            sub_category_stats = {}
            for sub_category, scores in category_stats.items():
                sub_category_stats[sub_category] = {
                    'average': sum(scores) / len(scores),
                    'max': max(scores),
                    'min': min(scores),
                    'median': self._calculate_median(scores)
                }

            compensation_evaluation['compensation_strategy_adaptation'] = sub_category_stats

        return {
            'primary_metric_statistics': {
                'speech_evaluation': speech_evaluation,
                'logic_evaluation': logic_evaluation,
                'compensation_evaluation': compensation_evaluation
            }
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary information"""
        if not self.evaluation_results:
            return {}

        overall_stats = self.calculate_overall_statistics()

        summary = {
            'total_dialogues': len(self.evaluation_results),
            'speech_evaluation': {
                'avg_score': overall_stats['speech_evaluation_score']['average'],
                'max_possible': overall_stats['speech_evaluation_score']['max_possible']
            },
            'logic_evaluation': {
                'avg_score': overall_stats['logic_evaluation_score']['average'],
                'max_possible': overall_stats['logic_evaluation_score']['max_possible']
            }
        }

        # If compensation evaluation exists, add compensation summary
        if 'compensation_evaluation_score' in overall_stats:
            summary['compensation_evaluation'] = {
                'avg_score': overall_stats['compensation_evaluation_score']['average'],
                'max_possible': overall_stats['compensation_evaluation_score']['max_possible'],
                'evaluated_dialogues': overall_stats['compensation_evaluation_score']['evaluated_dialogues'],
                'evaluated_dialogue_ratio': overall_stats['compensation_evaluation_score']['evaluated_dialogue_ratio']
            }

        return summary
