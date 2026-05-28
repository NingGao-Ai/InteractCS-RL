"""
Evaluation Statistics Tool - for aggregating and analyzing evaluation results
"""
import json
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationStatistics:
    """Evaluation Statistics Class - for generating evaluation reports"""

    def __init__(self, evaluation_results_file: str):
        """
        Initialize evaluation statistics

        Args:
            evaluation_results_file: Evaluation results file path (jsonl format)
        """
        self.results_file = evaluation_results_file
        self.results = self._load_results()
    
    def _load_results(self) -> List[Dict[str, Any]]:
        """Load evaluation results"""
        results = []
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result = json.loads(line)
                        results.append(result)
            logger.info(f"Loaded {len(results)} evaluation results")
            return results
        except Exception as e:
            logger.error(f"Failed to load evaluation results: {e}")
            return []
    
    def calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics"""
        if not self.results:
            return {}
        
        # Extract scores for each evaluation type
        speech_scores = [r["scores"].get("speech_score", 0) for r in self.results 
                        if "scores" in r and "speech_score" in r["scores"]]
        logic_scores = [r["scores"].get("logic_score", 0) for r in self.results 
                       if "scores" in r and "logic_score" in r["scores"]]
        
        # Compensation evaluation (only count conversations with compensation)
        compensation_results = [r for r in self.results 
                               if r.get("evaluations", {}).get("compensation_evaluation") is not None]
        compensation_scores = [r["scores"].get("compensation_score", 0) for r in compensation_results 
                              if "scores" in r and "compensation_score" in r["scores"]]
        
        stats = {
            'total_conversations': len(self.results),
            'speech_evaluation_scores': {
                'average': sum(speech_scores) / len(speech_scores) if speech_scores else 0,
                'max': max(speech_scores) if speech_scores else 0,
                'min': min(speech_scores) if speech_scores else 0,
                'median': self._calculate_median(speech_scores),
                'max_possible': 28
            },
            'logic_evaluation_scores': {
                'average': sum(logic_scores) / len(logic_scores) if logic_scores else 0,
                'max': max(logic_scores) if logic_scores else 0,
                'min': min(logic_scores) if logic_scores else 0,
                'median': self._calculate_median(logic_scores),
                'max_possible': 12
            }
        }

        # If compensation evaluation exists, add compensation statistics
        if compensation_scores:
            stats['compensation_evaluation_scores'] = {
                'average': sum(compensation_scores) / len(compensation_scores),
                'max': max(compensation_scores),
                'min': min(compensation_scores),
                'median': self._calculate_median(compensation_scores),
                'max_possible': 6,
                'evaluated_conversations': len(compensation_results),
                'evaluated_conversation_ratio': (len(compensation_results) / len(self.results)) * 100
            }
        
        return stats
    
    def calculate_detailed_statistics(self) -> Dict[str, Any]:
        """Calculate detailed statistics"""
        if not self.results:
            return {}
        
        # Speech evaluation detailed statistics
        speech_categories = [
            'identity_neutrality', 'dialogue_quality', 'language_adaptability',
            'content_quality', 'communication_effectiveness', 'natural_fluency', 'context_adaptability'
        ]
        
        speech_stats = {}
        for category in speech_categories:
            category_scores = []
            for result in self.results:
                speech_eval = result.get("evaluations", {}).get("speech_evaluation", {})
                if category in speech_eval and isinstance(speech_eval[category], dict):
                    category_score = sum(v for k, v in speech_eval[category].items() 
                                       if k != "reason" and isinstance(v, (int, float)))
                    category_scores.append(category_score)
            
            if category_scores:
                speech_stats[category] = {
                    'average': sum(category_scores) / len(category_scores),
                    'max': max(category_scores),
                    'min': min(category_scores),
                    'median': self._calculate_median(category_scores)
                }

        # Logic evaluation detailed statistics
        logic_categories = [
            'user_profile_recognition', 'business_rule_capability', 'ood_issues'
        ]
        
        logic_stats = {}
        for category in logic_categories:
            category_scores = []
            for result in self.results:
                logic_eval = result.get("evaluations", {}).get("logic_evaluation", {})
                if category in logic_eval and isinstance(logic_eval[category], dict):
                    category_score = sum(v for k, v in logic_eval[category].items() 
                                       if k != "reason" and isinstance(v, (int, float)))
                    category_scores.append(category_score)
            
            if category_scores:
                logic_stats[category] = {
                    'average': sum(category_scores) / len(category_scores),
                    'max': max(category_scores),
                    'min': min(category_scores),
                    'median': self._calculate_median(category_scores)
                }

        # Compensation evaluation detailed statistics
        compensation_stats = {}
        compensation_results = [r for r in self.results 
                               if r.get("evaluations", {}).get("compensation_evaluation") is not None]
        
        if compensation_results:
            category_scores = []
            for result in compensation_results:
                comp_eval = result.get("evaluations", {}).get("compensation_evaluation", {})
                if "compensation_strategy_adaptation" in comp_eval and isinstance(comp_eval["compensation_strategy_adaptation"], dict):
                    category_score = sum(v for k, v in comp_eval["compensation_strategy_adaptation"].items() 
                                       if k != "reason" and isinstance(v, (int, float)))
                    category_scores.append(category_score)
            
            if category_scores:
                compensation_stats['compensation_strategy_adaptation'] = {
                    'average': sum(category_scores) / len(category_scores),
                    'max': max(category_scores),
                    'min': min(category_scores),
                    'median': self._calculate_median(category_scores),
                    'evaluated_conversations': len(compensation_results)
                }

        return {
            'speech_evaluation': speech_stats,
            'logic_evaluation': logic_stats,
            'compensation_evaluation': compensation_stats
        }
    
    def calculate_score_distribution(self) -> Dict[str, Any]:
        """Calculate score distribution"""
        if not self.results:
            return {}
        
        distribution = {}
        
        # Speech score distribution (max 28)
        speech_scores = [r["scores"].get("speech_score", 0) for r in self.results
                        if "scores" in r and "speech_score" in r["scores"]]
        if speech_scores:
            speech_ranges = [(0, 7), (8, 14), (15, 21), (22, 28)]
            distribution['speech_score_distribution'] = {
                'score_ranges': ['0-7', '8-14', '15-21', '22-28'],
                'conversation_counts': self._count_scores_in_ranges(speech_scores, speech_ranges)
            }

        # Logic score distribution (max 12)
        logic_scores = [r["scores"].get("logic_score", 0) for r in self.results
                       if "scores" in r and "logic_score" in r["scores"]]
        if logic_scores:
            logic_ranges = [(0, 3), (4, 6), (7, 9), (10, 12)]
            distribution['logic_score_distribution'] = {
                'score_ranges': ['0-3', '4-6', '7-9', '10-12'],
                'conversation_counts': self._count_scores_in_ranges(logic_scores, logic_ranges)
            }

        # Compensation score distribution (max 6, only conversations with compensation)
        compensation_results = [r for r in self.results 
                               if r.get("evaluations", {}).get("compensation_evaluation") is not None]
        if compensation_results:
            compensation_scores = [r["scores"].get("compensation_score", 0) for r in compensation_results 
                                  if "scores" in r and "compensation_score" in r["scores"]]
            if compensation_scores:
                compensation_ranges = [(0, 2), (3, 4), (5, 6)]
                distribution['compensation_score_distribution'] = {
                    'score_ranges': ['0-2', '3-4', '5-6'],
                    'conversation_counts': self._count_scores_in_ranges(compensation_scores, compensation_ranges),
                    'total_evaluated_conversations': len(compensation_results)
                }
        
        return distribution
    
    def calculate_evaluation_success_statistics(self) -> Dict[str, Any]:
        """Calculate evaluation success/failure statistics"""
        if not self.results:
            return {}
        
        # Count success/failure for each evaluation type
        speech_success = sum(1 for r in self.results 
                            if r.get("evaluations", {}).get("speech_evaluation_success") == True)
        speech_fail = sum(1 for r in self.results 
                         if r.get("evaluations", {}).get("speech_evaluation_success") == False)
        
        logic_success = sum(1 for r in self.results 
                           if r.get("evaluations", {}).get("logic_evaluation_success") == True)
        logic_fail = sum(1 for r in self.results 
                        if r.get("evaluations", {}).get("logic_evaluation_success") == False)
        
        compensation_success = sum(1 for r in self.results 
                                  if r.get("evaluations", {}).get("compensation_evaluation_success") == True)
        compensation_fail = sum(1 for r in self.results 
                               if r.get("evaluations", {}).get("compensation_evaluation_success") == False)
        
        return {
            'speech_evaluation': {
                'success_count': speech_success,
                'failure_count': speech_fail,
                'success_rate': (speech_success / (speech_success + speech_fail) * 100) if (speech_success + speech_fail) > 0 else 0,
                'failure_rate': (speech_fail / (speech_success + speech_fail) * 100) if (speech_success + speech_fail) > 0 else 0
            },
            'logic_evaluation': {
                'success_count': logic_success,
                'failure_count': logic_fail,
                'success_rate': (logic_success / (logic_success + logic_fail) * 100) if (logic_success + logic_fail) > 0 else 0,
                'failure_rate': (logic_fail / (logic_success + logic_fail) * 100) if (logic_success + logic_fail) > 0 else 0
            },
            'compensation_evaluation': {
                'success_count': compensation_success,
                'failure_count': compensation_fail,
                'success_rate': (compensation_success / (compensation_success + compensation_fail) * 100) if (compensation_success + compensation_fail) > 0 else 0,
                'failure_rate': (compensation_fail / (compensation_success + compensation_fail) * 100) if (compensation_success + compensation_fail) > 0 else 0
            }
        }
    
    def calculate_format_statistics(self) -> Dict[str, Any]:
        """Calculate format correctness statistics"""
        if not self.results:
            return {}
        
        # Count format-correct conversations
        format_correct_count = sum(1 for r in self.results 
                                  if r.get("evaluations", {}).get("format_check", {}).get("format_correct") == True)
        
        # Count multiple vouchers
        multiple_vouchers_count = sum(1 for r in self.results 
                                     if r.get("evaluations", {}).get("format_check", {}).get("multiple_vouchers") == True)
        
        # Count format error types
        format_errors = {}
        for result in self.results:
            format_check = result.get("evaluations", {}).get("format_check", {})
            error_list = format_check.get("format_errors", [])
            for error in error_list:
                if error not in format_errors:
                    format_errors[error] = 0
                format_errors[error] += 1
        
        # Count total vouchers
        total_vouchers = sum(result.get("evaluations", {}).get("format_check", {}).get("voucher_count", 0)
                           for result in self.results)
        
        # Count total assistant turns
        total_assistant_turns = sum(result.get("evaluations", {}).get("format_check", {}).get("total_assistant_turns", 0)
                                  for result in self.results)
        
        format_error_count = len(self.results) - format_correct_count
        
        return {
            'format_correctness': {
                'format_correct_conversations': format_correct_count,
                'format_correct_rate': (format_correct_count / len(self.results)) * 100 if self.results else 0,
                'format_error_conversations': format_error_count,
                'format_error_rate': (format_error_count / len(self.results)) * 100 if self.results else 0
            },
            'multiple_voucher_check': {
                'multiple_voucher_conversations': multiple_vouchers_count,
                'multiple_voucher_rate': (multiple_vouchers_count / len(self.results)) * 100 if self.results else 0,
                'normal_voucher_conversations': len(self.results) - multiple_vouchers_count
            },
            'format_error_distribution': format_errors,
            'voucher_statistics': {
                'total_vouchers': total_vouchers,
                'avg_vouchers_per_conversation': total_vouchers / len(self.results) if self.results else 0,
                'total_assistant_turns': total_assistant_turns,
                'avg_assistant_turns_per_conversation': total_assistant_turns / len(self.results) if self.results else 0
            }
        }
    
    def calculate_voucher_statistics(self) -> Dict[str, Any]:
        """Calculate voucher statistics"""
        if not self.results:
            return {}
        
        # Filter conversations that issued vouchers (those with compensation evaluation)
        voucher_results = [r for r in self.results 
                          if r.get("evaluations", {}).get("compensation_evaluation") is not None]
        
        if not voucher_results:
            return {
                'voucher_conversations': 0,
                'voucher_conversation_ratio': 0.0,
                'compensation_strategy_scores': {
                    'average': 0.0,
                    'max': 0,
                    'min': 0
                }
            }

        # Calculate compensation strategy scores
        compensation_scores = []
        for result in voucher_results:
            comp_eval = result.get("evaluations", {}).get("compensation_evaluation", {})
            if "compensation_strategy_adaptation" in comp_eval and isinstance(comp_eval["compensation_strategy_adaptation"], dict):
                comp_score = sum(v for k, v in comp_eval["compensation_strategy_adaptation"].items() 
                               if k != "reason" and isinstance(v, (int, float)))
                compensation_scores.append(comp_score)
        
        return {
            'voucher_conversations': len(voucher_results),
            'voucher_conversation_ratio': (len(voucher_results) / len(self.results)) * 100,
            'compensation_strategy_scores': {
                'average': sum(compensation_scores) / len(compensation_scores) if compensation_scores else 0.0,
                'max': max(compensation_scores) if compensation_scores else 0,
                'min': min(compensation_scores) if compensation_scores else 0
            }
        }
    
    def calculate_satisfaction_statistics(self) -> Dict[str, Any]:
        """Calculate user satisfaction statistics (only format-correct conversations)"""
        if not self.results:
            return {}
        
        # Extract satisfaction data from summary, only for format-correct conversations
        satisfaction_scores = []
        for result in self.results:
            # Check if format is correct
            format_check = result.get("evaluations", {}).get("format_check", {})
            
            
            satisfaction_stat = result.get("satisfaction", {})
            if satisfaction_stat.get("has_satisfaction"):
                satisfaction = satisfaction_stat.get("satisfaction")
                if format_check.get("format_correct") != True:
                    satisfaction =1
                # Format incorrect, skip this conversation
                if satisfaction is not None:
                    satisfaction_scores.append(satisfaction)
        
        if not satisfaction_scores:
            return {
                'conversations_with_satisfaction': 0,
                'satisfaction_conversation_ratio': 0.0,
                'satisfaction_scores': {
                    'average': 0.0,
                    'max': 0,
                    'min': 0,
                    'median': 0.0
                },
                'satisfaction_distribution': {}
            }

        # Count satisfaction distribution (0-10 scale)
        satisfaction_distribution = {}
        for score in satisfaction_scores:
            score_int = int(score)
            if score_int not in satisfaction_distribution:
                satisfaction_distribution[score_int] = 0
            satisfaction_distribution[score_int] += 1
        
        satisfaction_distribution = dict(sorted(satisfaction_distribution.items()))
        
        return {
            'conversations_with_satisfaction': len(satisfaction_scores),
            'satisfaction_conversation_ratio': (len(satisfaction_scores) / len(self.results) * 100) if self.results else 0,
            'satisfaction_scores': {
                'average': sum(satisfaction_scores) / len(satisfaction_scores),
                'max': max(satisfaction_scores),
                'min': min(satisfaction_scores),
                'median': self._calculate_median(satisfaction_scores)
            },
            'satisfaction_distribution': satisfaction_distribution,
            'total_satisfaction_ratings': len(satisfaction_scores)
        }
    
    def calculate_transfer_statistics(self) -> Dict[str, Any]:
        """Calculate transfer-to-human rate statistics"""
        if not self.results:
            return {}
        
        # Count conversations transferred to human (end_reason is "assistant_ended")
        transfer_count = 0
        for result in self.results:
            summary = result.get("summary", {})
            end_reason = summary.get("end_reason")
            if end_reason == "assistant_ended":
                transfer_count += 1
        
        total_count = len(self.results)
        transfer_rate = (transfer_count / total_count * 100) if total_count > 0 else 0.0
        
        return {
            'transfer_conversations': transfer_count,
            'transfer_rate': transfer_rate,
            'total_conversations': total_count
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete report"""
        overall_stats = self.calculate_overall_statistics()
        
        if not overall_stats:
            return {
                'overall_statistics': {},
                'satisfaction_statistics': {},
                'voucher_statistics': {},
                'transfer_statistics': {},
                'detailed_statistics': {},
                'score_distribution': {},
                'evaluation_success_statistics': {},
                'format_correctness_statistics': {},
                'summary': {}
            }

        return {
            'overall_statistics': overall_stats,
            'satisfaction_statistics': self.calculate_satisfaction_statistics(),
            'voucher_statistics': self.calculate_voucher_statistics(),
            'transfer_statistics': self.calculate_transfer_statistics(),
            'detailed_statistics': self.calculate_detailed_statistics(),
            'score_distribution': self.calculate_score_distribution(),
            'evaluation_success_statistics': self.calculate_evaluation_success_statistics(),
            'format_correctness_statistics': self.calculate_format_statistics(),
            'summary': self._generate_summary()
        }
    
    def save_report(self, output_file: str) -> None:
        """Save evaluation report"""
        report = self.generate_report()
        
        try:
            # Ensure output directory exists
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Evaluation report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
    
    def print_summary(self) -> None:
        """Print evaluation summary"""
        report = self.generate_report()

        print("\n" + "=" * 70)
        print("Evaluation Statistics Report")
        print("=" * 70)

        # Overall statistics
        overall = report.get('overall_statistics', {})
        print(f"\n[Overall Statistics]")
        print(f"  Total conversations: {overall.get('total_conversations', 0)}")

        # Speech evaluation
        if 'speech_evaluation_scores' in overall:
            speech = overall['speech_evaluation_scores']
            print(f"\n[Speech Evaluation]")
            print(f"  Average: {speech['average']:.2f} / {speech['max_possible']}")
            print(f"  Highest: {speech['max']}, Lowest: {speech['min']}")
            print(f"  Median: {speech['median']:.2f}")

        # Logic evaluation
        if 'logic_evaluation_scores' in overall:
            logic = overall['logic_evaluation_scores']
            print(f"\n[Logic Evaluation]")
            print(f"  Average: {logic['average']:.2f} / {logic['max_possible']}")
            print(f"  Highest: {logic['max']}, Lowest: {logic['min']}")
            print(f"  Median: {logic['median']:.2f}")

        # Compensation evaluation
        if 'compensation_evaluation_scores' in overall:
            comp = overall['compensation_evaluation_scores']
            print(f"\n[Compensation Evaluation]")
            print(f"  Average: {comp['average']:.2f} / {comp['max_possible']}")
            print(f"  Highest: {comp['max']}, Lowest: {comp['min']}")
            print(f"  Median: {comp['median']:.2f}")
            print(f"  Evaluated conversations: {comp['evaluated_conversations']} ({comp['evaluated_conversation_ratio']:.2f}%)")

        # Evaluation success rate
        eval_success = report.get('evaluation_success_statistics', {})
        if eval_success:
            print(f"\n[Evaluation Success Rate]")
            if 'speech_evaluation' in eval_success:
                speech_success = eval_success['speech_evaluation']
                print(f"  Speech: success {speech_success['success_count']} ({speech_success['success_rate']:.2f}%), failure {speech_success['failure_count']} ({speech_success['failure_rate']:.2f}%)")
            if 'logic_evaluation' in eval_success:
                logic_success = eval_success['logic_evaluation']
                print(f"  Logic: success {logic_success['success_count']} ({logic_success['success_rate']:.2f}%), failure {logic_success['failure_count']} ({logic_success['failure_rate']:.2f}%)")
            if 'compensation_evaluation' in eval_success:
                comp_success = eval_success['compensation_evaluation']
                print(f"  Compensation: success {comp_success['success_count']} ({comp_success['success_rate']:.2f}%), failure {comp_success['failure_count']} ({comp_success['failure_rate']:.2f}%)")

        # Format correctness
        format_stats = report.get('format_correctness_statistics', {}).get('format_correctness', {})
        if format_stats:
            print(f"\n[Format Correctness]")
            print(f"  Correct: {format_stats['format_correct_conversations']} ({format_stats['format_correct_rate']:.2f}%)")
            print(f"  Incorrect: {format_stats['format_error_conversations']} ({format_stats['format_error_rate']:.2f}%)")

        # Voucher statistics
        voucher_stats = report.get('voucher_statistics', {})
        if voucher_stats:
            print(f"\n[Voucher Statistics]")
            print(f"  Voucher conversations: {voucher_stats['voucher_conversations']} ({voucher_stats['voucher_conversation_ratio']:.2f}%)")
            comp_score = voucher_stats['compensation_strategy_scores']
            print(f"  Compensation strategy avg score: {comp_score['average']:.2f}")

        # User satisfaction statistics
        satisfaction_stats = report.get('satisfaction_statistics', {})
        if satisfaction_stats and satisfaction_stats.get('total_satisfaction_ratings', 0) > 0:
            print(f"\n[User Satisfaction]")
            print(f"  Conversations with satisfaction data: {satisfaction_stats['conversations_with_satisfaction']} ({satisfaction_stats['satisfaction_conversation_ratio']:.2f}%)")
            sat_score = satisfaction_stats['satisfaction_scores']
            print(f"  Average satisfaction: {sat_score['average']:.2f} / 10")
            print(f"  Highest: {sat_score['max']}, Lowest: {sat_score['min']}")
            print(f"  Median: {sat_score['median']:.2f}")

            dist = satisfaction_stats.get('satisfaction_distribution', {})
            if dist:
                print(f"  Distribution: ", end="")
                dist_str = ", ".join([f"score {score}: {count}" for score, count in sorted(dist.items())])
                print(dist_str)

        # Transfer-to-human statistics
        transfer_stats = report.get('transfer_statistics', {})
        if transfer_stats:
            print(f"\n[Transfer to Human]")
            print(f"  Transfer conversations: {transfer_stats['transfer_conversations']}")
            print(f"  Transfer rate: {transfer_stats['transfer_rate']:.2f}%")

        print("\n" + "=" * 70)
    
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
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary information"""
        if not self.results:
            return {}
        
        overall_stats = self.calculate_overall_statistics()
        
        summary = {
            'total_conversations': len(self.results),
            'speech_evaluation': {
                'average_score': overall_stats.get('speech_evaluation_scores', {}).get('average', 0),
                'max_possible': overall_stats.get('speech_evaluation_scores', {}).get('max_possible', 28)
            },
            'logic_evaluation': {
                'average_score': overall_stats.get('logic_evaluation_scores', {}).get('average', 0),
                'max_possible': overall_stats.get('logic_evaluation_scores', {}).get('max_possible', 12)
            }
        }

        # If compensation evaluation exists, add compensation summary
        if 'compensation_evaluation_scores' in overall_stats:
            summary['compensation_evaluation'] = {
                'average_score': overall_stats['compensation_evaluation_scores']['average'],
                'max_possible': overall_stats['compensation_evaluation_scores']['max_possible'],
                'evaluated_conversations': overall_stats['compensation_evaluation_scores']['evaluated_conversations'],
                'evaluated_conversation_ratio': overall_stats['compensation_evaluation_scores']['evaluated_conversation_ratio']
            }
        
        return summary