"""
Parallel Evaluation Manager - Supports evaluating multiple conversations simultaneously
"""

import time
import json
import os
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from .evaluation_result import EvaluationResult, SpeechEvaluationResult, LogicEvaluationResult, CompensationEvaluationResult, FormatEvaluationResult
from evaluators.speech_evaluator import SpeechEvaluator
from evaluators.logic_evaluator import LogicEvaluator
from evaluators.compensation_evaluator import CompensationEvaluator
from evaluators.format_evaluator import FormatEvaluator


@dataclass
class ParallelEvaluationConfig:
    """Parallel evaluation configuration"""
    max_workers: int = 5
    batch_size: int = 10
    progress_callback: Optional[Callable] = None


class ParallelEvaluationManager:
    """Parallel evaluation manager"""

    def __init__(self,
                 config_manager,
                 config: ParallelEvaluationConfig = None):
        """
        Initialize parallel evaluation manager

        Args:
            config_manager: Config manager instance
            config: Parallel configuration
        """
        self.config_manager = config_manager
        self.config = config or ParallelEvaluationConfig()

        # Create evaluator instances
        eval_config = config_manager.evaluation_config
        self.speech_evaluator = SpeechEvaluator(
            llm_config=eval_config.llm_config,
            prompt_file=config_manager.get_full_path(eval_config.speech_evaluation_prompt_file)
        )
        self.logic_evaluator = LogicEvaluator(
            llm_config=eval_config.llm_config,
            prompt_file=config_manager.get_full_path(eval_config.logic_evaluation_prompt_file)
        )
        self.compensation_evaluator = CompensationEvaluator(
            llm_config=eval_config.llm_config,
            prompt_file=config_manager.get_full_path(eval_config.compensation_evaluation_prompt_file)
        )
        self.format_evaluator = FormatEvaluator()

        # Statistics
        self.total_conversations = 0
        self.completed_evaluations = 0
        self.failed_evaluations = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def evaluate_conversations(self,
                             conversations: List[Dict[str, Any]],
                             output_dir: str = None,
                             model_name: str = None) -> List[EvaluationResult]:
        """
        Evaluate multiple conversations in parallel

        Args:
            conversations: List of conversations, each containing conversation_id and conversation_history
            output_dir: Output directory path; if provided, results are saved
            model_name: Model name, used for file name identification

        Returns:
            List of evaluation results
        """
        self.total_conversations = len(conversations)
        self.completed_evaluations = 0
        self.failed_evaluations = 0
        self.start_time = time.time()

        print(f"Starting parallel evaluation")
        print(f"Conversation count: {self.total_conversations}")
        print(f"Parallel workers: {self.config.max_workers}")
        print(f"Batch size: {self.config.batch_size}")
        if output_dir:
            print(f"Results save directory: {output_dir}")
        if model_name:
            print(f"Evaluation model: {model_name}")
        print("=" * 60)

        results = []

        # Create results file (if output directory specified)
        results_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if model_name:
                # Sanitize special characters in model name
                safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                results_file = os.path.join(output_dir, f"evaluation_results_{safe_model_name}_{timestamp}.jsonl")
            else:
                results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.jsonl")
            print(f"Results file: {results_file}")

        # Process in batches
        for batch_start in range(0, self.total_conversations, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, self.total_conversations)
            batch_results = self._process_batch(
                conversations[batch_start:batch_end],
                batch_start
            )
            results.extend(batch_results)

            # Save batch results to file (append mode)
            if results_file:
                self._append_batch_results(batch_results, results_file)

        self.end_time = time.time()
        self._print_summary()

        return results

    def _process_batch(self,
                      conversations: List[Dict[str, Any]],
                      batch_offset: int) -> List[EvaluationResult]:
        """Process a batch of evaluations"""
        batch_size = len(conversations)
        batch_results = [None] * batch_size

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_index = {}
            for i, conversation in enumerate(conversations):
                conversation_id = conversation.get('conversation_id', f"conv_{batch_offset + i + 1}")
                conversation_history = conversation.get('conversation_history', [])

                # Submit evaluation task
                future = executor.submit(
                    self._evaluate_single_conversation,
                    conversation_id,
                    conversation_history
                )
                future_to_index[future] = i

            # Collect results
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                completed_count += 1

                try:
                    result = future.result()
                    batch_results[index] = result

                    # Update statistics
                    if result:
                        self.completed_evaluations += 1
                    else:
                        self.failed_evaluations += 1

                    # Call progress callback
                    if self.config.progress_callback:
                        self.config.progress_callback(
                            completed=self.completed_evaluations + self.failed_evaluations,
                            total=self.total_conversations,
                            result=result
                        )

                    # Display progress
                    progress = self._create_progress_bar(
                        self.completed_evaluations + self.failed_evaluations,
                        self.total_conversations
                    )
                    print(f"\rEvaluating... {progress}", end="", flush=True)

                except Exception as e:
                    print(f"\nConversation evaluation failed: {e}")
                    self.failed_evaluations += 1
                    batch_results[index] = None

        print(f"\rBatch complete! {self._create_progress_bar(batch_size, batch_size)}")
        return [result for result in batch_results if result is not None]

    def _evaluate_single_conversation(self, conversation_id: str, conversation_history: List[Dict[str, Any]]) -> Optional[EvaluationResult]:
        """Evaluate a single conversation"""
        try:
            # Run speech evaluation
            speech_result = self.speech_evaluator.evaluate(conversation_history)

            # Run logic evaluation
            logic_result = self.logic_evaluator.evaluate(conversation_history)

            # Run compensation evaluation (only for conversations with compensation)
            compensation_result = self.compensation_evaluator.evaluate(conversation_history)

            # Run format evaluation
            format_result = self.format_evaluator.evaluate_conversation(conversation_history)

            # Create format evaluation result object
            format_eval = FormatEvaluationResult(
                format_correct=format_result['format_correct'],
                format_errors=[error['errors'] for error in format_result['format_errors']],
                voucher_count=format_result['voucher_count'],
                multiple_vouchers=format_result['multiple_vouchers'],
                total_assistant_turns=format_result['total_assistant_turns']
            )

            # Create complete evaluation result
            evaluation_result = EvaluationResult(
                conversation_id=conversation_id,
                speech_evaluation=speech_result,
                logic_evaluation=logic_result,
                compensation_evaluation=compensation_result,
                format_evaluation=format_eval,
                conversation_history=conversation_history,
                metadata={
                    'total_turns': len(conversation_history),
                    'evaluation_time': time.time(),
                    'has_compensation': compensation_result is not None
                }
            )

            return evaluation_result

        except Exception as e:
            print(f"Evaluation of conversation {conversation_id} failed: {e}")
            return None

    def _append_batch_results(self, batch_results: List[EvaluationResult], results_file: str):
        """Append batch results to jsonl file"""
        try:
            # Write to jsonl file in append mode
            with open(results_file, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    if result:
                        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')

            # Batch result statistics
            successful = len(batch_results)
            failed = 0

            print(f"Batch results appended to file")
            print(f"   Successful: {successful}, Failed: {failed}")

        except Exception as e:
            print(f"Failed to append batch results: {e}")

    def _create_progress_bar(self, completed: int, total: int, bar_length: int = 30) -> str:
        """Create progress bar"""
        if total == 0:
            return "[>" + " " * (bar_length - 1) + "] 0/0 (0.0%)"

        progress = completed / total
        filled_length = int(bar_length * progress)
        bar = "=" * filled_length + ">" + " " * (bar_length - filled_length - 1)
        percentage = progress * 100
        return f"[{bar}] {completed}/{total} ({percentage:.1f}%)"

    def _print_summary(self):
        """Print summary information"""
        if not self.start_time or not self.end_time:
            return

        duration = self.end_time - self.start_time
        success_rate = (self.completed_evaluations / self.total_conversations * 100) if self.total_conversations > 0 else 0

        print("\n" + "=" * 60)
        print("Parallel evaluation complete")
        print("=" * 60)
        print(f"Total conversations: {self.total_conversations}")
        print(f"Successful evaluations: {self.completed_evaluations}")
        print(f"Failed evaluations: {self.failed_evaluations}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total duration: {duration:.2f}s")
        if self.total_conversations > 0:
            print(f"Average per conversation: {duration/self.total_conversations:.2f}s")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        success_rate = (self.completed_evaluations / self.total_conversations * 100) if self.total_conversations > 0 else 0

        return {
            'total_conversations': self.total_conversations,
            'completed_evaluations': self.completed_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'success_rate': success_rate,
            'duration': self.end_time - self.start_time if self.start_time and self.end_time else 0
        }
