"""
Parallel Dialogue Manager - Supports running multiple conversations simultaneously
"""

import time
import json
import os
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from .conversation_manager import ConversationManager


@dataclass
class ParallelConfig:
    """Parallel configuration"""
    max_workers: int = 5
    batch_size: int = 10
    progress_callback: Optional[Callable] = None


class ParallelDialogueManager:
    """Parallel dialogue manager"""

    def __init__(self,
                 user_simulator,
                 customer_service_simulator,
                 config: ParallelConfig = None):
        """
        Initialize parallel dialogue manager

        Args:
            user_simulator: User simulator instance
            customer_service_simulator: Customer service simulator instance
            config: Parallel configuration
        """
        self.user_simulator = user_simulator
        self.customer_service_simulator = customer_service_simulator
        self.config = config or ParallelConfig()

        # Statistics
        self.total_conversations = 0
        self.completed_conversations = 0
        self.failed_conversations = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def run_parallel_conversations(self,
                                 user_profiles: List[Dict[str, Any]],
                                 system_signals_list: List[Dict[str, Any]],
                                 max_turns: int = 10,
                                 output_dir: str = None,
                                 core_demands: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Run multiple conversations in parallel

        Args:
            user_profiles: List of user profiles
            system_signals_list: List of system signals
            max_turns: Maximum turns per conversation
            output_dir: Output directory path; if provided, results are saved
            core_demands: List of core demands; if None, auto-generated

        Returns:
            List of conversation results
        """
        if len(user_profiles) != len(system_signals_list):
            raise ValueError("User profiles and system signals lists must have the same length")

        self.total_conversations = len(user_profiles)
        self.completed_conversations = 0
        self.failed_conversations = 0
        self.start_time = time.time()

        print(f"Starting parallel dialogue simulation")
        print(f"Conversation count: {self.total_conversations}")
        print(f"Max turns: {max_turns}")
        print(f"Parallel workers: {self.config.max_workers}")
        print(f"Batch size: {self.config.batch_size}")
        if output_dir:
            print(f"Results save directory: {output_dir}")
        print("=" * 60)

        # If core demands not provided, auto-generate
        if core_demands is None:
            core_demands = []
            for user_profile in user_profiles:
                user_category = user_profile.get('category_info', {}).get('category_id', '1')
                core_demand = self.user_simulator.get_random_core_demand(user_category)
                core_demands.append(core_demand)

        results = []

        # Create results file (if output directory specified)
        results_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"customer_service_conversations_{timestamp}.jsonl")
            print(f"Results file: {results_file}")

        # Process in batches
        for batch_start in range(0, self.total_conversations, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, self.total_conversations)
            batch_results = self._process_batch(
                user_profiles[batch_start:batch_end],
                system_signals_list[batch_start:batch_end],
                core_demands[batch_start:batch_end],
                max_turns,
                batch_start
            )
            results.extend(batch_results)

            # Save batch results to file (append mode)
            if results_file:
                self._append_batch_results(batch_results, results_file)

        self.end_time = time.time()
        self._print_summary()
        self._print_customer_insights(results)

        return results

    def _process_batch(self,
                      user_profiles: List[Dict[str, Any]],
                      system_signals_list: List[Dict[str, Any]],
                      core_demands: List[Dict[str, Any]],
                      max_turns: int,
                      batch_offset: int) -> List[Dict[str, Any]]:
        """Process a batch of conversations"""
        batch_size = len(user_profiles)
        batch_results = [None] * batch_size

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all conversation tasks
            future_to_index = {}
            for i, (user_profile, system_signals, core_demand) in enumerate(zip(user_profiles, system_signals_list, core_demands)):
                conversation_id = f"conv_{batch_offset + i + 1}"

                # Create conversation manager
                conversation_manager = ConversationManager(
                    conversation_id=conversation_id,
                    user_simulator=self.user_simulator,
                    customer_service_simulator=self.customer_service_simulator,
                    user_profile=user_profile,
                    system_signals=system_signals,
                    max_turns=max_turns,
                    core_demand=core_demand
                )

                # Submit task
                future = executor.submit(conversation_manager.run_to_completion)
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
                    if result.get('success', False):
                        self.completed_conversations += 1
                    else:
                        self.failed_conversations += 1

                    # Display progress
                    progress = self._create_progress_bar(
                        self.completed_conversations + self.failed_conversations,
                        self.total_conversations
                    )
                    print(f"\rProcessing... {progress}", end="", flush=True)

                except Exception as e:
                    print(f"\nConversation processing failed: {e}")
                    self.failed_conversations += 1
                    batch_results[index] = {
                        'conversation_id': f"conv_{batch_offset + index + 1}",
                        'success': False,
                        'error': str(e)
                    }

        print(f"\rBatch complete! {self._create_progress_bar(batch_size, batch_size)}")
        return batch_results

    def _append_batch_results(self, batch_results: List[Dict[str, Any]], results_file: str):
        """Append batch results to jsonl file"""
        try:
            # Write to jsonl file in append mode
            with open(results_file, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            # Batch result statistics
            successful = sum(1 for result in batch_results if result.get('success', False))
            failed = len(batch_results) - successful

            print(f"Batch results appended to file")
            print(f"   Successful: {successful}, Failed: {failed}")

        except Exception as e:
            print(f"Failed to append batch results: {e}")

    def _save_all_results(self, results: List[Dict[str, Any]], output_dir: str):
        """Save all results to a single file"""
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Create results file name
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"customer_service_conversations_{timestamp}.jsonl")

            # Save all results in jsonl format
            with open(results_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            # Result statistics
            successful = sum(1 for result in results if result.get('success', False))
            failed = len(results) - successful

            print(f"\nAll results saved to: {results_file}")
            print(f"Total conversations: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")

        except Exception as e:
            print(f"\nFailed to save results: {e}")

    def _save_batch_results(self, batch_results: List[Dict[str, Any]], batch_start: int, batch_end: int, output_dir: str):
        """Save batch results to file"""
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Create batch file name
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            batch_file = os.path.join(output_dir, f"batch_{batch_start+1}_{batch_end}_{timestamp}.json")

            # Prepare batch data
            batch_data = {
                'batch_info': {
                    'batch_start': batch_start + 1,
                    'batch_end': batch_end,
                    'batch_size': len(batch_results),
                    'timestamp': timestamp,
                    'total_conversations_so_far': batch_end
                },
                'results': batch_results
            }

            # Save to file
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)

            # Batch result statistics
            successful = sum(1 for result in batch_results if result.get('success', False))
            failed = len(batch_results) - successful

            print(f"\nBatch {batch_start+1}-{batch_end} saved: {batch_file}")
            print(f"   Successful: {successful}, Failed: {failed}")

        except Exception as e:
            print(f"\nFailed to save batch: {e}")

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
        success_rate = (self.completed_conversations / self.total_conversations * 100) if self.total_conversations > 0 else 0

        print("\n" + "=" * 60)
        print("Parallel dialogue simulation complete")
        print("=" * 60)
        print(f"Total conversations: {self.total_conversations}")
        print(f"Successful: {self.completed_conversations}")
        print(f"Failed: {self.failed_conversations}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total duration: {duration:.2f}s")
        if self.total_conversations > 0:
            print(f"Average per conversation: {duration/self.total_conversations:.2f}s")

    def _print_customer_insights(self, results: List[Dict[str, Any]]):
        """Print customer service insights"""
        print("\n" + "=" * 60)
        print("Customer Service Insight Analysis")
        print("=" * 60)

        # Collect all agent actions
        actions = {}
        think_samples = []

        for result in results:
            if not result.get('success', False):
                continue

            conversation_history = result.get('conversation_history', [])
            for turn in conversation_history:
                if turn.get('role') == 'assistant' and turn.get('metadata'):
                    metadata = turn.get('metadata', {})
                    action = metadata.get('action', 'chat')
                    think = metadata.get('think', '')

                    # Count actions
                    actions[action] = actions.get(action, 0) + 1

                    # Collect thinking process samples (first 3)
                    if think and len(think_samples) < 3:
                        think_samples.append({
                            'conversation_id': result.get('conversation_id', ''),
                            'think': think[:200] + '...' if len(think) > 200 else think
                        })

        # Print action statistics
        print(f"\nAgent action statistics:")
        total_actions = sum(actions.values())
        for action, count in actions.items():
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"  {action}: {count} times ({percentage:.1f}%)")

        # Print thinking process samples
        if think_samples:
            print(f"\nAgent thinking process samples:")
            for i, sample in enumerate(think_samples, 1):
                print(f"\n  Sample {i} (conversation {sample['conversation_id']}):")
                print(f"    {sample['think']}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        success_rate = (self.completed_conversations / self.total_conversations * 100) if self.total_conversations > 0 else 0

        return {
            'total_conversations': self.total_conversations,
            'completed_conversations': self.completed_conversations,
            'failed_conversations': self.failed_conversations,
            'success_rate': success_rate,
            'duration': self.end_time - self.start_time if self.start_time and self.end_time else 0
        }
