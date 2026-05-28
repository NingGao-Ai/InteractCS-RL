"""
Customer Service Simulator Evaluation Main Script
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.evaluation_config_manager import EvaluationConfigManager
from core.parallel_evaluation_manager import ParallelEvaluationManager, ParallelEvaluationConfig
from core.evaluation_statistics import EvaluationStatistics


class EvaluationMain:
    """Evaluation main program"""

    def __init__(self, config_file: str = "config/config_yaml/config_evaluation.yaml"):
        """
        Initialize evaluation main program

        Args:
            config_file: Evaluation configuration file path
        """
        self.config_manager = EvaluationConfigManager(config_file)

    def load_conversations(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load conversation data

        Args:
            input_file: Input file path (jsonl format)

        Returns:
            List of conversations
        """
        conversations = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conversation = json.loads(line)
                        conversations.append(conversation)

            print(f"Successfully loaded {len(conversations)} conversations")
            return conversations

        except Exception as e:
            print(f"Failed to load conversation data: {e}")
            sys.exit(1)

    def run_evaluation(self, input_file: str = None, output_dir: str = None, max_conversations: int = None, model_name: str = None) -> None:
        """
        Run evaluation

        Args:
            input_file: Input file path (command-line args take priority, then config file)
            output_dir: Output directory (command-line args take priority, then config file)
            max_conversations: Maximum number of conversations to evaluate (command-line args take priority, then config file)
            model_name: Model name (used for file name identification)
        """
        # Determine input file path (command-line args take priority)

        self.config_manager.print_config_summary()
        final_input_file = input_file or self.config_manager.evaluation_config.input_file
        if not final_input_file:
            print("Input file path not specified")
            sys.exit(1)

        # Load conversation data
        conversations = self.load_conversations(final_input_file)

        # Determine max conversation count (command-line args take priority)
        final_max_conversations = max_conversations or self.config_manager.evaluation_config.max_conversations
        if final_max_conversations and final_max_conversations < len(conversations):
            conversations = conversations[:final_max_conversations]
            print(f"Limiting evaluation to {final_max_conversations} conversations")

        # Create parallel evaluation manager
        eval_config = self.config_manager.evaluation_config
        parallel_config = ParallelEvaluationConfig(
            max_workers=eval_config.parallel_config.max_workers,
            batch_size=eval_config.parallel_config.batch_size
        )

        evaluation_manager = ParallelEvaluationManager(
            config_manager=self.config_manager,
            config=parallel_config
        )

        # Set output directory (command-line args take priority)
        if not output_dir:
            output_dir = self.config_manager.evaluation_config.output_dir

        # If output directory not specified in config, use default output config
        if not output_dir:
            output_dir = self.config_manager.get_full_path(eval_config.output_config.evaluation_dir)

        # Get model name (from config or parameter)
        if not model_name:
            model_name = eval_config.llm_config.model
            # If model path is long, take only the last part
            if '/' in model_name:
                model_name = model_name.split('/')[-1]

        # Clean special characters from model name
        model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

        # Run evaluation
        print("\n" + "=" * 60)
        print("Starting dialogue evaluation")
        print("=" * 60)

        evaluation_results = evaluation_manager.evaluate_conversations(
            conversations=conversations,
            output_dir=output_dir,
            model_name=model_name
        )

        # Generate statistics report
        print("\n" + "=" * 60)
        print("Generating statistics report")
        print("=" * 60)

        statistics = EvaluationStatistics(evaluation_results)
        report = statistics.generate_report()

        # Save detailed report (includes model name)
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"evaluation_report_{model_name}_{timestamp}.json")
        statistics.save_report(report_file)
        print(f"\nDetailed report saved to: {report_file}")


def parse_hydra_args():
    """Parse hydra-style command-line arguments"""
    overrides = {}

    for arg in sys.argv[1:]:
        if arg.startswith('--config'):
            # Config file argument handled separately
            continue
        elif '=' in arg:
            # Handle key=value format arguments
            key, value = arg.split('=', 1)

            # Remove possible -- prefix
            if key.startswith('--'):
                key = key[2:]

            # Convert data types
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif value.startswith('{') and value.endswith('}'):
                # JSON-format dictionary
                try:
                    value = json.loads(value)
                except:
                    pass

            overrides[key] = value

    return overrides


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Customer Service Simulator Evaluation System')
    parser.add_argument('--config', '-c', default=Path(__file__).parent.joinpath('config/config_yaml/config_evaluation.yaml'), help='Evaluation configuration file path')

    # Parse hydra-style arguments
    hydra_overrides = parse_hydra_args()

    # Parse traditional arguments (backward compatible)
    args, unknown_args = parser.parse_known_args()

    # Build command-line parameter overrides
    overrides = {}

    # Add hydra-style arguments
    overrides.update(hydra_overrides)

    # Run evaluation
    evaluator = EvaluationMain(args.config)

    # Apply command-line parameter overrides
    if overrides:
        evaluator.config_manager.apply_command_line_overrides(overrides)

    # Get parameters from config
    eval_config = evaluator.config_manager.evaluation_config
    input_file = eval_config.input_file
    output_dir = eval_config.output_dir
    max_conversations = eval_config.max_conversations

    # Check if input file exists
    if not input_file:
        print("Input file path not specified")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    evaluator.run_evaluation(
        input_file=input_file,
        output_dir=output_dir,
        max_conversations=max_conversations
    )


if __name__ == "__main__":
    main()
