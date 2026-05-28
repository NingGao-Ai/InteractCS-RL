"""
Customer Service Evaluation Framework - Main Entry Point
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.parallel_dialogue_manager import ParallelDialogueManager
from simulators.user_simulator import UserSimulator
from simulators.customer_service_simulator import CustomerServiceSimulator
from clients.llm_client_factory import LLMClientFactory
from config.config_manager import ConfigManager


def run_evaluation(config_file: str = None, **kwargs):
    """
    Run customer service dialogue evaluation

    Args:
        config_file: Configuration file path
        **kwargs: Command-line parameter overrides
    """
    print("Customer Service Dialogue Evaluation Framework")
    print("=" * 60)

    # Create config manager
    config = ConfigManager(config_file)

    # Apply command-line parameter overrides
    if kwargs:
        config.apply_command_line_overrides(kwargs)

    config.print_config_summary()

    # Validate configuration
    if not config.validate_config():
        print("Configuration validation failed. Please check the config file.")
        return

    # Create LLM clients
    print("\nInitializing LLM clients...")

    # User simulator LLM client
    if config.user_llm_config.client_type == "openai":
        user_llm_client = LLMClientFactory.create_openai_client(
            api_url=config.user_llm_config.api_url,
            api_key=config.user_llm_config.api_key,
            model=config.user_llm_config.model,
            temperature=config.user_llm_config.temperature,
            max_tokens=config.user_llm_config.max_tokens,
            timeout=config.user_llm_config.timeout,
            max_retries=config.user_llm_config.max_retries
        )
    else:
        user_llm_client = LLMClientFactory.create_vllm_client(
            api_url=config.user_llm_config.api_url,
            model=config.user_llm_config.model,
            temperature=config.user_llm_config.temperature,
            max_tokens=config.user_llm_config.max_tokens,
            timeout=config.user_llm_config.timeout,
            max_retries=config.user_llm_config.max_retries
        )

    # Customer service simulator LLM client
    if config.customer_llm_config.client_type == "openai":
        customer_llm_client = LLMClientFactory.create_openai_client(
            api_url=config.customer_llm_config.api_url,
            api_key=config.customer_llm_config.api_key,
            model=config.customer_llm_config.model,
            temperature=config.customer_llm_config.temperature,
            max_tokens=config.customer_llm_config.max_tokens,
            max_retries=config.customer_llm_config.max_retries
        )
    else:
        customer_llm_client = LLMClientFactory.create_vllm_client(
            api_url=config.customer_llm_config.api_url,
            model=config.customer_llm_config.model,
            temperature=config.customer_llm_config.temperature,
            max_tokens=config.customer_llm_config.max_tokens,
            timeout=config.customer_llm_config.timeout,
            max_retries=config.customer_llm_config.max_retries
        )

    # Create simulators
    print("\nInitializing simulators...")

    user_simulator = UserSimulator(
        llm_client=user_llm_client,
        prompt_file=config.get_full_path(config.user_simulator_config.prompt_file),
        user_profiles_file=config.get_full_path(config.user_simulator_config.user_profiles_file),
        system_signals_file=config.get_full_path(config.user_simulator_config.system_signals_file),
        core_need_file=config.get_full_path(config.user_simulator_config.core_need_file)
    )

    customer_service_simulator = CustomerServiceSimulator(
        llm_client=customer_llm_client,
        prompt_file=config.get_full_path(config.customer_service_config.prompt_file)
    )

    # Prepare test data
    print(f"\nPreparing {config.simulation_config.num_conversations} conversation data entries...")

    user_profiles = []
    system_signals_list = []

    # Generate data based on configured user category distribution
    for category, count in config.simulation_config.user_category_distribution.items():
        for _ in range(count):
            user_profiles.append(user_simulator.get_random_user_profile(category))
            system_signals_list.append(user_simulator.get_random_system_signals(1))

    # Verify that the distribution total matches the configured conversation count
    total_distributed = sum(config.simulation_config.user_category_distribution.values())
    if total_distributed !=  config.simulation_config.num_conversations:
        raise ValueError(f"Configured conversation count {config.simulation_config.num_conversations} does not match the total user category distribution {total_distributed}")

    # Create parallel dialogue manager
    print("\nInitializing parallel dialogue manager...")

    parallel_config = config.parallel_config

    parallel_manager = ParallelDialogueManager(
        user_simulator=user_simulator,
        customer_service_simulator=customer_service_simulator,
        config=parallel_config
    )

    # Run parallel dialogues
    print(f"\nStarting parallel dialogue simulation...")

    start_time = time.time()
    results = parallel_manager.run_parallel_conversations(
        user_profiles=user_profiles,
        system_signals_list=system_signals_list,
        max_turns=config.parallel_config.max_turns_per_conversation,
        output_dir=config.get_full_path(config.output_config.output_dir)
    )
    total_duration = time.time() - start_time

    # Generate statistics report
    print(f"\nGenerating statistics report...")

    stats = parallel_manager.get_statistics()

    # Get results file path (saved by the parallel manager)
    output_dir = config.get_full_path(config.output_config.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"customer_service_conversations_{timestamp}.jsonl")

    report = {
        'evaluation_summary': {
            'total_conversations': stats['total_conversations'],
            'completed_conversations': stats['completed_conversations'],
            'failed_conversations': stats['failed_conversations'],
            'success_rate': stats['success_rate'],
            'total_duration': total_duration,
            'average_duration_per_conversation': total_duration / config.simulation_config.num_conversations if config.simulation_config.num_conversations > 0 else 0
        },
        'configuration': {
            'num_conversations': config.simulation_config.num_conversations,
            'user_category_distribution': config.simulation_config.user_category_distribution,
            'user_llm_config': config._to_dict(config.user_llm_config),
            'customer_llm_config': config._to_dict(config.customer_llm_config),
            'parallel_config': config._to_dict(config.parallel_config),
            'simulation_config': config._to_dict(config.simulation_config)
        },
        'results_file': results_file,
        'timestamp': timestamp
    }

    # Save report
    report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation complete!")
    print(f"Results file: {results_file}")
    print(f"Report file: {report_file}")
    print(f"\nEvaluation summary:")
    print(f"   Total conversations: {stats['total_conversations']}")
    print(f"   Successful: {stats['completed_conversations']}")
    print(f"   Failed: {stats['failed_conversations']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Total duration: {total_duration:.2f}s")
    print(f"   Average per conversation: {total_duration/config.simulation_config.num_conversations:.2f}s")

    return results, report


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
    parser = argparse.ArgumentParser(description='Customer Service Dialogue Evaluation Framework')
    parser.add_argument('--config', type=str, help='Configuration file path', default=Path(__file__).parent.joinpath('config/config_yaml/config.yaml'))

    # Parse hydra-style arguments
    hydra_overrides = parse_hydra_args()

    # Parse traditional arguments (backward compatible)
    args, unknown_args = parser.parse_known_args()

    # Build command-line parameter overrides
    overrides = {}

    # Add hydra-style arguments
    overrides.update(hydra_overrides)
    # Run evaluation
    run_evaluation(
        config_file=args.config,
        **overrides
    )


if __name__ == "__main__":
    main()
