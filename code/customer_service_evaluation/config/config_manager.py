"""
Config Manager - Manages framework configuration parameters
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM configuration base class"""
    client_type: str = None
    model: str = None
    api_url: str = None
    temperature: float = None
    max_tokens: int = None
    timeout: int = 180
    max_retries: int = 3
    api_key: str = "<your_api_key>"


@dataclass
class UserLLMConfig(LLMConfig):
    """User simulator LLM configuration"""
    client_type: str = None
    model: str = None
    temperature: float = None
    max_tokens: int = None


@dataclass
class CustomerLLMConfig(LLMConfig):
    """Customer service simulator LLM configuration"""
    client_type: str = None
    model: str = None
    api_url: str = None
    temperature: float = None
    max_tokens: int = None


@dataclass
class UserSimulatorConfig:
    """User simulator configuration"""
    prompt_file: str = None
    user_profiles_file: str = None
    system_signals_file: str = None
    core_need_file: str = None


@dataclass
class CustomerServiceConfig:
    """Customer service simulator configuration"""
    prompt_file: str = None


@dataclass
class ParallelConfig:
    """Parallel configuration"""
    max_workers: int = None
    batch_size: int = None
    max_turns_per_conversation: int = None


@dataclass
class EvaluationLLMConfig(LLMConfig):
    """Evaluation LLM configuration"""
    client_type: str = None
    model: str = None
    temperature: float = None
    max_tokens: int = None


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    num_conversations: int = None
    user_category_distribution: Dict[str, int] = None


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = None
    save_format: str = None


class ConfigManager:
    """Config manager"""

    def __init__(self, config_file: str = None):
        """
        Initialize config manager

        Args:
            config_file: Config file path; if None, uses default configuration
        """
        self.config_file = config_file
        self.base_path = Path(__file__).resolve().parent.parent.parent

        # Default configuration
        self.user_llm_config = UserLLMConfig()
        self.customer_llm_config = CustomerLLMConfig()
        self.user_simulator_config = UserSimulatorConfig()
        self.customer_service_config = CustomerServiceConfig()
        self.parallel_config = ParallelConfig()
        self.simulation_config = SimulationConfig()
        self.output_config = OutputConfig()

        # If config file provided, load it
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # Update each configuration
            if 'user_llm' in config_data:
                self._update_from_dict(self.user_llm_config, config_data['user_llm'])
            if 'customer_llm' in config_data:
                self._update_from_dict(self.customer_llm_config, config_data['customer_llm'])
            if 'user_simulator' in config_data:
                self._update_from_dict(self.user_simulator_config, config_data['user_simulator'])
            if 'customer_service' in config_data:
                self._update_from_dict(self.customer_service_config, config_data['customer_service'])
            if 'parallel' in config_data:
                self._update_from_dict(self.parallel_config, config_data['parallel'])
            if 'simulation' in config_data:
                self._update_from_dict(self.simulation_config, config_data['simulation'])
            if 'output' in config_data:
                self._update_from_dict(self.output_config, config_data['output'])

            print(f"YAML config file loaded successfully: {config_file}")

        except Exception as e:
            print(f"Failed to load YAML config file: {e}")
            print("Using default configuration")

    def validate_config(self):
        configs = [
            self.user_llm_config,
            self.customer_llm_config,
            self.user_simulator_config,
            self.customer_service_config,
            self.parallel_config,
            self.simulation_config,
            self.output_config
        ]
        for config in configs:
            for key, value in config.__dict__.items():
                if not key.startswith('_') and value is None:
                    raise ValueError(f"Config item {config.__class__.__name__}.{key} cannot be empty")
        return True

    def _update_from_dict(self, config_obj, config_dict: Dict[str, Any]):
        """Update config object from dictionary"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    def _to_dict(self, config_obj) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        return {key: value for key, value in config_obj.__dict__.items()
                if not key.startswith('_')}

    def get_full_path(self, relative_path: str) -> str:
        """Get full path"""
        return str(self.base_path / relative_path)

    def apply_command_line_overrides(self, overrides: Dict[str, Any]):
        """Apply command line parameter overrides

        Args:
            overrides: Command line parameter override dictionary
        """
        for key, value in overrides.items():
            # Handle hydra-style parameters (section.key=value)
            if '.' in key:
                section, subkey = key.split('.', 1)

                if section == 'simulation':
                    if hasattr(self.simulation_config, subkey):
                        setattr(self.simulation_config, subkey, value)
                elif section == 'user_llm':
                    if hasattr(self.user_llm_config, subkey):
                        setattr(self.user_llm_config, subkey, value)
                elif section == 'customer_llm':
                    if hasattr(self.customer_llm_config, subkey):
                        setattr(self.customer_llm_config, subkey, value)
                elif section == 'user_simulator':
                    if hasattr(self.user_simulator_config, subkey):
                        setattr(self.user_simulator_config, subkey, value)
                elif section == 'customer_service':
                    if hasattr(self.customer_service_config, subkey):
                        setattr(self.customer_service_config, subkey, value)
                elif section == 'parallel':
                    if hasattr(self.parallel_config, subkey):
                        setattr(self.parallel_config, subkey, value)
                elif section == 'output':
                    if hasattr(self.output_config, subkey):
                        setattr(self.output_config, subkey, value)
            else:
                # Backward compatibility: handle old parameter format
                if key == 'num_conversations':
                    self.simulation_config.num_conversations = value
                elif key == 'user_category_distribution':
                    self.simulation_config.user_category_distribution = value
                elif key == 'user_model':
                    self.user_llm_config.model = value
                elif key == 'user_temperature':
                    self.user_llm_config.temperature = value
                elif key == 'user_max_tokens':
                    self.user_llm_config.max_tokens = value
                elif key == 'user_client_type':
                    self.user_llm_config.client_type = value
                elif key == 'user_api_url':
                    self.user_llm_config.api_url = value
                elif key == 'customer_model':
                    self.customer_llm_config.model = value
                elif key == 'customer_temperature':
                    self.customer_llm_config.temperature = value
                elif key == 'customer_max_tokens':
                    self.customer_llm_config.max_tokens = value
                elif key == 'customer_client_type':
                    self.customer_llm_config.client_type = value
                elif key == 'customer_api_url':
                    self.customer_llm_config.api_url = value
                elif key == 'user_prompt_file':
                    self.user_simulator_config.prompt_file = value
                elif key == 'user_profiles_file':
                    self.user_simulator_config.user_profiles_file = value
                elif key == 'system_signals_file':
                    self.user_simulator_config.system_signals_file = value
                elif key == 'core_need_file':
                    self.user_simulator_config.core_need_file = value
                elif key == 'customer_prompt_file':
                    self.customer_service_config.prompt_file = value
                elif key == 'max_workers':
                    self.parallel_config.max_workers = value
                elif key == 'batch_size':
                    self.parallel_config.batch_size = value
                elif key == 'max_turns':
                    self.parallel_config.max_turns_per_conversation = value
                elif key == 'output_dir':
                    self.output_config.output_dir = value
                elif key == 'save_format':
                    self.output_config.save_format = value

    def print_config_summary(self):
        """Print configuration summary"""
        print("\nConfiguration Summary:")
        print("=" * 60)
        print(f"User Simulator LLM Config:")
        print(f"  Type: {self.user_llm_config.client_type}")
        print(f"  Model: {self.user_llm_config.model}")
        print(f"  Temperature: {self.user_llm_config.temperature}")
        print(f"  API URL: {self.user_llm_config.api_url}")
        print(f"\nCustomer Service Simulator LLM Config:")
        print(f"  Type: {self.customer_llm_config.client_type}")
        print(f"  Model: {self.customer_llm_config.model}")
        print(f"  Temperature: {self.customer_llm_config.temperature}")
        print(f"  API URL: {self.customer_llm_config.api_url}")
        print(f"\nUser Simulator Config:")
        print(f"  Prompt file: {self.user_simulator_config.prompt_file}")
        print(f"  User profiles: {self.user_simulator_config.user_profiles_file}")
        print(f"  System signals: {self.user_simulator_config.system_signals_file}")
        print(f"  Core demands: {self.user_simulator_config.core_need_file}")
        print(f"\nCustomer Service Simulator Config:")
        print(f"  CS prompt: {self.customer_service_config.prompt_file}")
        print(f"\nParallel Config:")
        print(f"  Workers: {self.parallel_config.max_workers}")
        print(f"  Batch size: {self.parallel_config.batch_size}")
        print(f"  Max turns: {self.parallel_config.max_turns_per_conversation}")
        print(f"\nSimulation Config:")
        print(f"  Conversation count: {self.simulation_config.num_conversations}")
        print(f"  User category distribution: {self.simulation_config.user_category_distribution}")
        print(f"\nOutput Config:")
        print(f"  Output directory: {self.output_config.output_dir}")
        print("=" * 60)
