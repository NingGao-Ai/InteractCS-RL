"""
Evaluation Config Manager - Manages evaluation-specific configuration parameters
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class EvaluationLLMConfig:
    """Evaluation LLM configuration"""
    client_type: str = "openai"
    model: str = "gpt-4.1"
    api_url: str = "<your_api_url>"
    api_key: str = "<your_api_key>"
    temperature: float = 0.1  # Lower temperature for evaluation consistency
    max_tokens: int = 4000
    timeout: int = 120
    max_retries: int = 3


@dataclass
class EvaluationParallelConfig:
    """Evaluation parallel configuration"""
    max_workers: int = 5
    batch_size: int = 10
    progress_callback: Optional[Callable] = None


@dataclass
class EvaluationOutputConfig:
    """Evaluation output configuration"""
    evaluation_dir: str = "data/customer_service_evaluation/evaluation_results"
    save_format: str = "jsonl"  # "json" or "jsonl"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    input_file: str = ""  # Input conversation file path
    output_dir: str = ""  # Output directory path
    max_conversations: Optional[int] = None  # Maximum number of conversations to evaluate
    speech_evaluation_prompt_file: str = "code/prompt/evaluation-speech.txt"
    logic_evaluation_prompt_file: str = "code/prompt/evaluation-logic.txt"
    compensation_evaluation_prompt_file: str = "code/prompt/evaluation-compensation.txt"
    llm_config: EvaluationLLMConfig = field(default_factory=EvaluationLLMConfig)
    parallel_config: EvaluationParallelConfig = field(default_factory=EvaluationParallelConfig)
    output_config: EvaluationOutputConfig = field(default_factory=EvaluationOutputConfig)


class EvaluationConfigManager:
    """Evaluation config manager"""

    def __init__(self, config_file: str = None):
        """
        Initialize evaluation config manager

        Args:
            config_file: Evaluation config file path; if None, uses default configuration
        """
        self.config_file = config_file
        self.base_path = Path(__file__).resolve().parent.parent.parent

        # Default configuration
        self.evaluation_config = EvaluationConfig()

        # If config file provided, load it
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # Update evaluation configuration
            if 'input_file' in config_data:
                self.evaluation_config.input_file = config_data['input_file']
            if 'output_dir' in config_data:
                self.evaluation_config.output_dir = config_data['output_dir']
            if 'max_conversations' in config_data:
                self.evaluation_config.max_conversations = config_data['max_conversations']
            if 'speech_evaluation_prompt_file' in config_data:
                self.evaluation_config.speech_evaluation_prompt_file = config_data['speech_evaluation_prompt_file']
            if 'logic_evaluation_prompt_file' in config_data:
                self.evaluation_config.logic_evaluation_prompt_file = config_data['logic_evaluation_prompt_file']
            if 'compensation_evaluation_prompt_file' in config_data:
                self.evaluation_config.compensation_evaluation_prompt_file = config_data['compensation_evaluation_prompt_file']

            # Update LLM configuration
            if 'llm' in config_data:
                self._update_from_dict(self.evaluation_config.llm_config, config_data['llm'])

            # Update parallel configuration
            if 'parallel' in config_data:
                self._update_from_dict(self.evaluation_config.parallel_config, config_data['parallel'])

            # Update output configuration
            if 'output' in config_data:
                self._update_from_dict(self.evaluation_config.output_config, config_data['output'])

            print(f"YAML evaluation config file loaded successfully: {config_file}")

        except Exception as e:
            print(f"Failed to load YAML evaluation config file: {e}")
            print("Using default evaluation configuration")

    def save_config(self, config_file: str):
        """Save configuration to YAML file

        Args:
            config_file: Config file path
        """
        try:
            config_data = {
                'input_file': self.evaluation_config.input_file,
                'output_dir': self.evaluation_config.output_dir,
                'max_conversations': self.evaluation_config.max_conversations,
                'speech_evaluation_prompt_file': self.evaluation_config.speech_evaluation_prompt_file,
                'logic_evaluation_prompt_file': self.evaluation_config.logic_evaluation_prompt_file,
                'compensation_evaluation_prompt_file': self.evaluation_config.compensation_evaluation_prompt_file,
                'llm': self._to_dict(self.evaluation_config.llm_config),
                'parallel': self._to_dict(self.evaluation_config.parallel_config),
                'output': self._to_dict(self.evaluation_config.output_config)
            }

            os.makedirs(os.path.dirname(config_file), exist_ok=True)

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True, indent=2)

            print(f"YAML evaluation config file saved successfully: {config_file}")

        except Exception as e:
            print(f"Failed to save YAML evaluation config file: {e}")

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

                if section == 'input':
                    if hasattr(self.evaluation_config, subkey):
                        setattr(self.evaluation_config, subkey, value)
                elif section == 'llm':
                    if hasattr(self.evaluation_config.llm_config, subkey):
                        setattr(self.evaluation_config.llm_config, subkey, value)
                elif section == 'prompts':
                    if hasattr(self.evaluation_config, subkey):
                        setattr(self.evaluation_config, subkey, value)
                elif section == 'parallel':
                    if hasattr(self.evaluation_config.parallel_config, subkey):
                        setattr(self.evaluation_config.parallel_config, subkey, value)
                elif section == 'output':
                    if hasattr(self.evaluation_config.output_config, subkey):
                        setattr(self.evaluation_config.output_config, subkey, value)
            else:
                # Backward compatibility: handle old parameter format
                if key == 'input_file':
                    self.evaluation_config.input_file = value
                elif key == 'output_dir':
                    self.evaluation_config.output_dir = value
                elif key == 'max_conversations':
                    self.evaluation_config.max_conversations = value
                elif key == 'speech_evaluation_prompt_file':
                    self.evaluation_config.speech_evaluation_prompt_file = value
                elif key == 'logic_evaluation_prompt_file':
                    self.evaluation_config.logic_evaluation_prompt_file = value
                elif key == 'compensation_evaluation_prompt_file':
                    self.evaluation_config.compensation_evaluation_prompt_file = value
                elif key == 'evaluation_model':
                    self.evaluation_config.llm_config.model = value
                elif key == 'evaluation_temperature':
                    self.evaluation_config.llm_config.temperature = value
                elif key == 'evaluation_max_tokens':
                    self.evaluation_config.llm_config.max_tokens = value
                elif key == 'evaluation_client_type':
                    self.evaluation_config.llm_config.client_type = value
                elif key == 'evaluation_api_url':
                    self.evaluation_config.llm_config.api_url = value
                elif key == 'max_workers':
                    self.evaluation_config.parallel_config.max_workers = value
                elif key == 'batch_size':
                    self.evaluation_config.parallel_config.batch_size = value
                elif key == 'evaluation_dir':
                    self.evaluation_config.output_config.evaluation_dir = value
                elif key == 'save_format':
                    self.evaluation_config.output_config.save_format = value

    def validate_config(self) -> bool:
        """Validate if configuration is valid"""
        errors = []

        # Check if prompt files exist
        required_files = [
            self.evaluation_config.speech_evaluation_prompt_file,
            self.evaluation_config.logic_evaluation_prompt_file,
            self.evaluation_config.compensation_evaluation_prompt_file
        ]

        for file_path in required_files:
            full_path = self.get_full_path(file_path)
            if not os.path.exists(full_path):
                errors.append(f"Evaluation prompt file not found: {full_path}")

        # Check if output directory is writable
        output_dir = self.get_full_path(self.evaluation_config.output_config.evaluation_dir)
        try:
            os.makedirs(output_dir, exist_ok=True)
            test_file = os.path.join(output_dir, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            errors.append(f"Evaluation output directory is not writable: {output_dir} - {e}")

        if errors:
            print("Evaluation config validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("Evaluation config validation passed")
            return True

    def print_config_summary(self):
        """Print configuration summary"""
        print("\nEvaluation Configuration Summary:")
        print("=" * 60)
        print(f"Input file: {self.evaluation_config.input_file}")
        print(f"Output directory: {self.evaluation_config.output_dir}")
        print(f"Max conversations: {self.evaluation_config.max_conversations}")
        print(f"Speech evaluation prompt file: {self.evaluation_config.speech_evaluation_prompt_file}")
        print(f"Logic evaluation prompt file: {self.evaluation_config.logic_evaluation_prompt_file}")
        print(f"Compensation evaluation prompt file: {self.evaluation_config.compensation_evaluation_prompt_file}")
        print(f"\nEvaluation LLM Config:")
        print(f"  Type: {self.evaluation_config.llm_config.client_type}")
        print(f"  Model: {self.evaluation_config.llm_config.model}")
        print(f"  Temperature: {self.evaluation_config.llm_config.temperature}")
        print(f"  API URL: {self.evaluation_config.llm_config.api_url}")
        print(f"\nEvaluation Parallel Config:")
        print(f"  Workers: {self.evaluation_config.parallel_config.max_workers}")
        print(f"  Batch size: {self.evaluation_config.parallel_config.batch_size}")
        print(f"\nEvaluation Output Config:")
        print(f"  Output directory: {self.evaluation_config.output_config.evaluation_dir}")
        print(f"  Save format: {self.evaluation_config.output_config.save_format}")
        print("=" * 60)
