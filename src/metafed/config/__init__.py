"""
Configuration management system for MetaFed-FL.

This module provides a centralized configuration system that supports
YAML files, environment variables, and command-line arguments.
"""

import yaml
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    
    # General settings
    seed: int = 42
    device: str = "auto"
    output_dir: str = "./results"
    log_level: str = "INFO"
    
    # Federated learning settings
    algorithm: str = "fedavg"
    num_clients: int = 50
    clients_per_round: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Data settings
    dataset: str = "mnist"
    non_iid_alpha: float = 0.5
    data_dir: str = "./data"
    
    # Model settings
    model_name: str = "simplecnn"
    input_channels: int = 1
    num_classes: int = 10
    pretrained: bool = False
    
    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Orchestration settings
    orchestrator: str = "random"
    orchestrator_params: Dict[str, Any] = field(default_factory=dict)
    
    # Green computing settings
    green_aware: bool = False
    carbon_tracking: bool = True
    carbon_region: str = "US"
    
    # Privacy settings
    privacy: str = "none"  # none, differential, homomorphic
    epsilon: float = 1.0
    delta: float = 1e-5
    privacy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation settings
    eval_frequency: int = 5
    save_checkpoints: bool = True
    save_results: bool = True
    
    # Experimental settings
    experiment_name: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: str = ""


@dataclass 
class MNISTConfig(BaseConfig):
    """Configuration for MNIST experiments."""
    
    dataset: str = "mnist"
    input_channels: int = 1
    num_classes: int = 10
    model_name: str = "simplecnn"
    
    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"mnist_{self.algorithm}"


@dataclass
class CIFAR10Config(BaseConfig):
    """Configuration for CIFAR-10 experiments."""
    
    dataset: str = "cifar10"
    input_channels: int = 3
    num_classes: int = 10
    model_name: str = "simplecnn"
    num_rounds: int = 200  # CIFAR-10 typically needs more rounds
    
    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"cifar10_{self.algorithm}"


class ConfigManager:
    """
    Configuration manager for MetaFed-FL experiments.
    
    Supports loading from YAML files, environment variables,
    and provides validation and merging capabilities.
    """
    
    def __init__(self):
        self.config_cache = {}
    
    def load_config(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        config_class: type = BaseConfig
    ) -> BaseConfig:
        """
        Load configuration from various sources.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Dictionary with configuration values
            config_class: Configuration class to use
            
        Returns:
            Configuration instance
        """
        # Start with default config
        config_data = {}
        
        # Load from YAML file
        if config_path and os.path.exists(config_path):
            config_data.update(self._load_yaml(config_path))
            logger.info(f"Loaded configuration from {config_path}")
        
        # Merge with provided dictionary
        if config_dict:
            config_data.update(config_dict)
        
        # Load environment variables
        env_config = self._load_from_env()
        config_data.update(env_config)
        
        # Create config instance
        try:
            config = config_class(**config_data)
            self._validate_config(config)
            return config
        except TypeError as e:
            logger.error(f"Invalid configuration parameters: {e}")
            # Fall back to default config
            return config_class()
    
    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except Exception as e:
            logger.error(f"Failed to load YAML config from {config_path}: {e}")
            return {}
    
    def _load_from_env(self, prefix: str = "METAFED_") -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                else:
                    try:
                        config[config_key] = float(value)
                    except ValueError:
                        config[config_key] = value
        
        if config:
            logger.info(f"Loaded {len(config)} settings from environment variables")
        
        return config
    
    def _validate_config(self, config: BaseConfig) -> None:
        """Validate configuration values."""
        errors = []
        
        # Validate algorithm
        valid_algorithms = ["fedavg", "fedprox", "scaffold"]
        if config.algorithm not in valid_algorithms:
            errors.append(f"Invalid algorithm: {config.algorithm}. Must be one of {valid_algorithms}")
        
        # Validate dataset
        valid_datasets = ["mnist", "cifar10"]
        if config.dataset not in valid_datasets:
            errors.append(f"Invalid dataset: {config.dataset}. Must be one of {valid_datasets}")
        
        # Validate numeric ranges
        if config.num_clients <= 0:
            errors.append("num_clients must be positive")
        
        if config.clients_per_round <= 0 or config.clients_per_round > config.num_clients:
            errors.append("clients_per_round must be positive and <= num_clients")
        
        if config.num_rounds <= 0:
            errors.append("num_rounds must be positive")
        
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Validate privacy settings
        if config.privacy == "differential":
            if config.epsilon <= 0:
                errors.append("epsilon must be positive for differential privacy")
            if config.delta <= 0 or config.delta >= 1:
                errors.append("delta must be in (0, 1) for differential privacy")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def save_config(self, config: BaseConfig, output_path: str) -> None:
        """Save configuration to YAML file."""
        try:
            # Convert dataclass to dict
            import dataclasses
            config_dict = dataclasses.asdict(config)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def merge_configs(self, base_config: BaseConfig, override_config: Dict[str, Any]) -> BaseConfig:
        """Merge two configurations with override taking precedence."""
        import dataclasses
        
        # Convert base config to dict
        base_dict = dataclasses.asdict(base_config)
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge(base_dict, override_config)
        
        # Create new config instance
        config_class = type(base_config)
        return config_class(**merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config_for_dataset(self, dataset: str) -> type:
        """Get appropriate config class for dataset."""
        config_mapping = {
            "mnist": MNISTConfig,
            "cifar10": CIFAR10Config
        }
        return config_mapping.get(dataset.lower(), BaseConfig)


# Global config manager instance
config_manager = ConfigManager()


def load_config(config_path: Optional[str] = None, **kwargs) -> BaseConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to YAML config file
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration instance
    """
    return config_manager.load_config(config_path=config_path, config_dict=kwargs)


def create_experiment_config(
    algorithm: str,
    dataset: str,
    **kwargs
) -> BaseConfig:
    """
    Create experiment configuration with common patterns.
    
    Args:
        algorithm: Federated learning algorithm
        dataset: Dataset name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration instance
    """
    # Get appropriate config class
    config_class = config_manager.get_config_for_dataset(dataset)
    
    # Set algorithm-specific defaults
    config_dict = {"algorithm": algorithm, "dataset": dataset}
    
    if algorithm == "fedprox":
        config_dict.setdefault("algorithm_params", {})["mu"] = kwargs.pop("fedprox_mu", 0.01)
    
    # Merge with provided kwargs
    config_dict.update(kwargs)
    
    return config_class(**config_dict)