import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """
    Centralized configuration management system.
    Loads, validates, and provides access to all configuration parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._create_directories()
    
    def _load_config(self) -> DictConfig:
        """
        Load configuration from YAML file using OmegaConf.
        
        Returns:
            DictConfig object with all configuration parameters
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at: {self.config_path}"
            )
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = OmegaConf.create(config_dict)
        
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: DictConfig) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = [
            'project', 'paths', 'dataset_generation', 
            'model', 'training', 'logging'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        if config.dataset_generation.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        splits = [
            config.dataset_generation.train_split,
            config.dataset_generation.val_split,
            config.dataset_generation.test_split
        ]
        
        if abs(sum(splits) - 1.0) > 0.001:
            raise ValueError(
                f"Train/val/test splits must sum to 1.0, got {sum(splits)}"
            )
        
        if config.model.architecture not in ['unet', 'swin_unet']:
            raise ValueError(
                f"Invalid architecture: {config.model.architecture}"
            )
        
        if config.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if config.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
    
    def _create_directories(self) -> None:
        """Create all necessary directories from configuration."""
        paths = self.config.paths
        
        directories = [
            paths.output_dir,
            paths.synthetic_data_dir,
            paths.images_dir,
            paths.masks_dir,
            paths.metadata_dir,
            paths.checkpoints_dir,
            paths.logs_dir,
            paths.predictions_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'training.batch_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates (supports dot notation in keys)
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(self.config), f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return OmegaConf.to_container(self.config, resolve=True)
    
    def print_config(self) -> None:
        """Print configuration in a readable format."""
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print(OmegaConf.to_yaml(self.config))
        print("=" * 80)


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)