"""
YAML Configuration Loader
Allows overriding any configuration parameter via YAML files
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class YAMLConfigError(Exception):
    """Exception raised for YAML configuration errors"""
    pass

class YAMLConfigLoader:
    """
    Loads and manages YAML configuration files
    
    Supports:
    - Multiple config files with priority order
    - Environment-specific overrides
    - Validation of config values
    - Nested configuration structures
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.config_data = {}
        self._loaded_files = []
    
    def load_config_files(self, filenames: list = None) -> Dict[str, Any]:
        """
        Load YAML configuration files in order of priority
        
        Args:
            filenames: List of YAML files to load. If None, uses default order.
        
        Returns:
            Merged configuration dictionary
        """
        if filenames is None:
            # Default loading order (lower priority to higher)
            filenames = [
                'config.yaml',           # Base configuration
                f'config.{os.getenv("ENVIRONMENT", "development")}.yaml',  # Environment-specific
                'config.local.yaml'      # Local overrides (highest priority)
            ]
        
        self.config_data = {}
        
        for filename in filenames:
            file_path = os.path.join(self.config_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            self._merge_configs(self.config_data, file_config)
                            self._loaded_files.append(filename)
                            logger.info(f"Loaded YAML config: {filename}")
                except yaml.YAMLError as e:
                    raise YAMLConfigError(f"Error parsing YAML file {filename}: {e}")
                except Exception as e:
                    raise YAMLConfigError(f"Error loading YAML file {filename}: {e}")
            else:
                logger.debug(f"YAML config file not found: {filename}")
        
        logger.info(f"Loaded {len(self._loaded_files)} YAML config files: {self._loaded_files}")
        return self.config_data
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge configuration dictionaries
        Override values take precedence over base values
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._merge_configs(base[key], value)
            else:
                # Override the value
                base[key] = value
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the config value (e.g., 'trading.position_size')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def update_module_config(self, module_globals: Dict[str, Any], section: str = None) -> int:
        """
        Update a module's global variables from YAML config
        
        Args:
            module_globals: Module's globals() dictionary
            section: Optional YAML section to use (if None, uses root level)
        
        Returns:
            Number of variables updated
        """
        config_section = self.config_data
        if section:
            config_section = self.get_config_value(section, {})
        
        updated_count = 0
        
        for key, value in config_section.items():
            # Convert to uppercase for constants
            const_name = key.upper()
            
            if const_name in module_globals:
                original_value = module_globals[const_name]
                original_type = type(original_value)
                
                # Type conversion and validation
                try:
                    if original_type == bool:
                        # Handle boolean conversion carefully
                        if isinstance(value, str):
                            converted_value = value.lower() in ('true', 'yes', '1', 'on')
                        else:
                            converted_value = bool(value)
                    elif original_type in (int, float, str):
                        converted_value = original_type(value)
                    else:
                        # Keep original type for complex types
                        converted_value = value
                    
                    module_globals[const_name] = converted_value
                    updated_count += 1
                    
                    logger.info(f"Updated {const_name}: {original_value} -> {converted_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {key}={value} to {original_type}: {e}")
        
        return updated_count
    
    def validate_config(self, validation_rules: Dict[str, Dict[str, Any]]) -> bool:
        """
        Validate configuration against rules
        
        Args:
            validation_rules: Dictionary of validation rules
                Format: {
                    'config.key': {
                        'type': int,
                        'min': 0,
                        'max': 100,
                        'required': True
                    }
                }
        
        Returns:
            True if valid, raises YAMLConfigError if invalid
        """
        errors = []
        
        for key_path, rules in validation_rules.items():
            value = self.get_config_value(key_path)
            
            # Required check
            if rules.get('required', False) and value is None:
                errors.append(f"Required config key '{key_path}' is missing")
                continue
            
            if value is not None:
                # Type check
                expected_type = rules.get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Config key '{key_path}' should be {expected_type.__name__}, got {type(value).__name__}")
                
                # Range checks for numbers
                if isinstance(value, (int, float)):
                    min_val = rules.get('min')
                    max_val = rules.get('max')
                    if min_val is not None and value < min_val:
                        errors.append(f"Config key '{key_path}' value {value} is below minimum {min_val}")
                    if max_val is not None and value > max_val:
                        errors.append(f"Config key '{key_path}' value {value} is above maximum {max_val}")
                
                # Choices check
                choices = rules.get('choices')
                if choices and value not in choices:
                    errors.append(f"Config key '{key_path}' value '{value}' not in allowed choices: {choices}")
        
        if errors:
            raise YAMLConfigError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        return True
    
    def save_current_config(self, filename: str) -> None:
        """
        Save current configuration to a YAML file
        
        Args:
            filename: Output filename
        """
        output_path = os.path.join(self.config_dir, filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config_data, f, default_flow_style=False, sort_keys=True)
            logger.info(f"Saved current config to {filename}")
        except Exception as e:
            raise YAMLConfigError(f"Failed to save config to {filename}: {e}")

# Global instance for easy access
_global_loader = None

def get_yaml_loader() -> YAMLConfigLoader:
    """Get the global YAML config loader instance"""
    global _global_loader
    if _global_loader is None:
        _global_loader = YAMLConfigLoader()
        _global_loader.load_config_files()
    return _global_loader

def load_yaml_config(config_dir: str = "config") -> YAMLConfigLoader:
    """
    Convenience function to create and load YAML configuration
    
    Args:
        config_dir: Directory containing YAML config files
    
    Returns:
        Configured YAMLConfigLoader instance
    """
    loader = YAMLConfigLoader(config_dir)
    loader.load_config_files()
    return loader