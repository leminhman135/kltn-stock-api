"""
Config Loader - Load configuration from YAML file
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration loader with environment variable support"""
    
    def __init__(self, config_file: str = None):
        if config_file is None:
            config_file = os.path.join(
                Path(__file__).parent.parent, 
                'config', 
                'etl_config.yaml'
            )
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Substitute environment variables
        self._substitute_env_vars(self.config)
    
    def _substitute_env_vars(self, config: Dict) -> None:
        """Recursively substitute ${VAR:default} with environment variables"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str) and '${' in value:
                # Extract variable name and default
                start = value.index('${') + 2
                end = value.index('}')
                var_spec = value[start:end]
                
                if ':' in var_spec:
                    var_name, default = var_spec.split(':', 1)
                else:
                    var_name = var_spec
                    default = ''
                
                # Get from environment or use default
                env_value = os.getenv(var_name, default)
                config[key] = value.replace('${' + var_spec + '}', env_value)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path
        
        Example:
            config.get('database.host')
            config.get('etl.batch_size', 1000)
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.config


# Global config instance
_config = None


def get_config(config_file: str = None) -> Config:
    """Get singleton config instance"""
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


if __name__ == "__main__":
    # Test config loader
    config = get_config()
    
    print("Database config:")
    print(f"  Host: {config.get('database.host')}")
    print(f"  Database: {config.get('database.database')}")
    
    print("\nTechnical Indicators:")
    print(f"  MACD: {config.get('technical_indicators.macd')}")
    print(f"  RSI Period: {config.get('technical_indicators.rsi.period')}")
    
    print("\nModel config:")
    print(f"  LSTM units: {config.get('model.lstm.units')}")
