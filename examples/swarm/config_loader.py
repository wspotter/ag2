"""
Configuration Loader - Handles system configuration
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not config_path.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return ConfigLoader._get_default_config()
            
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Merge with defaults
            default_config = ConfigLoader._get_default_config()
            merged_config = ConfigLoader._merge_configs(default_config, config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Using default configuration")
            return ConfigLoader._get_default_config()
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'system': {
                'name': 'Autonomous Agent System',
                'version': '1.0.0',
                'log_level': 'INFO',
                'max_concurrent_tasks': 5,
                'task_timeout': 3600,  # 1 hour
                'health_check_interval': 30
            },
            'llm_config': {
                'model': 'gpt-4o-mini',
                'temperature': 0.7,
                'max_tokens': 2000,
                'timeout': 120
            },
            'agents': {
                'coordinator': {
                    'enabled': True,
                    'max_rounds': 20
                },
                'researcher': {
                    'enabled': True,
                    'max_rounds': 15
                },
                'analyst': {
                    'enabled': True,
                    'max_rounds': 15
                },
                'executor': {
                    'enabled': True,
                    'max_rounds': 10
                }
            },
            'monitoring': {
                'metrics_interval': 300,  # 5 minutes
                'performance_logging': True,
                'health_checks': True
            }
        }
    
    @staticmethod
    def _merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result