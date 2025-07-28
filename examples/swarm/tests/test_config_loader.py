"""
Tests for Configuration Loader
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from config_loader import ConfigLoader

def test_default_config():
    """Test default configuration generation"""
    config = ConfigLoader._get_default_config()
    
    assert 'system' in config
    assert 'llm_config' in config
    assert 'agents' in config
    assert 'monitoring' in config
    
    assert config['system']['name'] == 'Autonomous Agent System'
    assert config['llm_config']['model'] == 'gpt-4o-mini'

def test_load_nonexistent_config():
    """Test loading non-existent config file"""
    config = ConfigLoader.load_config(Path("nonexistent.yaml"))
    
    # Should return default config
    default_config = ConfigLoader._get_default_config()
    assert config == default_config

def test_load_valid_config():
    """Test loading valid config file"""
    test_config = {
        'system': {
            'name': 'Test System',
            'max_concurrent_tasks': 10
        },
        'llm_config': {
            'model': 'gpt-4',
            'temperature': 0.5
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = Path(f.name)
    
    try:
        loaded_config = ConfigLoader.load_config(config_path)
        
        # Should merge with defaults
        assert loaded_config['system']['name'] == 'Test System'
        assert loaded_config['system']['max_concurrent_tasks'] == 10
        assert loaded_config['llm_config']['model'] == 'gpt-4'
        assert loaded_config['llm_config']['temperature'] == 0.5
        
        # Should keep defaults for unspecified values
        assert 'agents' in loaded_config
        assert 'monitoring' in loaded_config
        
    finally:
        config_path.unlink()

def test_config_merging():
    """Test configuration merging"""
    default = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3
        },
        'e': 4
    }
    
    user = {
        'b': {
            'c': 5
        },
        'f': 6
    }
    
    merged = ConfigLoader._merge_configs(default, user)
    
    assert merged['a'] == 1  # from default
    assert merged['b']['c'] == 5  # overridden by user
    assert merged['b']['d'] == 3  # from default
    assert merged['e'] == 4  # from default
    assert merged['f'] == 6  # from user

if __name__ == "__main__":
    pytest.main([__file__])