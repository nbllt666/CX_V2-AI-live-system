import pytest
import json
import os
from unittest.mock import mock_open, patch
from config_manager import ConfigManager

class TestConfigManager:
    """Test cases for the ConfigManager class."""
    
    def test_config_manager_initialization(self, sample_config):
        """Test ConfigManager initialization with a sample config."""
        with patch("config_manager.os.path.exists", return_value=False):
            with patch("builtins.open", mock_open(read_data=json.dumps(sample_config))):
                config_manager = ConfigManager()
                assert config_manager is not None
    
    def test_get_config_with_validation_valid_config(self, sample_config):
        """Test get_config_with_validation with a valid config."""
        with patch("config_manager.os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(sample_config))):
                config_manager = ConfigManager()
                config = config_manager.get_config_with_validation()
                assert config is not None
                assert "ollama_api_url" in config
                assert "ollama_model" in config
    
    def test_save_config(self, sample_config):
        """Test saving configuration."""
        with patch("config_manager.os.path.exists", return_value=True):
            with patch("builtins.open", mock_open()) as mock_file:
                config_manager = ConfigManager()
                config_manager.save_config(sample_config)
                mock_file.assert_called_once()