import pytest
from unittest.mock import Mock, patch
from main_model import AILiveModel

class TestAILiveModel:
    """Test cases for the AILiveModel class."""
    
    @patch('main_model.OllamaClient')
    def test_model_initialization(self, mock_ollama_client):
        """Test AILiveModel initialization."""
        # Mock the configuration
        config = {
            "ollama_api_url": "http://localhost:11434/api/chat",
            "ollama_model": "qwen2.5:0.5b",
            "ollama_temperature": 0.7,
            "max_conversation_history": 50
        }
        
        # Create an instance of AILiveModel
        model = AILiveModel(config)
        
        # Assert that the model was created successfully
        assert model is not None
        assert model.config == config
        
    def test_update_config(self):
        """Test updating configuration."""
        config = {
            "ollama_model": "qwen2.5:0.5b",
            "ollama_temperature": 0.7
        }
        
        model = AILiveModel(config)
        model.update_config("ollama_model", "new_model")
        
        assert model.config["ollama_model"] == "new_model"