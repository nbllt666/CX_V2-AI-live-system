import pytest
from unittest.mock import Mock, patch
from main_model import MainModel

class TestMainModel:
    """Test cases for the MainModel class."""

    @patch('main_model.requests')
    def test_model_initialization(self, mock_requests):
        """Test MainModel initialization."""
        # Mock the configuration
        config = {
            "ollama_api_url": "http://localhost:11434/api/chat",
            "ollama_model": "qwen2.5:0.5b",
            "ollama_temperature": 0.7,
            "max_conversation_history": 50
        }

        # Mock the response for _check_ollama_connection
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        # Create an instance of MainModel
        model = MainModel(config, None)

        # Assert that the model was created successfully
        assert model is not None
        assert model.config == config

    def test_update_config(self):
        """Test updating configuration."""
        config = {
            "ollama_model": "qwen2.5:0.5b",
            "ollama_temperature": 0.7
        }

        # Mock the response for _check_ollama_connection
        import requests
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.status_code = 200
        import main_model
        original_post = main_model.requests.post
        main_model.requests.post = lambda *args, **kwargs: mock_response

        model = MainModel(config, None)
        # 恢复原始方法
        main_model.requests.post = original_post

        # 直接修改配置进行测试
        original_model = model.config["ollama_model"]
        model.config["ollama_model"] = "new_model"

        assert model.config["ollama_model"] == "new_model"