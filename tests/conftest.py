import pytest
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration for tests."""
    return {
        "ollama_api_url": "http://localhost:11434/api/chat",
        "ollama_model": "qwen2.5:0.5b",
        "ollama_vision_model": "qwen2.5vl",
        "ollama_temperature": 0.7,
        "enable_sound_effects": True,
        "enable_music": True,
        "enable_ai_drawing": False,
        "enable_continuous_mode": True,
        "trigger_key": "F11",
        "stop_trigger_key": "F12"
    }