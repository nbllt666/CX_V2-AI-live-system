"""
单元测试：测试所有模块的 cleanup 和 stop 方法
确保后台线程能正常停止且资源被释放
"""

import pytest
import sys
import os
import time
import threading
from unittest.mock import Mock, MagicMock, patch

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from content_moderation import ContentModeration, ImageModeration
from audio_recorder import AudioRecorder
from sense_voice import SenseVoiceRecognizer
from cosy_voice import VoiceOutputManager, CosyVoiceSynthesizer
from vdb_manager import VDBManager
from tool_manager import ToolManager, ToolExecutor


class TestContentModerationCleanup:
    """Test ContentModeration cleanup functionality."""
    
    def test_content_moderation_cleanup(self, sample_config):
        """Test that ContentModeration cleanup properly stops threads."""
        # Create moderator instance
        moderator = ContentModeration(sample_config)
        
        # Verify processing thread is running
        assert moderator.processing_thread.is_alive()
        
        # Call cleanup
        moderator.cleanup()
        
        # Wait a bit for thread to stop
        time.sleep(0.5)
        
        # Verify thread has stopped
        assert not moderator.processing_thread.is_alive()
    
    def test_content_moderation_cleanup_idempotent(self, sample_config):
        """Test that calling cleanup multiple times is safe."""
        moderator = ContentModeration(sample_config)
        
        # Call cleanup multiple times
        moderator.cleanup()
        moderator.cleanup()
        moderator.cleanup()
        
        # Should not raise any exceptions
        time.sleep(0.3)
        assert not moderator.processing_thread.is_alive()


class TestImageModerationCleanup:
    """Test ImageModeration cleanup functionality."""
    
    def test_image_moderation_cleanup(self, sample_config):
        """Test that ImageModeration delegates cleanup to ContentModeration."""
        image_mod = ImageModeration(sample_config)
        
        # Verify internal moderator has running thread
        assert image_mod.content_moderator.processing_thread.is_alive()
        
        # Call cleanup
        image_mod.cleanup()
        
        # Wait a bit
        time.sleep(0.5)
        
        # Verify thread stopped
        assert not image_mod.content_moderator.processing_thread.is_alive()


class TestAudioRecorderCleanup:
    """Test AudioRecorder cleanup functionality."""
    
    def test_audio_recorder_cleanup_stops_listening(self, sample_config):
        """Test that AudioRecorder cleanup stops listening."""
        def mock_handler(text):
            pass
        
        recorder = AudioRecorder(sample_config, mock_handler)
        
        # Start listening (creates a thread)
        recorder.start_listening()
        assert recorder.is_listening
        
        # Call cleanup
        recorder.cleanup()
        
        # Verify listening is stopped
        assert not recorder.is_listening
        time.sleep(0.3)
    
    def test_audio_recorder_cleanup_multiple_times(self, sample_config):
        """Test that calling cleanup multiple times is safe."""
        def mock_handler(text):
            pass
        
        recorder = AudioRecorder(sample_config, mock_handler)
        
        # Call cleanup multiple times
        recorder.cleanup()
        recorder.cleanup()
        recorder.cleanup()
        
        # Should not raise any exceptions
        time.sleep(0.3)


class TestSenseVoiceCleanup:
    """Test SenseVoiceRecognizer cleanup functionality."""
    
    def test_sense_voice_cleanup(self, sample_config):
        """Test that SenseVoiceRecognizer cleanup stops its thread."""
        recognizer = SenseVoiceRecognizer(sample_config)
        
        # Verify thread is running
        assert recognizer.processing_thread.is_alive()
        
        # Call cleanup
        recognizer.cleanup()
        
        # Wait for thread to stop
        time.sleep(0.5)
        
        # Verify thread has stopped
        assert not recognizer.processing_thread.is_alive()


class TestCosyVoiceCleanup:
    """Test CosyVoiceSynthesizer and VoiceOutputManager cleanup."""
    
    def test_cosy_voice_cleanup(self, sample_config):
        """Test that CosyVoiceSynthesizer cleanup stops its thread."""
        synthesizer = CosyVoiceSynthesizer(sample_config)
        
        # Verify thread is running
        assert synthesizer.processing_thread.is_alive()
        
        # Call cleanup
        synthesizer.cleanup()
        
        # Wait for thread to stop
        time.sleep(0.5)
        
        # Verify thread has stopped
        assert not synthesizer.processing_thread.is_alive()
    
    def test_voice_output_manager_cleanup(self, sample_config):
        """Test that VoiceOutputManager cleanup stops all threads."""
        manager = VoiceOutputManager(sample_config)
        
        # Verify threads are running
        assert manager.output_thread.is_alive()
        assert manager.synthesizer.processing_thread.is_alive()
        
        # Call cleanup
        manager.cleanup()
        
        # Wait for threads to stop
        time.sleep(0.5)
        
        # Verify threads have stopped
        assert not manager.output_thread.is_alive()
        assert not manager.synthesizer.processing_thread.is_alive()


class TestVDBManagerCleanup:
    """Test VDBManager cleanup and stop functionality."""
    
    @patch('vdb_manager.QdrantClient')
    def test_vdb_expiry_worker_cleanup(self, mock_qdrant, sample_config):
        """Test that VDBManager expiry worker can be stopped."""
        config = sample_config.copy()
        config['vdb_auto_run'] = True
        config['qdrant_host'] = 'localhost'
        config['qdrant_port'] = 6333
        
        vdb = VDBManager(config)
        
        # Verify expiry thread is running
        assert vdb.expiry_thread.is_alive()
        
        # Stop VDB management
        vdb.stop_vdb_management()
        
        # Wait for threads to stop
        time.sleep(1)
        
        # Verify threads stopped
        assert not vdb.expiry_thread.is_alive()
    
    @patch('vdb_manager.QdrantClient')
    def test_vdb_stop_management_idempotent(self, mock_qdrant, sample_config):
        """Test that stop_vdb_management can be called multiple times safely."""
        config = sample_config.copy()
        config['vdb_auto_run'] = True
        config['qdrant_host'] = 'localhost'
        config['qdrant_port'] = 6333
        
        vdb = VDBManager(config)
        
        # Call stop multiple times
        vdb.stop_vdb_management()
        vdb.stop_vdb_management()
        vdb.stop_vdb_management()
        
        # Should not raise any exceptions
        time.sleep(0.5)


class TestToolManagerCleanup:
    """Test ToolManager cleanup functionality."""
    
    def test_tool_manager_cleanup(self, sample_config):
        """Test that ToolManager cleanup stops its thread."""
        tool_mgr = ToolManager(sample_config)
        
        # Verify thread is running
        assert tool_mgr.processing_thread.is_alive()
        
        # Call cleanup
        tool_mgr.cleanup()
        
        # Wait for thread to stop
        time.sleep(0.5)
        
        # Verify thread has stopped
        assert not tool_mgr.processing_thread.is_alive()
    
    def test_tool_executor_cleanup(self, sample_config):
        """Test that ToolExecutor cleanup delegates to ToolManager."""
        executor = ToolExecutor(sample_config)
        
        # Verify tool_manager exists and thread is running
        assert executor.tool_manager.processing_thread.is_alive()
        
        # Call cleanup
        executor.cleanup()
        
        # Wait for thread to stop
        time.sleep(0.5)
        
        # Verify thread has stopped
        assert not executor.tool_manager.processing_thread.is_alive()
    
    def test_tool_executor_cleanup_multiple_times(self, sample_config):
        """Test that calling ToolExecutor cleanup multiple times is safe."""
        executor = ToolExecutor(sample_config)
        
        # Call cleanup multiple times
        executor.cleanup()
        executor.cleanup()
        executor.cleanup()
        
        # Should not raise any exceptions
        time.sleep(0.3)


class TestCleanupConcurrency:
    """Test cleanup behavior under concurrent access."""
    
    def test_concurrent_cleanup_calls(self, sample_config):
        """Test that multiple threads can safely call cleanup."""
        moderator = ContentModeration(sample_config)
        errors = []
        
        def cleanup_in_thread():
            try:
                moderator.cleanup()
            except Exception as e:
                errors.append(e)
        
        # Start multiple cleanup threads
        threads = [
            threading.Thread(target=cleanup_in_thread)
            for _ in range(5)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=2)
        
        # Should have no errors
        assert len(errors) == 0
        time.sleep(0.3)
        assert not moderator.processing_thread.is_alive()


class TestCleanupTimeout:
    """Test cleanup timeout behavior."""
    
    def test_cleanup_respects_timeout(self, sample_config):
        """Test that cleanup with timeout doesn't hang indefinitely."""
        moderator = ContentModeration(sample_config)
        
        # Record time
        start_time = time.time()
        
        # Call cleanup (should use timeout internally)
        moderator.cleanup()
        
        # Record elapsed time
        elapsed = time.time() - start_time
        
        # Should complete relatively quickly (within 1 second)
        # We're generous here because CI can be slow
        assert elapsed < 2.0
        
        time.sleep(0.3)


class TestDanmuReceiverStop:
    """Test DanmuReceiver stop functionality."""
    
    @patch('danmu_receiver.aiohttp.ClientSession')
    @patch('danmu_receiver.RSocketClient')
    def test_danmu_receiver_stop(self, mock_rsocket, mock_session, sample_config):
        """Test that DanmuReceiver can be stopped safely."""
        from danmu_receiver import DanmuReceiverFixed
        
        config = sample_config.copy()
        config['danmu_websocket_uri'] = 'ws://localhost:9898'
        config['danmu_task_ids'] = [12345]
        
        def mock_handler(msg):
            pass
        
        receiver = DanmuReceiverFixed(config, None, mock_handler)
        
        # Verify initial state
        assert not receiver.running
        
        # Call stop_receiving (should be idempotent)
        receiver.stop_receiving()
        receiver.stop_receiving()
        
        # Should not raise any exceptions
        assert not receiver.running


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration for tests."""
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
        "stop_trigger_key": "F12",
        "sense_voice_api_url": "http://127.0.0.1:8877/api/v1/asr",
        "cosyvoice_api_url": "http://127.0.0.1:50000/voice",
        "sound_effects_dir": "./sound_effects",
        "songs_dir": "./songs",
        "images_dir": "./images",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "vdb_auto_run": False,
        "vdb_check_interval": 60,
    }
