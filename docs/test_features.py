#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify implemented features
"""
import sys
import os
import json
import time
import threading
# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from cosy_voice import VoiceOutputManager

def test_text_splitting():
    """Test the text splitting functionality"""
    print("Testing text splitting functionality...")
    
    # Create a mock config
    config = {
        'output_audio_dir': './output_audio',
        'use_molotts': False,
        'cosy_voice_api_url': 'http://127.0.0.1:8888/api/tts'
    }
    
    # Create VoiceOutputManager instance
    voice_manager = VoiceOutputManager(config)
    
    # Test text with various punctuation
    test_text = "你好。这是一个测试。我们将检查文本分割功能！这段话包含多个句子？还有一个句子。"
    
    print(f"Original text: {test_text}")
    
    # Test the splitting function
    chunks = voice_manager._split_text_by_punctuation(test_text, max_sentences=3)
    
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk}")
    
    # Test with longer text
    long_text = "这是第一句。这是第二句。这是第三句。这是第四句。这是第五句。这是第六句。这是第七句。"
    print(f"\nTesting with longer text: {long_text}")
    
    chunks = voice_manager._split_text_by_punctuation(long_text, max_sentences=3)
    
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk}")
    
    # Cleanup
    voice_manager.cleanup()
    
    print("\nText splitting test completed successfully!")
    return True

def test_webui_endpoints():
    """Test WebUI endpoints by importing and checking"""
    print("\nTesting WebUI endpoints...")

    try:
        import webui
        print("WebUI module imported successfully")

        # Check that the controller has the required methods
        controller = webui.WebUIController()
        print("WebUI Controller instantiated successfully")

        # Check required API methods exist
        methods_to_check = ['load_config', 'save_config', 'update_status', 'add_log_message']
        for method in methods_to_check:
            assert hasattr(controller, method), f"Missing method: {method}"
            print(f"Method {method} exists")

        print("WebUI endpoint test completed successfully!")
        return True

    except ImportError as e:
        print(f"Failed to import WebUI: {e}")
        return False
    except Exception as e:
        print(f"Error testing WebUI: {e}")
        return False

def test_direct_message_functionality():
    """Test direct message functionality by checking main.py integration"""
    print("\nTesting direct message functionality...")

    # Check that main.py has the add_manual_message method
    try:
        from main import AILiveSystem

        # Create a temporary config file
        temp_config = {
            'enable_danmu': False,
            'auto_restart': False,
            'log_dir': './logs',
            'output_audio_dir': './output_audio',
            'use_molotts': False,
            'ollama_api_url': 'http://localhost:11434/api/chat',
            'ollama_model': 'qwen2.5:0.5b',
            'ollama_vision_model': 'qwen2.5vl'
        }

        # Create a minimal instance
        system = AILiveSystem(temp_config)
        print("AILiveSystem instantiated successfully")

        # Check that the method exists
        assert hasattr(system, 'add_manual_message'), "add_manual_message method missing"
        print("add_manual_message method exists")

        # Clean up
        system.stop()

        print("Direct message functionality test completed successfully!")
        return True

    except ImportError as e:
        print(f"Failed to import AILiveSystem: {e}")
        return False
    except Exception as e:
        print(f"Error testing direct message functionality: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting feature tests...\n")
    
    tests = [
        ("Text Splitting", test_text_splitting),
        ("WebUI Endpoints", test_webui_endpoints),
        ("Direct Message Functionality", test_direct_message_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{test_name} test: {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            print(f"{test_name} test: FAILED with error: {e}\n")
            results.append((test_name, False))
    
    print("Test Results:")
    for test_name, result in results:
        print(f"  {test_name}: {'PASSED' if result else 'FAILED'}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()