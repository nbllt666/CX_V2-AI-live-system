#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify multiple context file management functionality
"""
import sys
import os
import time
import json
# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_multiple_context_files():
    """Test the multiple context file functionality"""
    print("Testing multiple context file management...")
    
    # Create a mock config
    config = {
        'output_audio_dir': './output_audio',
        'use_molotts': False,
        'ollama_api_url': 'http://localhost:11434',
        'summarizer_model': 'gemma2:2b',
        'qdrant_host': 'localhost',
        'qdrant_port': 6333,
        'max_context_items': 10
    }
    
    try:
        from vdb_manager import VDBManager
        
        # Create VDBManager instance
        vdb_manager = VDBManager(config)
        print("VDBManager instantiated successfully with multiple context files")
        
        # Create sample context messages
        sample_context = [
            {'role': 'system', 'content': 'You are a helpful AI assistant.', 'timestamp': '2024-01-01T00:00:00'},
            {'role': 'user', 'content': 'Hello, how are you?', 'timestamp': '2024-01-01T00:01:00'},
            {'role': 'assistant', 'content': 'I am fine, thank you for asking.', 'timestamp': '2024-01-01T00:01:05'},
            {'role': 'user', 'content': 'What can you help me with?', 'timestamp': '2024-01-01T00:02:00'},
            {'role': 'assistant', 'content': 'I can answer questions and have conversations.', 'timestamp': '2024-01-01T00:02:05'}
        ]
        
        # Test saving multiple context files
        result = vdb_manager.save_multiple_context_files(sample_context, "You are a helpful AI assistant.")
        print(f"Multiple context save result: {result}")
        
        # Test loading multiple context files
        loaded_contexts = vdb_manager.load_multiple_context_files()
        print(f"Loaded context types: {list(loaded_contexts.keys())}")
        for ctx_type, ctx_data in loaded_contexts.items():
            print(f"  {ctx_type}: {len(ctx_data)} items")
        
        # Test getting effective context with system prompt
        effective_context = vdb_manager.get_effective_context("You are a helpful AI assistant.", "medium")
        print(f"Effective context length: {len(effective_context)}")
        
        # Verify system prompt is in the context
        has_system_prompt = any(msg.get('role') == 'system' for msg in effective_context)
        print(f"System prompt in effective context: {has_system_prompt}")
        
        print("Multiple context file management test completed successfully!")
        vdb_manager.expiry_thread.join(timeout=1)  # 停止线程
        vdb_manager.summary_thread.join(timeout=1)  # 停止线程
        
        # Clean up test files
        for filename in ['context_short_term.json', 'context_medium_term.json', 'context_long_term.json']:
            if os.path.exists(filename):
                os.remove(filename)
        
        return True
        
    except Exception as e:
        print(f"Error testing multiple context files: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run multiple context management test"""
    print("Starting multiple context management test...\n")
    
    result = test_multiple_context_files()
    
    print(f"\nMultiple Context Management Test: {'PASSED' if result else 'FAILED'}")
    
    return result

if __name__ == "__main__":
    main()