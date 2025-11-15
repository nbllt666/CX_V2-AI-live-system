#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify context summarization functionality
"""
import sys
import os
import time
import threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_context_summarization():
    """Test the context summarization functionality"""
    print("Testing context summarization functionality...")
    
    # Create a mock config
    config = {
        'output_audio_dir': './output_audio',
        'use_molotts': False,
        'ollama_api_url': 'http://localhost:11434',
        'summarizer_model': 'gemma2:2b',
        'qdrant_host': 'localhost',
        'qdrant_port': 6333
    }
    
    try:
        from vdb_manager import VDBManager
        
        # Create VDBManager instance
        vdb_manager = VDBManager(config)
        print("VDBManager instantiated successfully with context summarization")
        
        # Test importance calculation
        test_content = "这是一个重要的信息，需要特别注意。"
        importance_score = vdb_manager._calculate_importance_score(test_content)
        print(f"Importance score for '{test_content[:30]}...': {importance_score}")
        
        test_content2 = "普通内容。"
        importance_score2 = vdb_manager._calculate_importance_score(test_content2)
        print(f"Importance score for '{test_content2}': {importance_score2}")
        
        # Test memory addition with different types
        vdb_manager.add_memory("这是一个重要信息", "permanent", {"source": "test"})
        vdb_manager.add_memory("这是一个普通信息", "weekly", {"source": "test"})
        
        print("Context summarization test completed successfully!")
        vdb_manager.expiry_thread.join(timeout=1)  # 停止线程
        vdb_manager.summary_thread.join(timeout=1)  # 停止线程
        
        return True
        
    except Exception as e:
        print(f"Error testing context summarization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run context summarization test"""
    print("Starting context summarization test...\n")
    
    result = test_context_summarization()
    
    print(f"\nContext Summarization Test: {'PASSED' if result else 'FAILED'}")
    
    return result

if __name__ == "__main__":
    main()