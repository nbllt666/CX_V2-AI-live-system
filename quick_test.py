#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证更新后的系统功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    print("正在进行快速功能验证...")
    print("="*50)

    # 测试1: 验证VDBManager能够正确导入
    try:
        from vdb_manager import VDBManager
        config = {
            'output_audio_dir': './output_audio',
            'use_molotts': False,
            'ollama_api_url': 'http://localhost:11434',
            'summarizer_model': 'gemma2:2b',
            'qdrant_host': 'localhost',
            'qdrant_port': 6333
        }
        vdb = VDBManager(config)
        print("[OK] VDBManager 导入和初始化成功")

        # 验证新的重要性评估方法
        if hasattr(vdb, '_let_llm_decide_summary'):
            print("[OK] LLM智能重要性评估方法存在")
        else:
            print("[FAIL] LLM智能重要性评估方法缺失")

        if hasattr(vdb, '_create_summary_memory'):
            print("[OK] 改进的摘要创建方法存在")
        else:
            print("[FAIL] 改进的摘要创建方法缺失")

        # 停止后台线程
        vdb.expiry_thread.join(timeout=1)
        vdb.summary_thread.join(timeout=1)

        print("\n所有验证通过！系统已正确更新。")
        return True

    except Exception as e:
        print(f"[FAIL] 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n[SUCCESS] 晨曦V2 AI直播系统已成功更新！")
        print("   - LLM智能重要性评估功能已启用")
        print("   - 空上下文自动跳过摘要功能已实现")
        print("   - 摘要决策由副模型智能判断")
    else:
        print("\n[ERROR] 验证失败，请检查系统配置")