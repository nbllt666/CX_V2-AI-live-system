#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本：运行所有测试
"""
import os
import sys
import subprocess

def run_all_tests():
    """运行所有测试"""
    print("运行所有测试...")
    
    # 获取docs目录中的测试文件
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    
    test_files = [
        'test_features.py',
        'test_context_management.py'
    ]
    
    all_passed = True
    
    for test_file in test_files:
        test_path = os.path.join(docs_dir, test_file)
        if os.path.exists(test_path):
            print(f"\n运行测试: {test_file}")
            print("-" * 50)
            
            try:
                result = subprocess.run([sys.executable, test_path], 
                                      cwd=os.path.dirname(__file__),
                                      capture_output=True, text=True)
                
                print(result.stdout)
                if result.stderr:
                    print("错误输出:")
                    print(result.stderr)
                
                if result.returncode != 0:
                    all_passed = False
                    print(f"测试 {test_file} 失败!")
                else:
                    print(f"测试 {test_file} 通过!")
                    
            except Exception as e:
                print(f"运行测试 {test_file} 时出错: {e}")
                all_passed = False
        else:
            print(f"测试文件不存在: {test_file}")
            all_passed = False
    
    print("\n" + "="*50)
    print(f"总体结果: {'所有测试通过' if all_passed else '部分测试失败'}")
    return all_passed

if __name__ == "__main__":
    run_all_tests()