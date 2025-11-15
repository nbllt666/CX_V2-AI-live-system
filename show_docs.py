#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目文档查看器
"""
import os

def show_project_overview():
    """显示项目概览"""
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    
    readme_path = os.path.join(docs_dir, 'README.md')
    
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("CX-V2 项目概览")
        print("="*50)
        print(content)
    else:
        print("项目文档未找到")

def show_docs_list():
    """显示文档列表"""
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    
    if os.path.exists(docs_dir):
        files = os.listdir(docs_dir)
        print("项目文档列表:")
        print("-" * 30)
        for file in files:
            print(f"  - {file}")
    else:
        print("文档目录不存在")

if __name__ == "__main__":
    print("CX-V2 项目文档管理")
    print("="*50)
    show_docs_list()
    print()
    show_project_overview()