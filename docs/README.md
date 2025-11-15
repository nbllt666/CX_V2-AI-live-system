# CX-V2 项目文档

## 项目结构

```
CX-V2/
├── docs/                    # 文档目录
│   ├── CHANGES_AND_FEATURES.md    # 功能变更说明
│   ├── FULL_PROJECT_REVIEW.md     # 项目全面检查总结
│   ├── test_features.py           # 主要功能测试脚本
│   ├── test_context_management.py # 上下文管理测试脚本
│   └── README.md                  # 本文档
├── config/                  # 配置文件
├── templates/               # WebUI模板
├── main.py                  # 主程序入口
├── main_model.py            # 主模型（AI对话逻辑）
├── vdb_manager.py           # 向量数据库管理器
├── webui.py                 # WebUI控制器
├── run_tests.py             # 运行所有测试的入口脚本
├── show_docs.py             # 项目文档查看器
├── ...                      # 其他模块文件
└── requirements.txt         # 依赖库
```

## 主要功能

1. **AI直播助手系统**：完整的AI直播助手，支持语音识别、语音合成、图像理解等功能

2. **WebUI控制面板**：
   - 配置管理
   - 系统控制（启动/停止/重启）
   - 直接消息发送功能
   - 实时日志显示

3. **三重上下文文件管理**：
   - 短期上下文（`context_short_term.json`）- 摘要后清空
   - 中期上下文（`context_medium_term.json`）- 保留50个条目，主模型使用
   - 长期上下文（`context_long_term.json`）- 永不清空，作为备份
   - 系统提示词始终保留在上下文中

4. **上下文摘要功能**：
   - 每24小时或启动时自动调用副模型生成摘要
   - 按重要性分级（高、中、低）保存记忆
   - 通过Ollama chat API实现
   - 工具调用的记忆作为永久记忆不删除

5. **TTS流式处理**：
   - 按标点符号分割文本（每3句一段）
   - 支持流式播放
   - 语音输出更自然

6. **音频播放功能**：
   - 支持音效和歌曲播放
   - 使用pygame作为主要播放库，支持多种后端

7. **向量数据库功能**：
   - 使用Qdrant向量数据库实现真正的语义搜索
   - 支持Ollama嵌入API、聊天API变通方法和多层回退机制
   - 语义相似度搜索
   - 过期记忆自动清理

## 运行测试

要运行所有测试，请执行：

```bash
python run_tests.py
```

或直接运行特定测试：

```bash
python docs/test_features.py
python docs/test_context_management.py
```

## 配置说明

- 主要配置文件：`config/config.json`
- WebUI会自动使用中期上下文文件（`context_medium_term.json`）
- 系统提示词始终保留在上下文中
- 所有重要记忆作为永久记忆保存