# 晨曦V2 AI直播系统 (CX-V2 AI Live System)

晨曦V2 AI直播系统是一个功能强大的AI直播助手，支持语音识别、语音合成、图像理解、弹幕交互等多种功能。

## 项目特色

1. **智能对话系统**：基于Ollama模型实现自然语言对话
2. **语音交互**：支持语音识别和语音合成
3. **图像理解**：支持视觉模型理解图像内容
4. **直播互动**：支持弹幕接收和实时互动
5. **WebUI控制**：提供Web界面进行系统控制
6. **三重上下文管理**：智能管理短期、中期、长期上下文
7. **工具调用**：支持多种工具调用（音效、点歌、AI绘画、屏幕点击等）
8. **流式TTS**：支持按标点符号分割的流式语音合成

## 功能特性

### 1. 三重上下文文件管理
- **短期上下文** (`context_short_term.json`) - 摘要后清空
- **中期上下文** (`context_medium_term.json`) - 保留设定数量条目（默认50条），主模型使用
- **长期上下文** (`context_long_term.json`) - 永不清空，作为备份
- 系统提示词始终保留在上下文中

### 2. 上下文摘要功能
- 每24小时或启动时自动调用副模型生成摘要
- 按重要性分级（高、中、低）保存记忆
- 通过Ollama chat API实现
- 工具调用的记忆作为永久记忆不删除

### 3. WebUI控制面板
- 配置管理（保存/加载）
- 系统控制（启动/停止/重启）
- 直接消息发送功能（绕过ASR）
- 实时日志显示

### 4. 智能语音合成
- 按标点符号分割文本（每3句一段）
- 支持流式播放
- 语音输出更自然
- 支持多种TTS后端

### 5. 向量数据库功能
- 使用Qdrant向量数据库实现真正的语义搜索
- 支持Ollama嵌入API、聊天API变通方法和多层回退机制
- 语义相似度搜索
- 过期记忆自动清理

### 6. 长期记忆智能管理
- **循环式管理**：持续优化长期记忆，直到LLM发出停止指令
- **智能操作**：支持删除、合并、修改和重新分类记忆
- **专用工具集**：提供delete_memory、merge_memories、modify_memory、update_metadata等工具
- **持久性保护**：重要的持久性记忆受到保护，防止误删
- **自动化管理**：副模型自动分析并管理记忆库

## 系统架构

```
CX-V2/
├── main.py              # 主程序入口
├── main_model.py        # 主AI模型（对话逻辑）
├── vdb_manager.py       # 向量数据库管理器（含LLM智能评估和长期记忆管理）
├── webui.py             # WebUI控制器
├── cosy_voice.py        # 语音输出管理器
├── sense_voice.py       # 语音识别器
├── content_moderation.py # 内容审核模块
├── danmu_receiver.py    # 弹幕接收器
├── screen_manager.py    # 屏幕管理器
├── tool_manager.py      # 工具管理器
├── long_term_memory.py  # 长期记忆系统
├── audio_recorder.py    # 音频录制器
├── utils.py             # 工具函数
├── config/              # 配置文件
├── templates/           # WebUI模板
├── docs/                # 文档目录
├── run_tests.py         # 运行测试入口
├── show_docs.py         # 项目文档查看器
├── demo_memory_management.py # 长期记忆管理演示
├── requirements.txt     # 依赖库
└── README.md            # 此文档
```

## 安装说明

1. 安装Python 3.8+
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```

3. 确保以下服务运行：
   - Ollama服务 (默认端口11434)
   - Qdrant向量数据库 (默认端口6333，可选)
   - 各种API服务（根据配置）

## 配置文件

主要配置文件：`config/config.json`

## 使用方法

### 启动系统
```bash
python main.py
```

### WebUI访问
系统启动后，WebUI将在 `http://localhost:5000` 上运行

### 主要功能
- **发送直接消息**：通过WebUI直接发送消息，绕过语音识别
- **重启系统**：通过WebUI一键重启
- **配置管理**：通过WebUI实时修改配置
- **查看日志**：实时查看系统日志

## 运行测试

要运行所有测试：
```bash
python run_tests.py
```

或单独运行测试：
```bash
python docs/test_features.py
python docs/test_context_management.py
```

## 项目文档

项目文档位于 `docs/` 目录中：
- `CHANGES_AND_FEATURES.md` - 功能变更说明
- `FULL_PROJECT_REVIEW.md` - 项目全面检查总结
- `README.md` - 项目文档说明

## 依赖库

详见 `requirements.txt`

## 项目状态

- ✅ 核心功能完整
- ✅ WebUI控制面板
- ✅ 三重上下文管理
- ✅ 语音交互功能
- ✅ 向量数据库集成
- ✅ 流式TTS功能
- ✅ 工具调用支持

## 许可证

本项目为AI直播助手系统，提供完整的直播互动解决方案。

## 贡献

欢迎提交Issue和Pull Request。

## 版本

晨曦V2 AI直播系统 - CX-V2