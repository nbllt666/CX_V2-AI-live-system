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

### 3. 工具调用功能
- **播放音效**：支持播放指定的音效文件
- **播放音乐**：支持播放指定的音乐文件
- **AI绘画**：支持文本生成图像功能，生成的图像会经过内容审核
- **屏幕点击**：支持指定坐标点击屏幕，实现人机交互
- **连续操作模式**：支持自动化的连续操作流程，可实现复杂的自动化任务

### 4. WebUI控制面板
- 配置管理（保存/加载）
- 系统控制（启动/停止/重启）
- 直接消息发送功能（绕过ASR）
- 实时日志显示
- 数据库管理设置控制（自动运行开关、启动时运行开关、检查间隔调整）
- 手动触发数据库管理操作功能

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
- **数据库管理设置控制**：支持通过API或WebUI控制自动清理、摘要生成和定期维护任务的启用与间隔设置

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
   - 让弹幕飞

## 配置文件

主要配置文件：`config/config.json`，包含以下关键配置项：

### 模型配置
- `ollama_api_url`: Ollama API地址
- `ollama_model`: 文本模型名称（默认qwen2.5:0.5b）
- `ollama_vision_model`: 视觉模型名称（默认qwen2.5vl）
- `ollama_temperature`: 生成温度参数
- `ollama_max_tokens`: 最大生成令牌数

### 功能开关
- `enable_sound_effects`: 启用音效播放
- `enable_music`: 启用音乐播放
- `enable_ai_drawing`: 启用AI绘画
- `enable_continuous_mode`: 启用连续操作模式
- `enable_danmu`: 启用弹幕功能
- `enable_voice_recognition`: 启用语音识别
- `enable_wake_sleep`: 启用唤醒/睡眠功能

### 语音交互配置
- `voice_trigger_enabled`: 启用语音触发
- `key_trigger_enabled`: 启用按键触发
- `trigger_key`: 触发按键（默认F11）
- `stop_trigger_key`: 停止触发按键（默认F12）
- `wake_word`: 唤醒词
- `sleep_word`: 睡眠词

### 向量数据库配置
- `vdb_auto_run`: 启用向量数据库自动运行
- `vdb_startup_run`: 启动时运行向量数据库管理
- `vdb_check_interval`: 检查间隔（秒）
- `enable_llm_vdb_control`: 启用LLM向量数据库控制

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
- **数据库管理**：控制向量数据库的自动运行和检查间隔

### 交互方式
- **弹幕交互**：需启用弹幕功能，配置WebSocket地址和任务ID
- **语音交互**：
  - 按键模式：按F11开始录音，F12停止
  - 语音触发模式：说唤醒词激活系统
- **WebUI交互**：通过Web界面手动输入消息

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
- ✅ 屏幕点击功能
- ✅ 连续操作模式

## 许可证

 MIT License

本项目为AI直播助手系统，提供完整的直播互动解决方案。

## 贡献

欢迎提交Issue和Pull Request。

## 版本

晨曦V2 AI直播系统 - CX-V2