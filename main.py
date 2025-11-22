import asyncio
import json
import logging
import sys
import threading
import time
from typing import Dict, Any, Optional
import queue

# 导入项目模块
from vdb_manager import VDBManager
from main_model import MainModel
from danmu_receiver import DanmuReceiverFixed as DanmuReceiver
from sense_voice import SenseVoiceRecognizer, VoiceTrigger
from cosy_voice import VoiceOutputManager
from content_moderation import ContentModeration, ImageModeration
from long_term_memory import LongTermMemory, ContextSummarizer
from screen_manager import ScreenManager, ScreenCapture
from tool_manager import ToolExecutor
from audio_recorder import AudioRecorder
from utils import LoggingSetup, RobustManager, ErrorHandler


class AILiveSystem:
    """
    AI直播系统主类
    整合所有模块，提供完整的AI直播功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        
        # 初始化日志
        LoggingSetup(log_dir=config.get('log_dir', './logs'))
        logging.info("AI Live System initializing...")
        
        # 初始化各模块
        self.vdb_manager = None
        self.main_model = None
        self.danmu_receiver = None
        self.sense_voice = None
        self.voice_trigger = None
        self.voice_output = None
        self.content_moderator = None
        self.image_moderator = None
        self.long_term_memory = None
        self.context_summarizer = None
        self.screen_manager = None
        self.screen_capture = None
        self.tool_executor = None
        self.audio_recorder = None
        self.webui_controller = None
        self.robust_manager = RobustManager()
        self.error_handler = ErrorHandler()
        
        # 纯聊天模式标志
        self.enable_danmu = config.get('enable_danmu', True)
        
        # 自动重启配置
        self.auto_restart = config.get('auto_restart', False)
        self.auto_restart_interval = config.get('auto_restart_interval', 3600)  # 默认1小时
        self.restart_timer = None
        
        # 消息队列
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # 初始化所有模块
        self._init_modules()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.danmu_thread = threading.Thread(target=self._run_danmu_receiver, daemon=True)
        
        logging.info("AI Live System initialized successfully")
        
        # 执行系统预热
        self._warmup_system()

    def _init_modules(self):
        """初始化所有模块"""
        try:
            # 1. 初始化向量数据库管理器（0.5b模型负责）
            logging.info("Initializing VDB Manager...")
            self.vdb_manager = VDBManager(self.config)
            
            # 2. 初始化长期记忆系统
            logging.info("Initializing Long Term Memory...")
            self.long_term_memory = LongTermMemory(self.config, self.vdb_manager)
            # 注意：ContextSummarizer 现在已在 LongTermMemory 中初始化和管理
            # self.context_summarizer = ContextSummarizer()
            
            # 3. 初始化主模型（VL:3b模型）
            logging.info("Initializing Main Model (Qwen2.5VL:3b)...")
            self.main_model = MainModel(self.config, self.vdb_manager)
            
            # 4. 初始化弹幕接收器（如果启用）
            if self.enable_danmu:
                logging.info("Initializing Danmu Receiver...")
                self.danmu_receiver = DanmuReceiver(self.config, self.vdb_manager, self.handle_message)
            else:
                logging.info("Danmu receiver disabled, running in chat-only mode")
            
            # 5. 初始化语音识别
            logging.info("Initializing SenseVoice...")
            self.sense_voice = SenseVoiceRecognizer(self.config)
            self.voice_trigger = VoiceTrigger(self.config, self.sense_voice)
            
            # 6. 初始化语音合成
            logging.info("Initializing Voice Output...")
            self.voice_output = VoiceOutputManager(self.config)
            
            # 7. 初始化内容审核
            logging.info("Initializing Content Moderation...")
            self.content_moderator = ContentModeration(self.config)
            self.image_moderator = ImageModeration(self.config)
            
            # 8. 初始化屏幕管理
            logging.info("Initializing Screen Manager...")
            self.screen_manager = ScreenManager(self.config)
            self.screen_capture = ScreenCapture(self.config)
            
            # 9. 初始化工具执行器
            logging.info("Initializing Tool Executor...")
            self.tool_executor = ToolExecutor(self.config)
            
            # 10. 初始化 WebUI 控制器
            logging.info("Initializing WebUI Controller...")
            try:
                from webui import WebUIController
                self.webui_controller = WebUIController(vdb_manager=self.vdb_manager)
            except ImportError as e:
                logging.warning(f"WebUI module not found: {e}. Run 'pip install Flask python-socketio' to enable WebUI.")
                self.webui_controller = None
            except Exception as e:
                logging.error(f"Error initializing WebUI Controller: {e}")
                self.webui_controller = None
            
            # 11. 初始化音频录制器
            logging.info("Initializing Audio Recorder...")
            self.audio_recorder = AudioRecorder(self.config, self.handle_speech_recognition)
            
            logging.info("All modules initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing modules: {e}")
            raise

    def _warmup_system(self):
        """
        系统预热功能，用于加载和初始化必要的模型和资源
        包括：Ollama模型加载、SenseVoice预热等
        """
        logging.info("Starting system warmup...")
        
        # 预热Ollama模型
        try:
            logging.info("Warming up Ollama models...")
            if self.main_model:
                # 使用现有的连接检查方法进行预热，确保模型已加载
                self.main_model._check_ollama_connection()
                
                # 发送一个简单的预热提示来确保模型完全加载
                logging.info("Loading text model: %s", self.main_model.ollama_model)
                if hasattr(self.main_model, 'ollama_api_url'):
                    import requests
                    response = requests.post(
                        self.main_model.ollama_api_url.replace('/chat', '/generate'),
                        json={
                            "model": self.main_model.ollama_model,
                            "prompt": "系统预热",
                            "stream": False,
                            "options": {"num_predict": 1}
                        },
                        timeout=20
                    )
                    if response.status_code == 200:
                        logging.info("Text model loaded successfully")
                    else:
                        logging.warning(f"Text model load response: {response.status_code}")
                
                # 预热视觉模型（如果有）
                if hasattr(self.main_model, 'ollama_vision_model') and self.main_model.ollama_vision_model:
                    logging.info("Loading vision model: %s", self.main_model.ollama_vision_model)
                    try:
                        response = requests.post(
                            self.main_model.ollama_api_url.replace('/chat', '/generate'),
                            json={
                                "model": self.main_model.ollama_vision_model,
                                "prompt": "系统预热",
                                "stream": False,
                                "options": {"num_predict": 1}
                            },
                            timeout=20
                        )
                        if response.status_code == 200:
                            logging.info("Vision model loaded successfully")
                        else:
                            logging.warning(f"Vision model load response: {response.status_code}")
                    except Exception as e:
                        logging.warning(f"Vision model warmup failed (this may be normal if no vision model is used): {e}")
        except Exception as e:
            logging.error(f"Ollama model warmup failed: {e}")
        
        # 预热SenseVoice
        try:
            logging.info("Warming up SenseVoice...")
            if self.sense_voice:
                # 检查SenseVoice API服务是否可用
                api_url = self.sense_voice.api_url
                logging.info(f"Checking SenseVoice API at: {api_url}")
                
                # 发送一个简单的请求来验证API是否响应
                try:
                    import requests
                    # 我们可以发送一个HEAD请求来检查服务是否在线
                    response = requests.head(api_url, timeout=10)
                    if response.status_code in [200, 405]:  # 200=成功, 405=方法不允许但服务在线
                        logging.info("SenseVoice API service is online")
                    else:
                        logging.warning(f"SenseVoice API returned status: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logging.error(f"SenseVoice API service check failed: {e}")
                    logging.warning("Starting SenseVoice service may be required")
                
                # 记录初始化状态
                logging.info("SenseVoice initialized and ready for use")
        except Exception as e:
            logging.error(f"SenseVoice warmup failed: {e}")
        
        logging.info("System warmup completed")
    
    def handle_message(self, message: str):
        """
        处理接收到的消息（弹幕、语音识别结果等）
        """
        try:
            # 将消息添加到处理队列
            self.message_queue.put({
                'type': 'user_message',
                'content': message,
                'timestamp': time.time()
            })
            logging.info(f"Message added to queue: {message[:50]}...")
            
        except Exception as e:
            logging.error(f"Error handling message: {e}")

    def handle_speech_recognition(self, recognized_text: str):
        """
        处理语音识别结果
        """
        try:
            # 将语音识别结果添加到处理队列
            self.message_queue.put({
                'type': 'speech_recognition',
                'content': recognized_text,
                'timestamp': time.time()
            })
            logging.info(f"Speech recognition result added to queue: {recognized_text[:50]}...")
            
        except Exception as e:
            logging.error(f"Error handling speech recognition: {e}")

    def _process_messages(self):
        """处理消息的后台线程"""
        while self.running:
            try:
                # 从队列获取消息
                try:
                    message_data = self.message_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                message_type = message_data['type']
                content = message_data['content']
                
                if message_type == 'user_message':
                    self._handle_user_message(content)
                elif message_type == 'speech_recognition':
                    self._handle_user_message(content)  # 语音识别结果和用户消息处理方式相同
                
                self.message_queue.task_done()
                
            except Exception as e:
                logging.error(f"Error processing messages: {e}")
                continue

    def _handle_user_message(self, message: str):
        """处理用户消息"""
        try:
            # 1. 内容审核
            if not self.content_moderator.moderate_text_for_response(message):
                logging.info(f"Message filtered by content moderation: {message[:50]}...")
                return
            
            # 2. 获取相关记忆
            context_memories = self.long_term_memory.search_contextual_memory(message, top_k=5)
            context_str = ""
            if context_memories:
                context_str = "相关上下文记忆：\n"
                for mem in context_memories[:3]:
                    context_str += f"- {mem['content']}\n"
            
            # 3. 准备工具列表
            tools = [
                {
                    'name': 'play_sound',
                    'description': '播放音效，参数：sound_file（音效文件名）'
                },
                {
                    'name': 'play_song',
                    'description': '播放歌曲，参数：song_file（歌曲文件名）'
                },
                {
                    'name': 'draw_image',
                    'description': '绘制图像，参数：prompt（绘图提示）'
                },
                {
                    'name': 'click_position',
                    'description': '点击屏幕位置，参数：x（X坐标）, y（Y坐标）'
                }
            ]
            
            # 4. 使用主模型处理消息
            result = self.main_model.process_message(
                f"{context_str}\n用户消息: {message}",
                tools=tools
            )
            
            if not result.get('success', False):
                logging.error(f"Error from main model: {result.get('error', 'Unknown error')}")
                return
            
            response = result.get('response', '')
            tool_calls = result.get('tool_calls', [])
            
            # 5. 内容审核响应
            if not self.content_moderator.moderate_text_for_response(response):
                logging.info(f"Response filtered by content moderation: {response[:50]}...")
                response = "不好意思，我不能回应这个内容。"
            
            # 6. 执行工具调用
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                arguments = tool_call['arguments']
                
                logging.info(f"Executing tool: {tool_name} with args: {arguments}")
                
                tool_result = self.tool_executor.execute_tool(tool_name, arguments, self.config)
                
                # 记录工具调用到长期记忆
                self.long_term_memory.add_tool_call_memory(tool_name, arguments, str(tool_result))
                
                # 如果是绘制图像，需要审核图像
                if tool_name == 'draw_image' and tool_result.get('success', False):
                    image_path = tool_result.get('image_path')
                    if image_path:
                        is_safe = self.image_moderator.moderate_generated_image(image_path)
                        if not is_safe:
                            logging.info(f"Generated image failed moderation: {image_path}")
            
            # 7. 语音合成和播放响应
            if response:
                # 获取当前时间作为instruct参考
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                instruct = f"当前时间 {current_time}，请用友好、生动的语调朗读以下内容："
                
                self.voice_output.speak(response, instruct)
            
            # 8. 将对话摘要添加到长期记忆
            full_context = f"用户: {message}\nAI: {response}"
            summary = self.long_term_memory.context_summarizer.summarize_context(full_context, detail_level='medium')
            self.long_term_memory.add_summary_memory(full_context, summary, detail_level='medium')
            
        except Exception as e:
            logging.error(f"Error handling user message: {e}")

    def _run_danmu_receiver(self):
        """运行弹幕接收器的后台线程"""
        while self.running:
            try:
                asyncio.run(self.danmu_receiver.start_receiving())
            except Exception as e:
                logging.error(f"Error in danmu receiver: {e}")
                # 等待一段时间后重试
                time.sleep(5)

    def start(self):
        """启动AI直播系统"""
        self.running = True
        
        # 启动处理线程
        self.processing_thread.start()
        
        # 根据配置决定是否启动弹幕线程
        if self.enable_danmu and self.danmu_receiver:
            self.danmu_thread.start()
            logging.info("Danmu receiver started")
        else:
            logging.info("Running in chat-only mode (danmu disabled)")
        
        # 启动音频录制器（如果启用语音识别功能）
        if self.config.get('enable_voice_recognition', True):
            self.audio_recorder.start_listening()
            
            # 设置按键监听
            self.audio_recorder.setup_key_listeners()
        
        logging.info("AI Live System started")
        
        # 如果启用自动重启，启动定时器
        if self.auto_restart:
            self._start_restart_timer()

    def stop(self):
        """停止AI直播系统"""
        self.running = False
        
        # 停止各模块
        if self.danmu_receiver and self.enable_danmu:
            self.danmu_receiver.stop_receiving()
        
        # 等待线程结束
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        if self.danmu_thread.is_alive() and self.enable_danmu:
            self.danmu_thread.join(timeout=5)
        
        # 保存上下文
        if self.main_model:
            try:
                self.main_model.save_context_to_file()
            except Exception as e:
                logging.error(f"Error saving context: {e}")
        
        # 清理各模块
        if self.main_model:
            self.main_model.cleanup()
        if self.sense_voice:
            self.sense_voice.cleanup()
        if self.voice_output:
            self.voice_output.cleanup()
        if self.content_moderator:
            self.content_moderator.cleanup()
        if self.image_moderator:
            self.image_moderator.cleanup()
        if self.audio_recorder:
            self.audio_recorder.cleanup()
        
        # 如果启用自动重启，启动定时器
        if self.auto_restart:
            self._start_restart_timer()
        
        # 取消重启定时器
        if self.restart_timer and self.restart_timer.is_alive():
            try:
                # 无法直接取消线程，但可以记录系统已停止
                pass
            except:
                pass
        
        logging.info("AI Live System stopped")

    def _start_restart_timer(self):
        """启动自动重启定时器"""
        def restart_task():
            import time
            time.sleep(self.auto_restart_interval)
            if self.running:  # 如果系统仍在运行，则执行重启
                logging.info(f"Auto restart triggered after {self.auto_restart_interval} seconds")
                self.restart()
        
        if self.restart_timer and self.restart_timer.is_alive():
            self.restart_timer.join(timeout=1)  # 等待之前的定时器结束
        
        self.restart_timer = threading.Thread(target=restart_task, daemon=True)
        self.restart_timer.start()
        logging.info(f"Auto restart timer started: {self.auto_restart_interval} seconds")

    def restart(self):
        """重启AI直播系统"""
        logging.info("Restarting AI Live System...")
        self.stop()  # 先停止当前系统
        time.sleep(2)  # 等待2秒确保完全停止
        self.start()  # 重新启动系统

    def add_manual_message(self, message: str):
        """添加手动消息"""
        self.handle_message(message)

    def update_system_memo(self, memo: str):
        """更新系统备忘录"""
        self.config['memo'] = memo
        logging.info(f"System memo updated: {memo}")


def load_config(config_path: str = 'config/config.json') -> Dict[str, Any]:
    """加载配置文件"""
    import json
    import os
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}, using default config")
        # 返回默认配置
        return {
            'danmu_websocket_uri': 'ws://localhost:9898',
            'danmu_task_ids': ['123456'],
            'sense_voice_api_url': 'http://127.0.0.1:8877/api/v1/asr',
            'cosy_voice_api_url': 'http://127.0.0.1:8888/api/tts',
            'use_molotts': False,
            'molotts_device': 'cpu',
            'molotts_language': 'ZH',
            'molotts_speaker': 'ZH',
            'ollama_api_url': 'http://localhost:11434/api/chat',
            'ollama_model': 'llama3.2',
            'ollama_vision_model': 'llava',
            'ollama_temperature': 0.7,
            'ollama_top_p': 0.9,
            'ollama_max_tokens': 2048,
            'enable_sound_effects': True,
            'enable_music': True,
            'enable_ai_drawing': True,
            'enable_screen_click': True,
            'enable_continuous_mode': True,
            'enable_voice_recognition': True,
            'enable_continuous_talk': False,
            'enable_wake_sleep': True,
            'wake_word': '唤醒',
            'sleep_word': '睡眠',
            'volume_threshold': 800.0,
            'silence_threshold': 15,
            'trigger_key': 'F11',
            'stop_trigger_key': 'F12',
            'audio_output_dir': './output_audio',
            'qwen2_5vl_mnn_path': 'qwen2_5vl_3b.mnn',
            'qwen_vl_api_url': 'http://127.0.0.1:8899/api/moderate',
            'qdrant_host': 'localhost',
            'qdrant_port': 6333,
            'system_prompt': '你是一个AI直播助手，正在直播中与观众互动。请友好、有趣地回应观众的弹幕和问题。今天是2024年12月14日，星期六。',
            'voice_trigger_volume': 0.05,
            'voice_trigger_enabled': True,
            'key_trigger_enabled': True,
            'output_audio_dir': './output_audio',
            'sound_effects_dir': './sound_effects',
            'songs_dir': './songs',
            'screenshot_dir': './screenshots',
            'images_dir': './images',
            'log_dir': './logs',
            'sensitive_keywords': ['敏感词1', '敏感词2'],
            'memo': ''
        }


def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 创建AI直播系统
    ai_system = AILiveSystem(config)
    
    # 用于控制程序退出的标志
    exit_event = threading.Event()
    
    # 信号处理函数
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal. Shutting down AI Live System...")
        exit_event.set()
    
    # 注册信号处理器
    try:
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except Exception as e:
        logging.warning(f"Could not register signal handlers: {e}")
    
    webui_thread = None
    try:
        # 启动系统
        ai_system.start()
        
        # 启动 WebUI (如果可用)
        if ai_system.webui_controller:
            def start_webui():
                try:
                    from webui import run_webui
                    run_webui(host='0.0.0.0', port=5000)
                except Exception as e:
                    logging.error(f"Error starting WebUI: {e}")
            
            webui_thread = threading.Thread(target=start_webui, daemon=True)
            webui_thread.start()
            print("WebUI started at http://localhost:5000")
        else:
            print("WebUI not available. Install Flask and python-socketio to enable WebUI.")
        
        print("AI Live System is running. Press Ctrl+C to stop.")
        
        # 保持主程序运行
        while not exit_event.is_set():
            try:
                # 使用较短的超时时间以便能及时响应退出信号
                exit_event.wait(timeout=1)
                
                # 可以在这里添加其他主循环逻辑
                # 例如：处理GUI事件、检查配置更新等
                
            except KeyboardInterrupt:
                print("\nShutting down AI Live System...")
                exit_event.set()
            
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        ai_system.stop()
        
        # 等待 WebUI 线程结束
        if webui_thread:
            try:
                webui_thread.join(timeout=2)  # 等待最多2秒
            except:
                pass
                
        print("AI Live System stopped.")


if __name__ == '__main__':
    main()