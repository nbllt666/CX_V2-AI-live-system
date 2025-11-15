import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
import threading
import queue
from PIL import Image
import io
import base64
import sseclient
import time
from functools import wraps
import os
import urllib.error

def retry_on_failure(max_retries=3, delay=1, backoff=2):
    """
    装饰器：为函数添加重试机制
    
    Args:
        max_retries (int): 最大重试次数，默认为3
        delay (float): 初始延迟时间（秒），默认为1
        backoff (int): 延迟时间倍数，默认为2
        
    Returns:
        function: 包装后的函数，具有自动重试功能
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logging.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise e
                    
                    logging.warning(f"Function {func.__name__} failed (attempt {retries}/{max_retries}), retrying in {current_delay}s: {str(e)}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

class MainModel:
    """
    主模型类，负责处理AI对话逻辑和工具调用
    使用Ollama API进行对话生成，支持流式响应、工具调用、连续操作模式等功能
    
    Attributes:
        config: 配置字典
        vdb_manager: 向量数据库管理器
        ollama_api_url: Ollama API的URL
        ollama_model: 用于文本的模型名称
        ollama_vision_model: 用于图像理解的模型名称
        continuous_mode: 是否处于连续操作模式
        task_queue: 任务队列
        conversation_history: 对话历史记录
        max_history_length: 最大对话历史长度
        config_file_path: 配置文件路径（用于动态加载）
    """
    
    def __init__(self, config: Dict[str, Any], vdb_manager, config_file_path: str = None):
        self.config = config
        self.vdb_manager = vdb_manager
        self.config_file_path = config_file_path  # 保存配置文件路径用于动态加载
        
        # Ollama API 配置
        self.ollama_api_url = self.config.get('ollama_api_url', 'http://localhost:11434/api/chat')
        self.ollama_model = self.config.get('ollama_model', 'llama3.2')
        self.ollama_vision_model = self.config.get('ollama_vision_model', 'llava')  # 用于图像理解的模型
        
        # 检查Ollama服务是否可用
        if self._check_ollama_connection():
            logging.info("Connected to Ollama successfully")
        else:
            logging.error("Failed to connect to Ollama")
            
        # 工具调用状态
        self.continuous_mode = False
        self.current_task = None
        self.continuous_mode_thread = None
        self.continuous_mode_stop_event = threading.Event()
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.processing_thread.start()
        
        # 内存管理相关
        self.conversation_history = []  # 存储对话历史
        self.max_history_length = self.config.get('max_conversation_history', 50)  # 最大对话历史长度
        self.context_window_size = self.config.get('context_window_size', 10)  # 上下文窗口大小（句数）
        self.last_cleanup_time = time.time()
        self.cleanup_interval = self.config.get('memory_cleanup_interval', 300)  # 5分钟清理一次
        self.context_file_path = self.config.get('context_file_path', 'context_history.json')  # 上下文文件路径
        
        # 尝试从文件加载上下文
        self.load_context_from_file()
        
        logging.info("Main model (Ollama) initialized successfully")
    
    @retry_on_failure(max_retries=2, delay=1, backoff=1)
    def _check_ollama_connection(self) -> bool:
        """检查Ollama连接"""
        try:
            # 尝试加载模型以检查连接
            response = requests.post(
                self.ollama_api_url.replace('/chat', '/generate'),  # 使用generate端点测试连接
                json={
                    "model": self.ollama_model,
                    "prompt": "ping",
                    "stream": False,
                    "options": {"num_predict": 10}
                },
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logging.error(f"Error checking Ollama connection: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error checking Ollama connection: {e}")
            return False

    def _process_tasks(self):
        """处理任务的后台线程"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break
                    
                response = self._execute_task(task)
                self.response_queue.put(response)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing task: {e}")
                continue

    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务"""
        try:
            message = task.get('message', '')
            image_path = task.get('image_path', None)
            tools = task.get('tools', [])
            
            # 构建消息，包括对话历史
            full_prompt = self._build_prompt(message, tools)
            
            # 使用Ollama API生成响应
            response = self._generate_with_ollama(full_prompt, image_path, tools)
            
            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)
            
            # 添加对话记录到历史
            self.add_conversation_entry(message, response)
            
            return {
                'response': response,
                'tool_calls': tool_calls,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error executing task: {e}")
            return {
                'error': str(e),
                'success': False
            }

    def _build_prompt_with_context(self, message: str, tools: List[Dict], include_context: bool = True) -> str:
        """构建包含上下文的提示词"""
        # 获取当前时间 - 使用更准确的表述
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        current_date = datetime.now().strftime("%Y年%m月%d日")
        current_weekday = datetime.now().strftime("%A")
        weekday_map = {
            "Monday": "星期一",
            "Tuesday": "星期二",
            "Wednesday": "星期三",
            "Thursday": "星期四",
            "Friday": "星期五",
            "Saturday": "星期六",
            "Sunday": "星期日"
        }
        current_weekday_cn = weekday_map.get(current_weekday, current_weekday)
        current_date_cn = f"{current_date} {current_weekday_cn}"

        # 获取相关的记忆信息
        context_memories = self.vdb_manager.search_memory(message, top_k=5)

        context_str = ""
        if context_memories:
            context_str = "相关上下文记忆：\n"
            for mem in context_memories[:3]:  # 只取前3个最相关的
                context_str += f"- {mem['content']}\n"

        # 构建基础系统提示词
        system_prompt = self.config.get('system_prompt', '你是一个AI直播助手。')

        # 替换系统提示词中的时间占位符
        system_prompt = system_prompt.replace("{{CURRENT_DATE}}", current_date_cn)

        # 根据功能开关添加功能状态信息
        disabled_features = []
        if not self.config.get('enable_sound_effects', True):
            disabled_features.append("音效功能已关闭")
        if not self.config.get('enable_music', True):
            disabled_features.append("点歌功能已关闭")
        if not self.config.get('enable_ai_drawing', True):
            disabled_features.append("AI绘画功能已关闭")
        if not self.config.get('enable_screen_click', True):
            disabled_features.append("屏幕点击功能已关闭")
        if not self.config.get('enable_continuous_mode', True):
            disabled_features.append("连续操作模式已关闭")
        if not self.config.get('enable_voice_recognition', True):
            disabled_features.append("语音识别功能已关闭")

        if disabled_features:
            system_prompt += f"\n\n已禁用功能: {', '.join(disabled_features)}"

        system_prompt = self.vdb_manager.update_system_prompt_with_memo(
            system_prompt,
            self.config.get('memo', '')
        )

        # 详细工具说明
        detailed_tool_descriptions = []

        if self.config.get('enable_sound_effects', True):
            detailed_tool_descriptions.append(
                "play_sound: 播放音效文件。参数：sound_file (音效文件名，必须是sound_effects目录中存在的文件)。"
                "示例：[PLAY_SOUND:surprise.wav]"
            )

        if self.config.get('enable_music', True):
            detailed_tool_descriptions.append(
                "play_song: 播放歌曲文件。参数：song_file (歌曲文件名，必须是songs目录中存在的文件)。"
                "示例：[PLAY_SONG:happy_song.mp3]"
            )

        if self.config.get('enable_ai_drawing', True):
            detailed_tool_descriptions.append(
                "draw_image: 生成图像。参数：prompt (用于生成图像的详细描述)。"
                "示例：[DRAW_IMAGE:一只可爱的小猫在草地上玩耍]"
            )

        if self.config.get('enable_screen_click', True):
            detailed_tool_descriptions.append(
                "click_position: 点击屏幕指定位置。参数：x (X轴坐标), y (Y轴坐标)。"
                "示例：[CLICK:500,300]"
            )

        if self.config.get('enable_continuous_mode', True):
            detailed_tool_descriptions.append(
                "start_continuous_mode: 开始连续操作模式，进入此模式后你将能连续接收屏幕截图并执行操作。"
                "示例：进入连续操作模式"
            )
            detailed_tool_descriptions.append(
                "stop_continuous_mode: 停止连续操作模式。"
                "示例：停止，退出连续操作模式"
            )

        # 添加连续对话和唤醒/睡眠功能说明
        if self.config.get('enable_voice_recognition', True):
            wake_word = self.config.get('wake_word', '唤醒')
            sleep_word = self.config.get('sleep_word', '睡眠')

            detailed_tool_descriptions.append(
                f"语音识别功能：系统支持语音输入。如果需要唤醒AI，请使用唤醒词 '{wake_word}'。"
                f"如果需要让AI进入睡眠状态，请使用睡眠词 '{sleep_word}'。"
                f"在连续对话模式下，AI会持续监听语音输入，直到检测到睡眠词或手动停止。"
            )

        # 动态构建可用工具列表，根据功能开关
        available_tools_names = set()
        if self.config.get('enable_sound_effects', True):
            available_tools_names.add('play_sound')
        if self.config.get('enable_music', True):
            available_tools_names.add('play_song')
        if self.config.get('enable_ai_drawing', True):
            available_tools_names.add('draw_image')
        if self.config.get('enable_screen_click', True):
            available_tools_names.add('click_position')
        if self.config.get('enable_continuous_mode', True):
            available_tools_names.add('start_continuous_mode')
            available_tools_names.add('stop_continuous_mode')

        # 只保留在功能开关中启用的工具
        enabled_tools = [tool for tool in tools if tool['name'] in available_tools_names]

        # 构建工具描述字符串
        tools_description = ""
        if detailed_tool_descriptions:
            tools_description = "\n工具调用说明：\n"
            for desc in detailed_tool_descriptions:
                tools_description += f"{desc}\n"

        # 获取最近的对话上下文，确保系统提示词始终在上下文中
        recent_context_messages = ""
        if include_context:
            # 使用VDB管理器的多上下文功能
            effective_context = self.vdb_manager.get_effective_context(system_prompt, "medium")

            # 从有效上下文构建对话历史
            for item in effective_context:
                if item.get('role') == 'user':
                    recent_context_messages += f"用户: {item.get('content', '')}\n"
                elif item.get('role') == 'assistant':
                    recent_context_messages += f"AI: {item.get('content', '')}\n"
                elif item.get('role') == 'system':
                    # 系统消息已包含在system_prompt中，不需要重复显示
                    pass

        if recent_context_messages:
            recent_context_messages = "\n最近的对话历史：\n" + recent_context_messages

        # 构建最终提示词
        full_prompt = f"""{system_prompt}

当前时间: {current_time}
{recent_context_messages}

{context_str}

{tools_description}

用户消息: {message}

请根据上述信息生成回复。如果需要使用工具，请在回复中明确说明。"""

        return full_prompt
        
    def _build_prompt(self, message: str, tools: List[Dict]) -> str:
        """构建提示词（兼容旧方法）"""
        return self._build_prompt_with_context(message, tools)

    @retry_on_failure(max_retries=3, delay=1, backoff=2)
    def _generate_with_ollama_stream(self, prompt: str, image_path: Optional[str] = None, tools: List[Dict] = None, callback=None) -> str:
        """使用Ollama API流式生成响应"""
        try:
            # 构建消息列表，包含上下文历史
            # 使用新方法获取包含上下文的提示
            context_prompt = self._build_prompt_with_context(prompt, tools or [], include_context=True)
            
            # 准备消息列表
            messages = [
                {
                    "role": "user",
                    "content": context_prompt
                }
            ]
            
            # 如果有图像，添加到消息中
            if image_path:
                # 将图像编码为base64
                try:
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                except FileNotFoundError:
                    logging.error(f"Image file not found: {image_path}")
                    return f"抱歉，图像文件不存在: {image_path}"
                except Exception as e:
                    logging.error(f"Error reading image file: {e}")
                    return f"抱歉，读取图像文件时出错: {str(e)}"
                
                # 使用适合图像理解的模型
                model_to_use = self.ollama_vision_model
                messages[0]["images"] = [image_data]
            else:
                # 使用普通模型
                model_to_use = self.ollama_model
            
            # 准备请求参数 - 开启流式传输
            # 根据是否是视觉模型使用不同的最大token数
            max_tokens = (self.config.get('ollama_vision_max_tokens', 1024) 
                         if image_path else self.config.get('ollama_max_tokens', 2048))
            
            data = {
                "model": model_to_use,
                "messages": messages,
                "stream": True,  # 流式响应
                "options": {
                    "temperature": self.config.get('ollama_temperature', 0.7),
                    "top_p": self.config.get('ollama_top_p', 0.9),
                    "num_predict": max_tokens
                }
            }
            
            # 如果有工具，添加到请求中
            if tools:
                data["tools"] = tools
            
            # 发送请求到Ollama API
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.ollama_api_url, json=data, headers=headers, timeout=60, stream=True)
            
            # 检查请求是否成功
            if response.status_code != 200:
                logging.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"抱歉，AI模型暂时无法响应。错误代码: {response.status_code}"
            
            # 使用 sseclient 解析流式响应
            client = sseclient.SSEClient(response)
            full_response = ""
            tool_calls = []
            
            for event in client.events():
                if event.data == '[DONE]':
                    break
                    
                try:
                    result = json.loads(event.data)
                    
                    # 检查是否完成
                    done = result.get('done', False)
                    
                    # 获取内容
                    content = result.get('message', {}).get('content', '')
                    if content:
                        full_response += content
                        
                        # 如果提供了回调函数，实时调用
                        if callback:
                            callback(content)
                    
                    # 检查是否有工具调用
                    message_tool_calls = result.get('message', {}).get('tool_calls', [])
                    if message_tool_calls:
                        tool_calls.extend(message_tool_calls)
                    
                    # 如果完成，跳出循环
                    if done:
                        break
                        
                except json.JSONDecodeError:
                    logging.warning(f"Failed to decode JSON: {event.data}")
                    continue
                except Exception as e:
                    logging.error(f"Error processing stream event: {e}")
                    continue
            
            # 处理工具调用
            if tool_calls:
                logging.info(f"Ollama responded with tool calls: {tool_calls}")
                # 将工具调用信息添加到响应中
                tool_call_str = "\n".join([f"[TOOL_CALL:{tc['function']['name']}({tc['function']['arguments']})]" for tc in tool_calls])
                full_response += f"\n{tool_call_str}"
            
            logging.info(f"Ollama stream response completed: {full_response[:100]}...")
            return full_response
            
        except requests.exceptions.Timeout:
            logging.error("Ollama API request timed out")
            return "抱歉，AI模型响应超时，请稍后再试。"
        except requests.exceptions.ConnectionError:
            logging.error("Failed to connect to Ollama API")
            return "抱歉，无法连接到AI模型服务，请检查服务是否正常运行。"
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Ollama API for streaming: {e}")
            return "抱歉，AI模型暂时无法连接，请稍后再试。"
        except Exception as e:
            logging.error(f"Unexpected error generating with Ollama stream: {e}")
            return "抱歉，AI模型处理时出现未知错误。"

    def _generate_with_ollama(self, prompt: str, image_path: Optional[str] = None, tools: List[Dict] = None) -> str:
        """使用Ollama API生成响应（使用流式实现）"""
        return self._generate_with_ollama_stream(prompt, image_path, tools)

    def _generate_with_api(self, prompt: str, image_path: Optional[str] = None, tools: List[Dict] = None) -> str:
        """使用Ollama API生成响应"""
        return self._generate_with_ollama(prompt, image_path, tools)

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """解析工具调用"""
        try:
            # 实现工具调用解析逻辑
            # 检查响应中是否包含工具调用指令
            tool_calls = []
            
            # 示例：检测工具调用指令
            if "[PLAY_SOUND:" in response:
                import re
                pattern = r'\[PLAY_SOUND:(.*?)\]'
                matches = re.findall(pattern, response)
                for sound in matches:
                    tool_calls.append({
                        'name': 'play_sound',
                        'arguments': {'sound_file': sound.strip()}
                    })
            
            if "[PLAY_SONG:" in response:
                import re
                pattern = r'\[PLAY_SONG:(.*?)\]'
                matches = re.findall(pattern, response)
                for song in matches:
                    tool_calls.append({
                        'name': 'play_song',
                        'arguments': {'song_file': song.strip()}
                    })
            
            if "[DRAW_IMAGE:" in response:
                import re
                pattern = r'\[DRAW_IMAGE:(.*?)\]'
                matches = re.findall(pattern, response)
                for prompt in matches:
                    tool_calls.append({
                        'name': 'draw_image',
                        'arguments': {'prompt': prompt.strip()}
                    })
            
            if "[CLICK:" in response:
                import re
                pattern = r'\[CLICK:(\d+),(\d+)\]'
                matches = re.findall(pattern, response)
                for x, y in matches:
                    tool_calls.append({
                        'name': 'click_position',
                        'arguments': {'x': int(x), 'y': int(y)}
                    })
            
            if "CONTINUOUS_MODE" in response.upper():
                tool_calls.append({
                    'name': 'start_continuous_mode',
                    'arguments': {}
                })
            
            if "STOP" in response.upper() and self.continuous_mode:
                tool_calls.append({
                    'name': 'stop_continuous_mode',
                    'arguments': {}
                })
            
            return tool_calls
            
        except Exception as e:
            logging.error(f"Error parsing tool calls: {e}")
            return []

    def process_message(self, message: str, image_path: Optional[str] = None, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        处理消息并返回AI响应
        
        将消息添加到处理队列，并等待处理结果。
        
        Args:
            message (str): 用户输入的消息
            image_path (str, optional): 图像文件路径，用于图像理解
            tools (List[Dict], optional): 可用工具列表
            
        Returns:
            Dict[str, Any]: 包含处理结果的字典
                - response (str): AI的响应文本
                - tool_calls (List[Dict]): 解析出的工具调用
                - success (bool): 处理是否成功
                - error (str, optional): 错误信息（如果处理失败）
        """
        try:
            task = {
                'message': message,
                'image_path': image_path,
                'tools': tools or []
            }
            
            self.task_queue.put(task)
            
            # 等待响应
            response = self.response_queue.get(timeout=60)  # 60秒超时
            return response
            
        except queue.Empty:
            logging.error("Timeout waiting for model response")
            return {
                'error': 'Timeout waiting for model response',
                'success': False
            }
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def process_message_stream(self, message: str, image_path: Optional[str] = None, tools: Optional[List[Dict]] = None, callback=None) -> Dict[str, Any]:
        """
        处理消息（流式响应）
        
        以流式方式处理消息，可以实时接收AI的响应。
        
        Args:
            message (str): 用户输入的消息
            image_path (str, optional): 图像文件路径，用于图像理解
            tools (List[Dict], optional): 可用工具列表
            callback (callable, optional): 回调函数，用于实时处理流式响应片段
            
        Returns:
            Dict[str, Any]: 包含处理结果的字典
                - response (str): AI的完整响应文本
                - tool_calls (List[Dict]): 解析出的工具调用
                - success (bool): 处理是否成功
                - error (str, optional): 错误信息（如果处理失败）
        """
        try:
            # 构建消息
            full_prompt = self._build_prompt(message, tools or [])
            
            # 使用Ollama API流式生成响应
            response = self._generate_with_ollama_stream(full_prompt, image_path, tools, callback)
            
            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)
            
            return {
                'response': response,
                'tool_calls': tool_calls,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error processing message stream: {e}")
            return {
                'error': str(e),
                'success': False
            }

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        try:
            if tool_name == 'play_sound':
                return self._play_sound(arguments.get('sound_file', ''))
            elif tool_name == 'play_song':
                return self._play_song(arguments.get('song_file', ''))
            elif tool_name == 'draw_image':
                return self._draw_image(arguments.get('prompt', ''))
            elif tool_name == 'click_position':
                return self._click_position(arguments.get('x', 0), arguments.get('y', 0))
            elif tool_name == 'start_continuous_mode':
                return self._start_continuous_mode()
            elif tool_name == 'stop_continuous_mode':
                return self._stop_continuous_mode()
            else:
                return {'error': f'Unknown tool: {tool_name}', 'success': False}
                
        except Exception as e:
            logging.error(f"Error calling tool {tool_name}: {e}")
            return {'error': str(e), 'success': False}

    def _play_sound(self, sound_file: str) -> Dict[str, Any]:
        """播放音效"""
        try:
            import os
            import pygame
            import time

            sound_path = os.path.join(self.config.get('sound_effects_dir', 'sound_effects'), sound_file)

            if not os.path.exists(sound_path):
                return {'error': f'Sound file not found: {sound_path}', 'success': False}

            # 初始化pygame mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            # 加载并播放音效
            sound = pygame.mixer.Sound(sound_path)
            sound.play()

            # 等待声音播放完成
            while pygame.mixer.get_busy():
                time.sleep(0.1)

            logging.info(f"Playing sound: {sound_path}")

            return {'message': f'Playing sound: {sound_file}', 'success': True}

        except ImportError:
            # 如果pygame不可用，尝试使用winsound (Windows) 或其他方法
            import os
            try:
                if os.name == 'nt':  # Windows
                    import winsound
                    sound_path = os.path.join(self.config.get('sound_effects_dir', 'sound_effects'), sound_file)
                    if not os.path.exists(sound_path):
                        return {'error': f'Sound file not found: {sound_path}', 'success': False}
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME)
                    logging.info(f"Playing sound: {sound_path}")
                    return {'message': f'Playing sound: {sound_file}', 'success': True}
                else:
                    # 尝试使用系统命令播放
                    sound_path = os.path.join(self.config.get('sound_effects_dir', 'sound_effects'), sound_file)
                    if not os.path.exists(sound_path):
                        return {'error': f'Sound file not found: {sound_path}', 'success': False}
                    import subprocess
                    subprocess.run(['aplay', sound_path], check=True)
                    logging.info(f"Playing sound: {sound_path}")
                    return {'message': f'Playing sound: {sound_file}', 'success': True}
            except Exception as e:
                logging.error(f"Error playing sound with fallback method: {e}")
                return {'error': f'Error playing sound: {str(e)}', 'success': False}
        except Exception as e:
            logging.error(f"Error playing sound: {e}")
            return {'error': str(e), 'success': False}

    def _play_song(self, song_file: str) -> Dict[str, Any]:
        """播放歌曲"""
        try:
            import os
            import pygame
            import time

            song_path = os.path.join(self.config.get('songs_dir', 'songs'), song_file)

            if not os.path.exists(song_path):
                return {'error': f'Song file not found: {song_path}', 'success': False}

            # 初始化pygame mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            # 加载并播放歌曲
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()

            # 等待音乐播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            logging.info(f"Playing song: {song_path}")

            return {'message': f'Playing song: {song_file}', 'success': True}

        except ImportError:
            # 如果pygame不可用，尝试使用winsound (Windows) 或其他方法
            import os
            try:
                if os.name == 'nt':  # Windows
                    import winsound
                    song_path = os.path.join(self.config.get('songs_dir', 'songs'), song_file)
                    if not os.path.exists(song_path):
                        return {'error': f'Song file not found: {song_path}', 'success': False}
                    winsound.PlaySound(song_path, winsound.SND_FILENAME)
                    logging.info(f"Playing song: {song_path}")
                    return {'message': f'Playing song: {song_file}', 'success': True}
                else:
                    # 尝试使用系统命令播放
                    song_path = os.path.join(self.config.get('songs_dir', 'songs'), song_file)
                    if not os.path.exists(song_path):
                        return {'error': f'Song file not found: {song_path}', 'success': False}
                    import subprocess
                    subprocess.run(['aplay', song_path], check=True)
                    logging.info(f"Playing song: {song_path}")
                    return {'message': f'Playing song: {song_file}', 'success': True}
            except Exception as e:
                logging.error(f"Error playing song with fallback method: {e}")
                return {'error': f'Error playing song: {str(e)}', 'success': False}
        except Exception as e:
            logging.error(f"Error playing song: {e}")
            return {'error': str(e), 'success': False}

    @retry_on_failure(max_retries=3, delay=2, backoff=2)
    def _draw_image(self, prompt: str) -> Dict[str, Any]:
        """绘制图像"""
        try:
            import http.client
            import json
            import urllib.request
            import os
            from datetime import datetime
            
            logging.info(f"Drawing image with prompt: {prompt}")
            
            # 从配置中获取 API 密钥
            sd_api_key = self.config.get('sd_api_key', '')
            if not sd_api_key:
                return {
                    'error': 'Stable Diffusion API key not configured',
                    'success': False
                }
            
            # 准备请求数据
            payload = json.dumps({
                "key": sd_api_key,
                "prompt": prompt,
                "negative_prompt": "((out of frame)), ((extra fingers)), mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), (((tiling))), ((naked)), ((tile)), ((fleshpile)), ((ugly)), (((abstract))), blurry, ((bad anatomy)), ((bad proportions)), ((extra limbs)), cloned face, glitchy, ((extra breasts)), ((double torso)), ((extra arms)), ((extra hands)), ((mangled fingers)), ((missing breasts)), (missing lips), ((ugly face)), ((fat)), ((extra legs))",
                "width": "512",
                "height": "512",
                "samples": "1",
                "num_inference_steps": "20",
                "safety_checker": "no",
                "enhance_prompt": "yes",
                "seed": None,
                "guidance_scale": 7.5,
                "multi_lingual": "no",
                "panorama": "no",
                "self_attention": "no",
                "upscale": "no",
                "embeddings_model": None,
                "webhook": None,
                "track_id": None
            })
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 发送请求到 Stable Diffusion API
            conn = None
            try:
                conn = http.client.HTTPSConnection("stablediffusionapi.com", timeout=120)
                conn.request("POST", "/api/v3/text2img", payload, headers)
                res = conn.getresponse()
                response_data = res.read()
                conn.close()
            except http.client.HTTPException as e:
                if conn:
                    conn.close()
                logging.error(f"HTTP error when calling Stable Diffusion API: {e}")
                return {
                    'error': f'Stable Diffusion API HTTP error: {str(e)}',
                    'success': False
                }
            except Exception as e:
                if conn:
                    conn.close()
                logging.error(f"Network error when calling Stable Diffusion API: {e}")
                return {
                    'error': f'Network error calling Stable Diffusion API: {str(e)}',
                    'success': False
                }
            
            # 解析响应
            try:
                result = json.loads(response_data.decode("utf-8"))
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, response: {response_data}")
                return {
                    'error': 'Invalid response from Stable Diffusion API',
                    'success': False
                }
            except UnicodeDecodeError as e:
                logging.error(f"Response decode error: {e}")
                return {
                    'error': 'Response encoding error from Stable Diffusion API',
                    'success': False
                }
            
            if result.get("status") == "success":
                # 获取图像 URL
                image_urls = result.get("output", [])
                if image_urls:
                    image_url = image_urls[0]
                    
                    try:
                        # 下载图像到本地
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        local_image_path = os.path.join(
                            self.config.get('ai_drawing_output_dir', 'generated_images'), 
                            f"generated_{timestamp}.png"
                        )
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(local_image_path), exist_ok=True)
                        
                        # 下载图像
                        urllib.request.urlretrieve(image_url, local_image_path)
                        
                        logging.info(f"Image generated successfully: {local_image_path}")
                        return {
                            'message': f'Image generated successfully with prompt: {prompt}',
                            'image_path': local_image_path,
                            'success': True
                        }
                    except urllib.error.URLError as e:
                        logging.error(f"Failed to download image: {e}")
                        return {
                            'error': f'Failed to download generated image: {str(e)}',
                            'success': False
                        }
                    except Exception as e:
                        logging.error(f"Error saving generated image: {e}")
                        return {
                            'error': f'Error saving generated image: {str(e)}',
                            'success': False
                        }
                else:
                    return {
                        'error': 'No image URL returned from API',
                        'success': False
                    }
            else:
                error_message = result.get('message', result.get('error', 'Unknown error from Stable Diffusion API'))
                logging.error(f"Stable Diffusion API error: {error_message}")
                return {
                    'error': f'Stable Diffusion API error: {error_message}',
                    'success': False
                }
            
        except Exception as e:
            logging.error(f"Unexpected error drawing image: {e}")
            return {'error': f'Unexpected error: {str(e)}', 'success': False}

    def _click_position(self, x: int, y: int) -> Dict[str, Any]:
        """点击屏幕位置"""
        try:
            import pyautogui
            pyautogui.click(x, y)
            logging.info(f"Clicked position: ({x}, {y})")
            
            return {'message': f'Clicked position: ({x}, {y})', 'success': True}
            
        except Exception as e:
            logging.error(f"Error clicking position: {e}")
            return {'error': str(e), 'success': False}

    def _start_continuous_mode(self, screen_capture_callback=None) -> Dict[str, Any]:
        """
        开始连续操作模式
        
        连续操作模式会定期截取屏幕、分析内容并执行AI建议的操作。
        该功能在单独的线程中运行，直到被明确停止。
        
        Args:
            screen_capture_callback (callable, optional): 屏幕截图回调函数，
                应返回截图文件路径
            
        Returns:
            Dict[str, Any]: 包含操作结果的字典
                - success (bool): 操作是否成功
                - message (str): 操作结果消息
        """
        if self.continuous_mode:
            return {'message': 'Continuous mode is already running', 'success': False}
        
        self.continuous_mode = True
        self.continuous_mode_stop_event.clear()
        
        # 启动连续模式处理线程
        self.continuous_mode_thread = threading.Thread(
            target=self._run_continuous_mode, 
            args=(screen_capture_callback,), 
            daemon=True
        )
        self.continuous_mode_thread.start()
        
        logging.info("Started continuous mode")
        return {'message': 'Started continuous mode', 'success': True}

    def _stop_continuous_mode(self) -> Dict[str, Any]:
        """停止连续操作模式"""
        self.continuous_mode = False
        self.continuous_mode_stop_event.set()
        
        # 等待线程结束（最多等待5秒）
        if self.continuous_mode_thread and self.continuous_mode_thread.is_alive():
            self.continuous_mode_thread.join(timeout=5)
        
        self.current_task = None
        logging.info("Stopped continuous mode")
        
        return {'message': 'Stopped continuous mode', 'success': True}

    def _run_continuous_mode(self, screen_capture_callback=None):
        """
        连续模式的核心运行逻辑
        
        在循环中不断截取屏幕，分析内容，并执行AI建议的操作。
        该方法在独立线程中运行。
        
        Args:
            screen_capture_callback (callable): 屏幕截图回调函数，
                应返回截图文件路径
        """
        try:
            while self.continuous_mode and not self.continuous_mode_stop_event.is_set():
                # 获取屏幕截图
                if screen_capture_callback:
                    try:
                        screenshot_path = screen_capture_callback()
                        if screenshot_path and os.path.exists(screenshot_path):
                            # 分析屏幕截图并生成操作
                            response = self._analyze_screenshot_and_get_action(screenshot_path)
                            
                            if response and response.get('success', False):
                                # 执行操作
                                self._execute_continuous_action(response.get('response', ''))
                                
                        # 等待一段时间再进行下一次截图
                        for _ in range(5):  # 每秒检查一次，总共等待5秒
                            if self.continuous_mode_stop_event.is_set():
                                break
                            time.sleep(1)
                    except Exception as e:
                        logging.error(f"Error in continuous mode: {e}")
                        time.sleep(2)  # 出错后等待2秒再重试
                else:
                    logging.error("No screen capture callback provided for continuous mode")
                    break
        except Exception as e:
            logging.error(f"Error running continuous mode: {e}")
        finally:
            # 确保连续模式被正确关闭
            self.continuous_mode = False
            self.continuous_mode_stop_event.clear()

    def _analyze_screenshot_and_get_action(self, screenshot_path: str) -> Dict[str, Any]:
        """分析屏幕截图并获取操作指令"""
        try:
            # 构建提示词，告诉AI当前处于连续操作模式
            prompt = ("你当前处于连续操作模式。请分析屏幕截图并提供下一步操作指令。"
                     "你可以使用的操作包括：点击位置、执行其他已定义的工具。"
                     "请在你的回复中明确说明你应该执行什么操作。")
            
            # 使用Ollama API分析屏幕截图
            response = self._generate_with_ollama_stream(prompt, image_path=screenshot_path)
            
            if response:
                return {
                    'response': response,
                    'success': True
                }
            else:
                return {
                    'error': 'No response from AI model',
                    'success': False
                }
        except Exception as e:
            logging.error(f"Error analyzing screenshot: {e}")
            return {
                'error': str(e),
                'success': False
            }

    def _execute_continuous_action(self, action_response: str):
        """执行从AI返回的操作指令"""
        try:
            # 解析并执行工具调用
            tool_calls = self._parse_tool_calls(action_response)
            
            if tool_calls:
                for tool_call in tool_calls:
                    result = self.call_tool(tool_call['name'], tool_call['arguments'])
                    
                    if not result.get('success', False):
                        logging.error(f"Failed to execute tool {tool_call['name']}: {result.get('error', 'Unknown error')}")
                    else:
                        logging.info(f"Successfully executed tool {tool_call['name']}")
            else:
                # 没有找到明确的工具调用，但可能在文本中有指示
                # 可以根据配置决定是否记录或处理这类情况
                pass
        except Exception as e:
            logging.error(f"Error executing continuous action: {e}")

    def _cleanup_memory(self):
        """
        清理内存，移除旧的对话历史和临时文件
        
        此方法会：
        1. 限制对话历史长度不超过最大值
        2. 删除过期的临时文件（如截图和生成的图像）
        """
        try:
            # 限制对话历史长度
            if len(self.conversation_history) > self.max_history_length:
                # 保留较新的历史记录
                self.conversation_history = self.conversation_history[-self.max_history_length:]
                logging.debug(f"Trimmed conversation history to last {self.max_history_length} entries")
            
            # 清理临时文件（如果有）
            # 例如清理过期的临时图像文件等
            temp_dirs = [
                self.config.get('screenshot_dir', './screenshots'),
                self.config.get('ai_drawing_output_dir', './generated_images')
            ]
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for file_name in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file_name)
                        try:
                            # 检查是否为临时文件或过期文件（可根据具体需求定制）
                            # 这里我们移除超过24小时的文件
                            if os.path.isfile(file_path):
                                file_age = time.time() - os.path.getmtime(file_path)
                                if file_age > 24 * 3600:  # 24小时
                                    os.remove(file_path)
                                    logging.debug(f"Removed old temporary file: {file_path}")
                        except Exception as e:
                            logging.warning(f"Could not remove temporary file {file_path}: {e}")
            
            self.last_cleanup_time = time.time()
            logging.debug("Memory cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during memory cleanup: {e}")

    def _should_cleanup(self) -> bool:
        """检查是否需要进行内存清理"""
        return (time.time() - self.last_cleanup_time) > self.cleanup_interval

    def add_conversation_entry(self, user_message: str, ai_response: str):
        """添加对话记录到历史"""
        entry = {
            'timestamp': datetime.now(),
            'user_message': user_message,
            'ai_response': ai_response
        }
        self.conversation_history.append(entry)
        
        # 检查是否需要清理内存
        if self._should_cleanup():
            self._cleanup_memory()
    
    def save_context_to_file(self):
        """保存上下文到文件（使用多文件策略）"""
        # 从对话历史中提取最近的对话
        recent_context = self.get_recent_conversation_history(self.context_window_size)
        if self.vdb_manager:
            # 使用多文件保存策略
            return self.vdb_manager.save_multiple_context_files(recent_context, self.config.get('system_prompt', ''))
        return False

    def load_context_from_file(self):
        """从文件加载上下文（使用多文件策略）"""
        if self.vdb_manager:
            # 加载多上下文文件
            context_data = self.vdb_manager.load_multiple_context_files()
            # 默认加载中期上下文
            loaded_context = context_data.get('medium_term', [])
            if loaded_context:
                self.conversation_history = loaded_context
                logging.info(f"Loaded {len(loaded_context)} conversation entries from medium-term context file")
    
    def get_recent_context_messages(self, n: int = None) -> List[Dict[str, str]]:
        """获取最近的上下文消息，用于发送给模型"""
        if n is None:
            n = self.context_window_size
            
        recent_history = self.get_recent_conversation_history(n)
        messages = []
        
        for entry in recent_history:
            # 添加用户消息
            if 'user_message' in entry:
                messages.append({
                    'role': 'user',
                    'content': entry['user_message']
                })
            # 添加AI响应
            if 'ai_response' in entry:
                messages.append({
                    'role': 'assistant', 
                    'content': entry['ai_response']
                })
        
        return messages

    def get_recent_conversation_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的对话历史"""
        return self.conversation_history[-n:] if len(self.conversation_history) >= n else self.conversation_history

    def load_config(self, config_file_path: str = None) -> bool:
        """
        动态加载配置文件
        
        从指定的JSON配置文件中加载配置，并更新当前实例的配置。
        
        Args:
            config_file_path (str, optional): 配置文件路径。
                如果未提供，则使用实例初始化时的配置文件路径。
                
        Returns:
            bool: 配置加载是否成功
        """
        try:
            path_to_use = config_file_path or self.config_file_path
            if not path_to_use:
                logging.error("No config file path provided for dynamic loading")
                return False
            
            # 读取配置文件
            with open(path_to_use, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
            
            # 更新配置
            old_config = self.config
            self.config = new_config
            
            # 更新相关的配置项
            self.ollama_api_url = self.config.get('ollama_api_url', 'http://localhost:11434/api/chat')
            self.ollama_model = self.config.get('ollama_model', 'llama3.2')
            self.ollama_vision_model = self.config.get('ollama_vision_model', 'llava')
            self.max_history_length = self.config.get('max_conversation_history', 50)
            self.cleanup_interval = self.config.get('memory_cleanup_interval', 300)
            
            logging.info(f"Configuration reloaded from {path_to_use}")
            
            # 如果配置有重要更改，可能需要重新初始化某些组件
            if old_config.get('ollama_api_url') != self.config.get('ollama_api_url'):
                logging.info("Ollama API URL changed, checking connection...")
                if not self._check_ollama_connection():
                    logging.error("Failed to connect to new Ollama API URL")
            
            return True
            
        except FileNotFoundError:
            logging.error(f"Config file not found: {config_file_path or self.config_file_path}")
            return False
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return False

    def update_config(self, key: str, value: Any) -> bool:
        """更新单个配置项"""
        try:
            # 更新内存中的配置
            self.config[key] = value
            
            # 如果有配置文件路径，也更新文件
            if self.config_file_path:
                try:
                    with open(self.config_file_path, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                    
                    file_config[key] = value
                    
                    with open(self.config_file_path, 'w', encoding='utf-8') as f:
                        json.dump(file_config, f, ensure_ascii=False, indent=2)
                    
                    logging.info(f"Configuration updated: {key} = {value}")
                except Exception as file_error:
                    logging.error(f"Error updating config file: {file_error}")
                    # 即使文件更新失败，内存中的配置已更新，所以返回True
                    return True
            
            # 根据配置项更新相关设置
            if key == 'ollama_api_url':
                self.ollama_api_url = value
            elif key == 'ollama_model':
                self.ollama_model = value
            elif key == 'ollama_vision_model':
                self.ollama_vision_model = value
            elif key == 'max_conversation_history':
                self.max_history_length = int(value)
            elif key == 'memory_cleanup_interval':
                self.cleanup_interval = int(value)
            
            return True
            
        except Exception as e:
            logging.error(f"Error updating config: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        # 清理连续模式线程
        if self.continuous_mode:
            self._stop_continuous_mode()
        
        # 执行内存清理
        try:
            self._cleanup_memory()
        except Exception as e:
            logging.error(f"Error during final memory cleanup: {e}")
        
        logging.info("Main model resources cleaned up")