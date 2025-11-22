import logging
import os
import subprocess
import threading
import queue
from typing import Dict, Any, Optional
import time
from PIL import Image

class ToolManager:
    """
    工具调用管理器 - 管理各种工具调用（音效、点歌、AI绘画等）
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sound_effects_dir = config.get('sound_effects_dir', './sound_effects')
        self.songs_dir = config.get('songs_dir', './songs')
        self.images_dir = config.get('images_dir', './images')
        
        # VDB管理器实例
        self.vdb_manager = None
        
        # 任务队列
        self.task_queue = queue.Queue()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_tool_tasks, daemon=True)
        self.processing_thread.start()
        
        logging.info("Tool manager initialized")
    
    def set_vdb_manager(self, vdb_manager):
        """设置VDB管理器实例"""
        self.vdb_manager = vdb_manager
        logging.info("VDB manager set in tool manager")

    def _process_tool_tasks(self):
        """处理工具任务的后台线程"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break
                    
                result = self._execute_tool(task['tool_name'], task['arguments'], task.get('config'))
                
                # 如果有回调函数，执行它
                if 'callback' in task:
                    try:
                        task['callback'](result)
                    except Exception as e:
                        logging.error(f"Error in tool callback: {e}")
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing tool task: {e}")
                continue

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行工具"""
        try:
            # 检查功能是否被启用
            if config:
                if tool_name == 'play_sound' and not config.get('enable_sound_effects', True):
                    return {'error': '音效功能已关闭', 'success': False}
                elif tool_name == 'play_song' and not config.get('enable_music', True):
                    return {'error': '点歌功能已关闭', 'success': False}
                elif tool_name == 'draw_image' and not config.get('enable_ai_drawing', True):
                    return {'error': 'AI绘画功能已关闭', 'success': False}
                elif tool_name == 'click_position' and not config.get('enable_screen_click', True):
                    return {'error': '屏幕点击功能已关闭', 'success': False}
                elif tool_name in ['start_continuous_mode', 'stop_continuous_mode'] and not config.get('enable_continuous_mode', True):
                    return {'error': '连续操作模式已关闭', 'success': False}
                elif tool_name == 'start_vdb_management' and not config.get('enable_llm_vdb_control', True):
                    return {'error': 'LLM控制VDB管理功能已关闭', 'success': False}
            
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
            elif tool_name == 'start_vdb_management':
                return self._start_vdb_management()
            else:
                return {'error': f'Unknown tool: {tool_name}', 'success': False}
                
        except Exception as e:
            logging.error(f"Error executing tool {tool_name}: {e}")
            return {'error': str(e), 'success': False}

    def play_sound(self, sound_file: str, config: Dict[str, Any] = None, callback: Optional[callable] = None) -> bool:
        """
        播放音效
        :param sound_file: 音效文件名
        :param config: 配置字典
        :param callback: 执行完成后的回调函数
        :return: 是否成功添加到任务队列
        """
        try:
            task = {
                'tool_name': 'play_sound',
                'arguments': {'sound_file': sound_file},
                'config': config,
                'callback': callback
            }
            self.task_queue.put(task)
            logging.info(f"Added sound play task: {sound_file}")
            return True
        except Exception as e:
            logging.error(f"Error adding sound play task: {e}")
            return False

    def play_song(self, song_file: str, config: Dict[str, Any] = None, callback: Optional[callable] = None) -> bool:
        """
        播放歌曲
        :param song_file: 歌曲文件名
        :param config: 配置字典
        :param callback: 执行完成后的回调函数
        :return: 是否成功添加到任务队列
        """
        try:
            task = {
                'tool_name': 'play_song',
                'arguments': {'song_file': song_file},
                'config': config,
                'callback': callback
            }
            self.task_queue.put(task)
            logging.info(f"Added song play task: {song_file}")
            return True
        except Exception as e:
            logging.error(f"Error adding song play task: {e}")
            return False

    def draw_image(self, prompt: str, config: Dict[str, Any] = None, callback: Optional[callable] = None) -> bool:
        """
        绘制图像
        :param prompt: 绘画提示
        :param config: 配置字典
        :param callback: 执行完成后的回调函数
        :return: 是否成功添加到任务队列
        """
        try:
            task = {
                'tool_name': 'draw_image',
                'arguments': {'prompt': prompt},
                'config': config,
                'callback': callback
            }
            self.task_queue.put(task)
            logging.info(f"Added image draw task: {prompt}")
            return True
        except Exception as e:
            logging.error(f"Error adding image draw task: {e}")
            return False

    def start_vdb_management(self, config: Dict[str, Any] = None, callback: Optional[callable] = None) -> bool:
        """
        启动VDB管理
        :param config: 配置字典
        :param callback: 执行完成后的回调函数
        :return: 是否成功添加到任务队列
        """
        try:
            task = {
                'tool_name': 'start_vdb_management',
                'arguments': {},
                'config': config,
                'callback': callback
            }
            self.task_queue.put(task)
            logging.info("Added VDB management start task")
            return True
        except Exception as e:
            logging.error(f"Error adding VDB management start task: {e}")
            return False

    def _play_sound(self, sound_file: str) -> Dict[str, Any]:
        """实际执行播放音效"""
        try:
            sound_path = os.path.join(self.sound_effects_dir, sound_file)
            
            if not os.path.exists(sound_path):
                return {'error': f'Sound file not found: {sound_path}', 'success': False}
            
            # 检查文件扩展名并选择适当的播放方法
            _, ext = os.path.splitext(sound_file.lower())
            
            if ext in ['.wav', '.mp3', '.ogg']:
                # 使用系统命令播放音频
                if os.name == 'nt':  # Windows
                    import winsound
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME)
                else:  # Unix-like systems
                    # 尝试使用aplay或mpg123等命令
                    if ext == '.mp3':
                        subprocess.run(['mpg123', sound_path], check=True)
                    else:
                        subprocess.run(['aplay', sound_path], check=True)
            else:
                return {'error': f'Unsupported audio format: {ext}', 'success': False}
            
            logging.info(f"Played sound: {sound_path}")
            return {'message': f'Played sound: {sound_file}', 'success': True}
            
        except subprocess.CalledProcessError as e:
            return {'error': f'Audio playback failed: {e}', 'success': False}
        except Exception as e:
            logging.error(f"Error playing sound: {e}")
            return {'error': str(e), 'success': False}

    def _play_song(self, song_file: str) -> Dict[str, Any]:
        """实际执行播放歌曲"""
        try:
            song_path = os.path.join(self.songs_dir, song_file)
            
            if not os.path.exists(song_path):
                return {'error': f'Song file not found: {song_path}', 'success': False}
            
            # 检查文件扩展名并选择适当的播放方法
            _, ext = os.path.splitext(song_file.lower())
            
            if ext in ['.wav', '.mp3', '.flac', '.m4a']:
                # 使用系统命令播放音频
                if os.name == 'nt':  # Windows
                    import winsound
                    winsound.PlaySound(song_path, winsound.SND_FILENAME)
                else:  # Unix-like systems
                    # 尝试使用合适的播放器
                    if ext == '.mp3':
                        subprocess.run(['mpg123', song_path], check=True)
                    elif ext in ['.flac', '.wav']:
                        subprocess.run(['aplay', song_path], check=True)
                    else:
                        subprocess.run(['vlc', song_path], check=True)
            else:
                return {'error': f'Unsupported audio format: {ext}', 'success': False}
            
            logging.info(f"Played song: {song_path}")
            return {'message': f'Played song: {song_file}', 'success': True}
            
        except subprocess.CalledProcessError as e:
            return {'error': f'Audio playback failed: {e}', 'success': False}
        except Exception as e:
            logging.error(f"Error playing song: {e}")
            return {'error': str(e), 'success': False}

    def _draw_image(self, prompt: str) -> Dict[str, Any]:
        """实际执行绘制图像 - 使用Table Diffusion或其他AI绘画服务"""
        try:
            # 这里需要集成Table Diffusion或类似的AI绘画服务
            # 以下是一个框架，实际实现需要调用具体的AI绘画API
            
            # 生成图像文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_filename = f"generated_{timestamp}_{hash(prompt) % 10000}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            
            # 临时实现：创建一个简单的占位图像
            # 在实际应用中，这里会调用AI绘画服务
            image = Image.new('RGB', (512, 512), color='lightblue')
            image.save(image_path)
            
            # 模拟AI绘画过程（实际应用中会调用API）
            logging.info(f"Drew image with prompt: {prompt}")
            logging.info(f"Image saved to: {image_path}")
            
            return {
                'message': f'Drew image with prompt: {prompt}',
                'image_path': image_path,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error drawing image: {e}")
            return {'error': str(e), 'success': False}

    def _start_vdb_management(self) -> Dict[str, Any]:
        """实际执行启动VDB管理"""
        try:
            if not self.vdb_manager:
                return {'error': 'VDB管理器未初始化', 'success': False}
            
            result = self.vdb_manager.start_vdb_management()
            logging.info(f"VDB management start result: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Error starting VDB management: {e}")
            return {'error': str(e), 'success': False}

class AudioPlayer:
    """
    音频播放器 - 专门处理音频播放
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sound_effects_dir = config.get('sound_effects_dir', './sound_effects')
        self.songs_dir = config.get('songs_dir', './songs')
        
    def play_file(self, filepath: str) -> bool:
        """播放音频文件"""
        try:
            _, ext = os.path.splitext(filepath.lower())
            
            if os.name == 'nt':  # Windows
                import winsound
                winsound.PlaySound(filepath, winsound.SND_FILENAME)
            else:  # Unix-like systems
                if ext == '.mp3':
                    subprocess.run(['mpg123', filepath], check=True)
                else:
                    subprocess.run(['aplay', filepath], check=True)
            
            return True
        except Exception as e:
            logging.error(f"Error playing audio file: {e}")
            return False

class ImageGenerator:
    """
    图像生成器 - 专门处理图像生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.images_dir = config.get('images_dir', './images')
        self.table_diffusion_api_url = config.get('table_diffusion_api_url', '')
        
    def generate_image(self, prompt: str) -> Optional[str]:
        """
        生成图像
        :param prompt: 生成提示
        :return: 图像文件路径
        """
        try:
            # 这里需要集成Table Diffusion或其他AI绘画服务
            # 以下是一个框架
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_filename = f"generated_{timestamp}_{hash(prompt) % 10000}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            
            # 临时实现：创建一个简单的占位图像
            # 在实际应用中，这里会调用Table Diffusion API
            image = Image.new('RGB', (512, 512), color='lightblue')
            image.save(image_path)
            
            logging.info(f"Generated image with prompt: {prompt}, saved to {image_path}")
            return image_path
            
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            return None

class ToolExecutor:
    """
    工具执行器 - 提供统一的工具执行接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.tool_manager = ToolManager(config)
        self.audio_player = AudioPlayer(config)
        self.image_generator = ImageGenerator(config)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行工具"""
        if tool_name == 'play_sound':
            sound_file = arguments.get('sound_file', '')
            return self.tool_manager._execute_tool(tool_name, {'sound_file': sound_file}, config)
        elif tool_name == 'play_song':
            song_file = arguments.get('song_file', '')
            return self.tool_manager._execute_tool(tool_name, {'song_file': song_file}, config)
        elif tool_name == 'draw_image':
            prompt = arguments.get('prompt', '')
            return self.tool_manager._execute_tool(tool_name, {'prompt': prompt}, config)
        elif tool_name == 'click_position':
            x = arguments.get('x', 0)
            y = arguments.get('y', 0)
            return self.tool_manager._execute_tool(tool_name, {'x': x, 'y': y}, config)
        elif tool_name == 'start_continuous_mode':
            return self.tool_manager._execute_tool(tool_name, {}, config)
        elif tool_name == 'stop_continuous_mode':
            return self.tool_manager._execute_tool(tool_name, {}, config)
        else:
            return {'error': f'Unknown tool: {tool_name}', 'success': False}