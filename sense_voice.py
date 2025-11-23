import logging
import requests
import threading
import queue
from typing import Dict, Any, Optional
import time

class SenseVoiceRecognizer:
    """
    SenseVoice 语音识别模块
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_url = config.get('sense_voice_api_url', 'http://127.0.0.1:8877/api/v1/asr')
        self.api_key = config.get('sense_voice_api_key', '')
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_audio_files, daemon=True)
        self.processing_thread.start()
        
        logging.info("SenseVoice recognizer initialized")

    def _process_audio_files(self):
        """处理音频文件的后台线程"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break
                    
                result = self._recognize_audio(task['file_path'], task.get('lang', 'auto'))
                self.result_queue.put({
                    'task_id': task['task_id'],
                    'result': result,
                    'timestamp': time.time()
                })
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing audio: {e}")
                continue

    def _recognize_audio(self, file_path: str, lang: str = 'auto') -> Optional[str]:
        """识别音频文件"""
        try:
            # 设置请求头
            headers = {
                "accept": "application/json",
            }
            
            # 添加API密钥（如果有的话）
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 准备文件
            with open(file_path, 'rb') as audio_file:
                files = [
                    ("files", (file_path.split('/')[-1], audio_file, "audio/wav")),
                ]
                
                # 修正API参数，使用正确的参数名
                data = {
                    "keys": "asr",  # 修改为更合适的值，表示ASR任务
                    "lang": lang
                }

                # 发送请求
                response = requests.post(self.api_url, headers=headers, files=files, data=data)

                # 检查响应
                if response.status_code == 200:
                    result_data = response.json()
                    if 'result' in result_data and len(result_data['result']) > 0:
                        recognized_text = result_data['result'][0].get('text', '')
                        logging.info(f"Recognized text: {recognized_text}")
                        return recognized_text
                    else:
                        logging.warning("No text found in recognition result")
                        return None
                else:
                    logging.error(f"Request failed with status code: {response.status_code}")
                    logging.error(f"Response: {response.text}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error recognizing audio {file_path}: {e}")
            return None

    def recognize(self, file_path: str, lang: str = 'auto', task_id: Optional[str] = None) -> Optional[str]:
        """
        识别音频文件
        :param file_path: 音频文件路径
        :param lang: 语言 ('auto', 'zh', 'en', etc.)
        :param task_id: 任务ID（可选）
        :return: 识别结果文本
        """
        try:
            task = {
                'file_path': file_path,
                'lang': lang,
                'task_id': task_id or str(int(time.time() * 1000))
            }
            
            self.task_queue.put(task)
            
            # 等待结果（这里可以调整超时时间）
            result_item = self.result_queue.get(timeout=30)  # 30秒超时
            return result_item['result']
            
        except queue.Empty:
            logging.error("Timeout waiting for recognition result")
            return None
        except Exception as e:
            logging.error(f"Error in recognize method: {e}")
            return None

    def cleanup(self):
        """清理资源"""
        self.task_queue.put(None)  # 发送结束信号
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)  # 最多等待5秒

class VoiceTrigger:
    """
    语音触发模块 - 监听麦克风音量并触发语音识别
    """
    
    def __init__(self, config: Dict[str, Any], recognizer: SenseVoiceRecognizer):
        self.config = config
        self.recognizer = recognizer
        self.trigger_volume = config.get('voice_trigger_volume', 0.05)  # 默认触发音量
        self.trigger_enabled = config.get('voice_trigger_enabled', False)
        self.key_trigger_enabled = config.get('key_trigger_enabled', False)
        
        # 音频录制相关
        self.recording = False
        self.audio_data = []
        
        logging.info("Voice trigger initialized")

    def enable_voice_trigger(self, enabled: bool):
        """启用/禁用语音触发"""
        self.trigger_enabled = enabled
        logging.info(f"Voice trigger {'enabled' if enabled else 'disabled'}")

    def enable_key_trigger(self, enabled: bool):
        """启用/禁用按键触发"""
        self.key_trigger_enabled = enabled
        logging.info(f"Key trigger {'enabled' if enabled else 'disabled'}")

    def set_trigger_volume(self, volume: float):
        """设置触发音量阈值"""
        self.trigger_volume = max(0.0, min(1.0, volume))  # 限制在0-1之间
        logging.info(f"Trigger volume set to {volume}")

    def check_volume_and_trigger(self, audio_level: float) -> bool:
        """
        检查音量并决定是否触发录音
        :param audio_level: 当前音频音量水平 (0.0-1.0)
        :return: 是否触发了录音
        """
        if not self.trigger_enabled:
            return False
            
        if audio_level > self.trigger_volume:
            logging.info(f"Audio level {audio_level} exceeded trigger volume {self.trigger_volume}")
            # 这里应该启动录音和识别流程
            # 注意：在实际实现中，你需要集成音频录制功能
            return True
            
        return False

    def manual_trigger(self):
        """手动触发录音"""
        if not self.key_trigger_enabled:
            logging.info("Key trigger is disabled")
            return False
            
        logging.info("Manual trigger activated")
        # 这里应该启动录音流程
        # 注意：在实际实现中，你需要集成音频录制功能
        return True