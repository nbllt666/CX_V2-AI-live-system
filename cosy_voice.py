import logging
import requests
import threading
import queue
import os
import subprocess
import tempfile
import re
from typing import Dict, Any, Optional
import time

class CosyVoiceSynthesizer:
    """
    CosyVoice 语音合成模块
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cosyvoice_api_url = config.get('cosyvoice_api_url', 'http://127.0.0.1:50000/voice')
        self.melotts_api_url = config.get('melotts_api_url', 'http://127.0.0.1:8888/tts')
        self.api_key = config.get('cosyvoice_api_key', '')
        self.enable_melotts = config.get('enable_melotts', False)
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.processing_thread.start()
        
        logging.info("CosyVoice synthesizer initialized")

    def _process_tasks(self):
        """处理任务的后台线程"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break
                    
                audio_path = self._synthesize_text(task['text'], task.get('instruct', ''))
                self.result_queue.put({
                    'task_id': task['task_id'],
                    'audio_path': audio_path,
                    'timestamp': time.time()
                })
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing synthesis task: {e}")
                continue

    def _synthesize_text(self, text: str, instruct: str = "") -> Optional[str]:
        """合成语音文本"""
        # 优先使用CosyVoice
        if not self.enable_melotts:
            return self._synthesize_with_cosyvoice(text, instruct)
        
        # 如果启用了MeloTTS，尝试使用MeloTTS
        melotts_result = self._synthesize_with_melotts(text)
        if melotts_result:
            return melotts_result
        
        # 如果MeloTTS失败，回退到CosyVoice
        return self._synthesize_with_cosyvoice(text, instruct)

    def _synthesize_with_cosyvoice(self, text: str, instruct: str = "") -> Optional[str]:
        """使用CosyVoice API合成语音"""
        try:
            # 生成唯一的音频文件名
            timestamp = int(time.time() * 1000)
            audio_path = os.path.join("uploads", f"cosyvoice_{timestamp}.wav")
            
            # 确保上传目录存在
            os.makedirs("uploads", exist_ok=True)
            
            # 设置请求头
            headers = {
                "accept": "application/json",
            }
            
            # 添加API密钥（如果有的话）
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 准备请求数据
            data = {
                "text": text,
                "instruct": instruct
            }

            # 发送请求
            response = requests.post(self.cosyvoice_api_url, headers=headers, json=data)

            # 检查响应
            if response.status_code == 200:
                # 保存音频文件
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Audio synthesized with CosyVoice API and saved to {audio_path}")
                return audio_path
            else:
                logging.error(f"CosyVoice API request failed with status code: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return self._fallback_synthesize(text, audio_path)
                
        except Exception as e:
            logging.error(f"Error synthesizing audio with CosyVoice API: {e}")
            return self._fallback_synthesize(text, f"uploads/cosyvoice_fallback_{int(time.time() * 1000)}.wav")

    def _synthesize_with_melotts(self, text: str) -> Optional[str]:
        """使用MeloTTS API合成语音（同步方式）"""
        try:
            # 生成唯一的音频文件名
            timestamp = int(time.time() * 1000)
            audio_path = os.path.join("uploads", f"melotts_{timestamp}.wav")
            
            # 确保上传目录存在
            os.makedirs("uploads", exist_ok=True)
            
            # 设置请求头
            headers = {
                "accept": "application/json",
            }
            
            # 添加API密钥（如果有的话）
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 准备请求数据
            data = {
                "text": text,
                "language": "EN",  # 默认英文
                "speaker_id": 0    # 默认说话人
            }

            # 发送请求（同步方式）
            response = requests.post(self.melotts_api_url, headers=headers, json=data)

            # 检查响应
            if response.status_code == 200:
                # 保存音频文件
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Audio synthesized with MeloTTS API and saved to {audio_path}")
                return audio_path
            else:
                logging.error(f"MeloTTS API request failed with status code: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error synthesizing audio with MeloTTS API: {e}")
            return None

    def _fallback_synthesize(self, text: str, audio_path: str) -> Optional[str]:
        """备用语音合成方法"""
        try:
            # 尝试使用系统TTS（如espeak等）
            # 这里只是一个示例，实际实现可能需要根据可用的TTS引擎调整
            import tempfile

            # 创建临时文本文件
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(text)
                temp_file_path = temp_file.name

            # 使用espeak（如果安装的话）生成语音
            cmd = [
                "espeak",
                "-w", audio_path,
                "-ven+f3",  # 英文+女声
                "-s", "150",  # 速度
                "-p", "50",   # 音调
                "-f", temp_file_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # 删除临时文件
            os.unlink(temp_file_path)

            if result.returncode == 0:
                logging.info(f"Audio synthesized with fallback method and saved to {audio_path}")
                return audio_path
            else:
                logging.error(f"Fallback synthesis failed: {result.stderr}")
                return None

        except FileNotFoundError:
            logging.error("espeak not found, fallback synthesis failed")
            return None
        except Exception as e:
            logging.error(f"Error in fallback synthesis: {e}")
            return None

    def synthesize(self, text: str, instruct: str = "", task_id: Optional[str] = None) -> Optional[str]:
        """
        合成语音
        :param text: 要合成的文本
        :param instruct: 指令（用于调整语调、情感等）
        :param task_id: 任务ID（可选）
        :return: 音频文件路径
        """
        try:
            task = {
                'text': text,
                'instruct': instruct,
                'task_id': task_id or str(int(time.time() * 1000))
            }

            self.task_queue.put(task)

            # 等待结果（这里可以调整超时时间）
            result_item = self.result_queue.get(timeout=60)  # 60秒超时
            return result_item['audio_path']

        except queue.Empty:
            logging.error("Timeout waiting for synthesis result")
            return None
        except Exception as e:
            logging.error(f"Error in synthesize method: {e}")
            return None

    def play_audio(self, audio_path: str):
        """播放音频文件"""
        try:
            # 使用系统默认音频播放器播放音频
            if os.name == 'nt':  # Windows
                import winsound
                winsound.PlaySound(audio_path, winsound.SND_FILENAME)
            else:  # Unix-like systems
                subprocess.run(["aplay", audio_path])  # 或使用其他播放命令如 mpg123, afplay等

            logging.info(f"Audio played: {audio_path}")

        except Exception as e:
            logging.error(f"Error playing audio {audio_path}: {e}")

    def cleanup(self):
        """清理资源"""
        self.task_queue.put(None)  # 发送结束信号
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)  # 最多等待5秒

class VoiceOutputManager:
    """
    语音输出管理器 - 管理文本到语音的转换和播放
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.synthesizer = CosyVoiceSynthesizer(config)
        self.output_queue = queue.Queue()

        # 启动输出处理线程
        self.output_thread = threading.Thread(target=self._process_output_queue, daemon=True)
        self.output_thread.start()

        logging.info("Voice output manager initialized")

    def _process_output_queue(self):
        """处理输出队列的后台线程"""
        while True:
            try:
                item = self.output_queue.get(timeout=1)
                if item is None:  # 结束信号
                    break

                audio_path = self.synthesizer.synthesize(item['text'], item.get('instruct', ''))
                if audio_path:
                    self.synthesizer.play_audio(audio_path)
                else:
                    logging.error(f"Failed to synthesize audio for text: {item['text'][:50]}...")

                self.output_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing output: {e}")
                continue

    def _split_text_by_punctuation(self, text: str, max_sentences: int = 3) -> list[str]:
        """
        按标点符号将文本分割成段落，每段最多包含指定数量的句子
        :param text: 要分割的文本
        :param max_sentences: 每段最多包含的句子数量
        :return: 分割后的文本段落列表
        """
        # 匹配中文和英文的句号、问号、感叹号
        sentence_endings = r'[。！？.!?]'
        sentences = re.split(f'({sentence_endings})', text)

        # 重新组合句子和标点
        combined_sentences = []
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i]
            punctuation = sentences[i+1] if i+1 < len(sentences) else ''
            combined_sentences.append(sentence + punctuation)

        # 如果没有匹配到标点符号，则按每3句话分段
        if not combined_sentences:
            # 按空格切分，然后每max_sentences个词组成一段
            words = text.split()
            combined_sentences = []
            for i in range(0, len(words), max_sentences * 5):  # 假设每句大约5个词
                combined_sentences.append(' '.join(words[i:i+max_sentences*5]))

        # 按max_sentences合并句子组合
        result = []
        for i in range(0, len(combined_sentences), max_sentences):
            chunk = ''.join(combined_sentences[i:i+max_sentences])
            if chunk.strip():  # 只添加非空段落
                result.append(chunk.strip())

        return result

    def speak(self, text: str, instruct: str = ""):
        """
        将文本转换为语音并播放（带文本分割功能）
        :param text: 要朗读的文本
        :param instruct: 指令（用于调整语调、情感等）
        """
        try:
            # 按标点符号分割文本，每3句一段
            text_chunks = self._split_text_by_punctuation(text, max_sentences=3)
            
            for chunk in text_chunks:
                if not chunk.strip():
                    continue
                    
                # 将文本块添加到输出队列
                self.output_queue.put({
                    'text': chunk,
                    'instruct': instruct
                })
                
                logging.info(f"Added text chunk to output queue: {chunk[:50]}...")
            
        except Exception as e:
            logging.error(f"Error in speak method: {e}")

    def cleanup(self):
        """清理资源"""
        self.output_queue.put(None)  # 发送结束信号
        if self.output_thread.is_alive():
            self.output_thread.join(timeout=5)  # 最多等待5秒
        
        # 清理合成器资源
        self.synthesizer.cleanup()