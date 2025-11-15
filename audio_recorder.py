import pyaudio
import wave
import numpy as np
import threading
import time
import logging
import keyboard
import queue
import os
from typing import Dict, Any, Callable, Optional

class AudioRecorder:
    """
    音频录制器 - 处理音频录制和语音识别触发
    """
    
    def __init__(self, config: Dict[str, Any], on_recognition_complete: Callable[[str], None]):
        self.config = config
        self.on_recognition_complete = on_recognition_complete
        
        # 音频参数
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        
        # 音频录制相关
        self.is_recording = False
        self.is_listening = False
        self.is_awake = True  # 是否处于唤醒状态
        
        # 用于录制的变量
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        
        # 任务队列
        self.recognition_queue = queue.Queue()
        
        # 启动音频识别线程
        self.recognition_thread = threading.Thread(target=self._process_recognition_queue, daemon=True)
        self.recognition_thread.start()
        
        logging.info("Audio recorder initialized")

    def start_listening(self):
        """开始监听音频"""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._listen_for_speech, daemon=True)
        self.listening_thread.start()
        logging.info("Started listening for speech")

    def stop_listening(self):
        """停止监听音频"""
        self.is_listening = False
        logging.info("Stopped listening for speech")

    def _listen_for_speech(self):
        """监听语音输入的后台线程"""
        # 打开音频流
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        # 检查功能开关
        enable_wake_sleep = self.config.get('enable_wake_sleep', True)
        enable_voice_recognition = self.config.get('enable_voice_recognition', True)
        enable_continuous_talk = self.config.get('enable_continuous_talk', False)
        
        if not enable_voice_recognition:
            logging.info("Voice recognition is disabled, stopping listener")
            return

        logging.info("Audio listener started, waiting for speech...")

        while self.is_listening:
            try:
                # 读取音频数据
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                max_dB = np.max(np.abs(audio_data))

                volume_threshold = self.config.get('volume_threshold', 800.0)

                # 如果音量超过阈值，开始录制
                if max_dB > volume_threshold:
                    if enable_wake_sleep and not self.is_awake:
                        # 如果处于睡眠状态，检查是否需要唤醒
                        logging.info("Listening for wake word...")
                        # 这里应该使用语音识别检查唤醒词，但为了简化，暂时跳过
                        continue
                        
                    logging.info("Speech detected, starting recording...")
                    self._record_audio()
                    
            except Exception as e:
                logging.error(f"Error in audio listening: {e}")
                time.sleep(0.1)  # 短暂休眠后继续

        # 关闭音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def _record_audio(self):
        """录制音频片段"""
        if self.is_recording:
            return  # 避免重复录制
            
        self.is_recording = True
        frames = []
        
        silent_threshold = self.config.get('silence_threshold', 15)
        silence_count = 0
        is_speaking = False
        
        logging.info("Recording started...")

        try:
            while self.is_listening:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                
                # 检查音量是否下降到沉默阈值以下
                audio_data = np.frombuffer(data, dtype=np.int16)
                max_dB = np.max(np.abs(audio_data))
                
                if max_dB > self.config.get('volume_threshold', 800.0):
                    is_speaking = True
                    silence_count = 0
                elif is_speaking:
                    silence_count += 1
                
                # 如果沉默时间超过阈值，停止录制
                if silence_count >= silent_threshold and is_speaking:
                    break
                    
        except Exception as e:
            logging.error(f"Error during recording: {e}")
        
        if frames:  # 如果有录制到音频
            # 保存音频文件
            audio_dir = self.config.get('audio_output_dir', './output_audio')
            os.makedirs(audio_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(audio_dir, filename)
            
            # 保存为WAV文件
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            logging.info(f"Audio recorded and saved to {filepath}")
            
            # 将文件路径添加到识别队列
            self.recognition_queue.put(filepath)
        
        self.is_recording = False

    def _process_recognition_queue(self):
        """处理语音识别队列的后台线程"""
        from sense_voice import SenseVoiceRecognizer
        
        # 初始化语音识别器
        recognizer = SenseVoiceRecognizer(self.config)
        
        while True:
            try:
                filepath = self.recognition_queue.get(timeout=1)
                
                if filepath is None:  # 结束信号
                    break
                
                # 使用SenseVoice进行语音识别
                recognized_text = recognizer.recognize(filepath)
                
                if recognized_text:
                    # 检查唤醒/睡眠词
                    wake_word = self.config.get('wake_word', '唤醒')
                    sleep_word = self.config.get('sleep_word', '睡眠')
                    
                    if wake_word in recognized_text:
                        self.is_awake = True
                        logging.info("Wake word detected, AI is now awake")
                        # 不执行后续处理，只是唤醒
                        continue
                    elif sleep_word in recognized_text:
                        self.is_awake = False
                        logging.info("Sleep word detected, AI is now asleep")
                        # 不执行后续处理，只是睡眠
                        continue
                    
                    # 如果处于唤醒状态，将识别结果传递给处理函数
                    if self.is_awake:
                        logging.info(f"Recognized text: {recognized_text}")
                        self.on_recognition_complete(recognized_text)
                    else:
                        logging.info("AI is in sleep mode, ignoring speech")
                
                # 删除临时音频文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                self.recognition_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing recognition: {e}")
                continue

    def start_manual_recording(self):
        """开始手动录音（通过按键触发）"""
        if self.config.get('enable_voice_recognition', True):
            self._record_audio()
        else:
            logging.warning("Voice recognition is disabled")

    def setup_key_listeners(self):
        """设置按键监听器"""
        trigger_key = self.config.get('trigger_key', 'F11')
        stop_trigger_key = self.config.get('stop_trigger_key', 'F12')
        
        def on_key_press(event):
            if event.name == trigger_key.lower():
                logging.info(f"Trigger key {trigger_key} pressed, starting recording...")
                self.start_manual_recording()
            elif event.name == stop_trigger_key.lower():
                logging.info(f"Stop trigger key {stop_trigger_key} pressed")
                
        # 注册按键监听
        keyboard.on_press(on_key_press)

    def cleanup(self):
        """清理资源"""
        self.is_listening = False
        self.recognition_queue.put(None)  # 发送结束信号
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        
        if self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=2)