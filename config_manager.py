import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """配置验证错误异常"""
    pass

class ConfigManager:
    """配置管理器 - 提供配置加载、验证和自动检测功能"""
    
    def __init__(self, config_path: str = 'config/config.json'):
        self.config_path = config_path
        self.default_config = self._get_default_config()
        self.required_fields = [
            'ollama_api_url', 'ollama_model', 'ollama_vision_model',
            'sense_voice_api_url', 'cosy_voice_api_url',
            'qdrant_host', 'qdrant_port'
        ]
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
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
            'system_prompt': '你是一个AI直播助手，正在直播中与观众互动。请友好、有趣地回应观众的弹幕和问题。',
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
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            # 确保配置目录存在
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using default config")
            return self.default_config.copy()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise ConfigValidationError(f"配置文件格式错误: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise ConfigValidationError(f"加载配置文件时出错: {e}")
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """保存配置文件"""
        try:
            # 确保配置目录存在
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
                
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, list]:
        """验证配置文件"""
        errors = []
        
        # 检查必需字段
        for field in self.required_fields:
            if field not in config or not config[field]:
                errors.append(f"缺少必需配置项: {field}")
        
        # 检查API URL格式
        url_fields = ['ollama_api_url', 'sense_voice_api_url', 'cosy_voice_api_url']
        for field in url_fields:
            if field in config and config[field]:
                if not isinstance(config[field], str) or not config[field].startswith(('http://', 'https://')):
                    errors.append(f"{field} 必须是有效的URL地址")
        
        # 检查数字类型配置
        numeric_fields = {
            'ollama_temperature': (0, 1),
            'ollama_max_tokens': (1, 100000),
            'volume_threshold': (0, 10000),
            'silence_threshold': (0, 100),
            'voice_trigger_volume': (0, 1),
            'qdrant_port': (1, 65535)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config and config[field] is not None:
                try:
                    value = float(config[field])
                    if not (min_val <= value <= max_val):
                        errors.append(f"{field} 必须在 {min_val} 到 {max_val} 之间")
                except (ValueError, TypeError):
                    errors.append(f"{field} 必须是数字")
        
        # 检查布尔类型配置
        bool_fields = [
            'enable_sound_effects', 'enable_music', 'enable_ai_drawing',
            'enable_screen_click', 'enable_continuous_mode', 'enable_voice_recognition',
            'enable_continuous_talk', 'enable_wake_sleep', 'use_molotts',
            'voice_trigger_enabled', 'key_trigger_enabled'
        ]
        
        for field in bool_fields:
            if field in config and config[field] is not None:
                if not isinstance(config[field], bool):
                    errors.append(f"{field} 必须是布尔值 (true/false)")
        
        return len(errors) == 0, errors
    
    def auto_detect_config(self) -> Dict[str, Any]:
        """自动检测配置"""
        config = self.load_config()
        
        # 如果某些配置项为空，使用默认值填充
        for key, default_value in self.default_config.items():
            if key not in config or config[key] == "":
                config[key] = default_value
                logger.info(f"自动填充配置项 {key}: {default_value}")
        
        # 特殊处理Qdrant配置
        if not config.get('qdrant_host'):
            config['qdrant_host'] = 'localhost'
        if not config.get('qdrant_port'):
            config['qdrant_port'] = 6333
            
        # 确保目录路径存在
        dir_fields = ['audio_output_dir', 'sound_effects_dir', 'songs_dir', 
                     'screenshot_dir', 'images_dir', 'log_dir']
        for field in dir_fields:
            if field in config:
                dir_path = config[field]
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
        
        return config
    
    def get_config_with_validation(self) -> Dict[str, Any]:
        """获取经过验证的配置"""
        # 加载配置
        config = self.load_config()
        
        # 自动检测和填充
        config = self.auto_detect_config()
        
        # 验证配置
        is_valid, errors = self.validate_config(config)
        if not is_valid:
            logger.warning(f"配置验证失败: {errors}")
            # 不抛出异常，而是记录警告并返回配置
            for error in errors:
                logger.warning(f"配置错误: {error}")
        
        return config

# 全局配置管理器实例
config_manager = ConfigManager()

def load_config(config_path: str = 'config/config.json') -> Dict[str, Any]:
    """兼容旧版本的配置加载函数"""
    manager = ConfigManager(config_path)
    return manager.get_config_with_validation()