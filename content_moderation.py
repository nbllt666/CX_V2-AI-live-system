import logging
import requests
from typing import Dict, Any, Optional
import threading
import queue
import time
import json

class ContentModeration:
    """
    内容审核模块 - 使用Qwen-VL进行安全过滤
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_url = config.get('qwen_vl_api_url', 'http://127.0.0.1:8899/api/moderate')
        self.api_key = config.get('qwen_vl_api_key', '')
        self.ollama_api_url = config.get('ollama_api_url', 'http://localhost:11434/api/generate')
        self.moderation_model = config.get('moderation_model', 'llama3.2')
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_moderation_requests, daemon=True)
        self.processing_thread.start()
        
        logging.info("Content moderation module initialized")

    def _process_moderation_requests(self):
        """处理审核请求的后台线程"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break
                    
                result = self._moderate_content(task['content'], task.get('content_type', 'text'))
                
                self.result_queue.put({
                    'task_id': task['task_id'],
                    'result': result,
                    'timestamp': time.time()
                })
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing moderation: {e}")
                continue

    def _moderate_content(self, content: str, content_type: str = 'text') -> Dict[str, Any]:
        """审核内容"""
        try:
            # 设置请求头
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json"
            }
            
            # 添加API密钥（如果有的话）
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 准备数据
            data = {
                "content": content,
                "content_type": content_type,
                "moderation_policies": [
                    "hate_speech",
                    "violence",
                    "nudity",
                    "profanity",
                    "discrimination"
                ]
            }

            # 发送请求
            response = requests.post(self.api_url, headers=headers, json=data)

            # 检查响应
            if response.status_code == 200:
                result_data = response.json()
                logging.info(f"Moderation result: {result_data}")
                return result_data
            else:
                logging.error(f"Moderation failed with status code: {response.status_code}")
                logging.error(f"Response: {response.text}")
                
                # 如果审核服务不可用，使用更保守的策略
                return {
                    "is_safe": False,
                    "reason": "Moderation service unavailable, blocking content as precaution",
                    "original_content": content
                }
                
        except Exception as e:
            logging.error(f"Error moderating content: {e}")
            # 如果审核出错，采取保守策略，标记为不安全
            return {
                "is_safe": False,
                "reason": f"Moderation error: {str(e)}",
                "original_content": content
            }

    def moderate_text(self, text: str, task_id: Optional[str] = None) -> bool:
        """
        审核文本内容
        :param text: 要审核的文本
        :param task_id: 任务ID（可选）
        :return: True表示内容安全，False表示内容不安全
        """
        try:
            task = {
                'content': text,
                'content_type': 'text',
                'task_id': task_id or str(int(time.time() * 1000))
            }
            
            self.task_queue.put(task)
            
            # 等待结果（这里可以调整超时时间）
            result_item = self.result_queue.get(timeout=30)  # 30秒超时
            return result_item['result'].get('is_safe', False)
            
        except queue.Empty:
            logging.error("Timeout waiting for moderation result")
            # 超时情况下，保守地认为内容不安全
            return False
        except Exception as e:
            logging.error(f"Error in moderate_text method: {e}")
            # 错误情况下，保守地认为内容不安全
            return False

    def moderate_image(self, image_path: str, task_id: Optional[str] = None) -> bool:
        """
        审核图像内容
        :param image_path: 图像文件路径
        :param task_id: 任务ID（可选）
        :return: True表示内容安全，False表示内容不安全
        """
        try:
            # 读取图像文件
            with open(image_path, 'rb') as image_file:
                # 这里需要根据实际的API来调整
                # 可能需要将图像编码为base64或直接作为文件上传
                
                # 由于我们没有具体的API实现，这里提供一个框架
                return True  # 临时返回True
                
        except Exception as e:
            logging.error(f"Error in moderate_image method: {e}")
            return False

    def moderate_text_with_ollama(self, text: str) -> Dict[str, Any]:
        """
        使用 Ollama API 进行 AI 辅助内容审核
        :param text: 需要审核的文本
        :return: 审核结果字典
        """
        try:
            # 构建审核提示词
            prompt = f"""请分析以下文本内容，判断其是否包含任何不当内容。仅以JSON格式返回分析结果，格式如下：
{{
  "is_safe": true/false,
  "reason": "简短的原因说明",
  "categories": ["可能的问题类别，如仇恨言论、暴力、色情等"]
}}

需要审核的文本：
{text}

分析结果："""
            
            # 准备请求数据
            data = {
                "model": self.moderation_model,
                "prompt": prompt,
                "stream": False,  # 使用非流式响应以获取完整结果
                "options": {
                    "temperature": 0.1,  # 降低温度以获得更一致的结果
                    "top_p": 0.9,
                    "num_predict": 200  # 限制响应长度
                },
                "format": "json"  # 强制返回JSON格式
            }
            
            # 发送请求到 Ollama API
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.ollama_api_url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                moderation_result = result.get("response", "")
                
                try:
                    # 解析AI返回的JSON结果
                    parsed_result = json.loads(moderation_result)
                    return parsed_result
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse JSON from moderation result: {moderation_result}")
                    # 如果无法解析JSON，保守地认为内容不安全
                    return {
                        "is_safe": False,
                        "reason": "Could not parse moderation response",
                        "categories": ["parsing_error"]
                    }
            else:
                logging.error(f"Ollama moderation API error: {response.status_code} - {response.text}")
                return {
                    "is_safe": False,
                    "reason": f"Moderation API error: {response.status_code}",
                    "categories": ["api_error"]
                }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Ollama API for moderation: {e}")
            return {
                "is_safe": False,
                "reason": f"Request error: {str(e)}",
                "categories": ["request_error"]
            }
        except Exception as e:
            logging.error(f"Error in Ollama moderation: {e}")
            return {
                "is_safe": False,
                "reason": f"Moderation error: {str(e)}",
                "categories": ["moderation_error"]
            }

    def moderate_text_for_response(self, text: str) -> bool:
        """
        审核AI生成的文本响应
        :param text: AI生成的文本
        :return: True表示响应安全，可以发送，False表示不安全，需要过滤
        """
        try:
            # 检查是否包含敏感词
            sensitive_keywords = self.config.get('sensitive_keywords', [])
            text_lower = text.lower()
            
            for keyword in sensitive_keywords:
                if keyword.lower() in text_lower:
                    logging.info(f"Sensitive keyword detected: {keyword}")
                    return False
            
            # 使用AI模型进行内容审核
            ollama_result = self.moderate_text_with_ollama(text)
            
            # 同时使用原始审核方法
            original_result = self.moderate_text(text)
            
            # 如果任一审核认为内容不安全，则返回 False（更保守的策略）
            ollama_safe = ollama_result.get('is_safe', False)
            
            if not ollama_safe or not original_result:
                logging.info(f"Content flagged by AI moderation: {ollama_result.get('reason', 'Unknown reason')}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error moderating response text: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        self.task_queue.put(None)  # 发送结束信号
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)  # 最多等待5秒

class ImageModeration:
    """
    图像审核模块 - 专门用于审核通过AI绘画生成的图像
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.content_moderator = ContentModeration(config)
        
    def moderate_generated_image(self, image_path: str) -> bool:
        """
        审核AI生成的图像
        :param image_path: 图像路径
        :return: True表示图像安全，False表示图像不安全
        """
        try:
            # 读取并编码图像
            with open(image_path, "rb") as image_file:
                import base64
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 使用Ollama的多模态模型进行图像审核（如果配置了视觉模型）
            vision_model = self.config.get('ollama_vision_model', 'llava')
            ollama_api_url = self.config.get('ollama_api_url', 'http://localhost:11434/api/generate')
            
            # 构建审核提示词
            prompt = f"""请分析以下图像内容，判断其是否包含任何不当内容，如裸露、暴力、仇恨符号等。
            仅以JSON格式返回分析结果，格式如下：{{"is_safe": true/false, "reason": "简短的原因说明"}}"""
            
            # 准备请求数据
            data = {
                "model": vision_model,
                "prompt": prompt,
                "images": [image_data],  # 提供base64编码的图像
                "stream": False,  # 使用非流式响应
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 150
                },
                "format": "json"  # 强制返回JSON格式
            }
            
            # 发送请求到 Ollama API
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(ollama_api_url, json=data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                moderation_result = result.get("response", "")
                
                try:
                    # 解析AI返回的JSON结果
                    parsed_result = json.loads(moderation_result)
                    is_safe = parsed_result.get('is_safe', False)
                    
                    if is_safe:
                        logging.info(f"Image moderation for {image_path}: PASSED")
                        return True
                    else:
                        reason = parsed_result.get('reason', 'Unknown reason')
                        logging.info(f"Image moderation for {image_path}: FAILED - {reason}")
                        return False
                        
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse JSON from image moderation result: {moderation_result}")
                    # 如果无法解析JSON，保守地认为图像不安全
                    return False
            else:
                logging.error(f"Ollama image moderation API error: {response.status_code} - {response.text}")
                # API错误时保守地认为图像不安全
                return False
            
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            return False
        except Exception as e:
            logging.error(f"Error moderating image {image_path}: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        self.content_moderator.cleanup()