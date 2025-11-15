import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import hashlib
import requests

class LongTermMemory:
    """
    长期记忆存储系统
    实现三级记忆存储（永久、一周、三个月）
    两种记忆类型（摘要型、工具调用型）
    """
    
    def __init__(self, config: Dict[str, Any], vdb_manager):
        self.config = config
        self.vdb_manager = vdb_manager
        
        # 初始化上下文摘要器
        summarizer_api_url = self.config.get('small_model_api_url', 'http://localhost:11434/api/generate')
        summarizer_model = self.config.get('summarizer_model', 'gemma2:2b')
        self.context_summarizer = ContextSummarizer(summarizer_api_url, summarizer_model)
        
        logging.info("Long term memory system initialized")

    def add_summary_memory(self, context: str, summary: str, detail_level: str = 'medium') -> bool:
        """
        添加摘要型记忆（根据详细程度分层）
        :param context: 上下文
        :param summary: 摘要
        :param detail_level: 详细程度 ('low', 'medium', 'high')
        :return: 成功与否
        """
        try:
            # 根据详细程度决定存储到哪个层级
            memory_type = self._get_memory_type_by_detail_level(detail_level)
            
            metadata = {
                'type': 'summary',
                'detail_level': detail_level,
                'context': context,
                'summary': summary
            }
            
            content = f"Context: {context}\nSummary: {summary}\nDetail Level: {detail_level}"
            
            success = self.vdb_manager.add_memory(content, memory_type, metadata)
            
            if success:
                logging.info(f"Added summary memory (level: {detail_level}) to {memory_type}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error adding summary memory: {e}")
            return False

    def add_tool_call_memory(self, tool_name: str, arguments: Dict[str, Any], result: str) -> bool:
        """
        添加工具调用型记忆（永久保存）
        :param tool_name: 工具名称
        :param arguments: 工具参数
        :param result: 工具执行结果
        :return: 成功与否
        """
        try:
            # 工具调用型记忆总是存储在永久记忆中
            memory_type = 'permanent'
            
            metadata = {
                'type': 'tool_call',
                'tool_name': tool_name,
                'arguments': arguments,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            content = f"Tool Call: {tool_name}\nArguments: {json.dumps(arguments)}\nResult: {result}"
            
            success = self.vdb_manager.add_memory(content, memory_type, metadata)
            
            if success:
                logging.info(f"Added tool call memory: {tool_name}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error adding tool call memory: {e}")
            return False

    def _get_memory_type_by_detail_level(self, detail_level: str) -> str:
        """
        根据详细程度确定记忆类型
        :param detail_level: 详细程度
        :return: 记忆类型
        """
        level_to_type = {
            'low': 'monthly',    # 低详细程度存3个月
            'medium': 'weekly',  # 中等详细程度存一周
            'high': 'permanent'  # 高详细程度永久保存
        }
        return level_to_type.get(detail_level, 'weekly')

    def search_contextual_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索上下文相关记忆
        :param query: 查询
        :param top_k: 返回结果数量
        :return: 搜索结果列表
        """
        try:
            results = self.vdb_manager.search_memory(query, top_k)
            logging.info(f"Found {len(results)} contextual memories for query: {query[:30]}...")
            return results
            
        except Exception as e:
            logging.error(f"Error searching contextual memory: {e}")
            return []

    def update_memo_in_system_prompt(self, system_prompt: str, new_memo: str) -> str:
        """
        更新系统提示词中的备忘录
        :param system_prompt: 原始系统提示词
        :param new_memo: 新备忘录
        :return: 更新后的系统提示词
        """
        return self.vdb_manager.update_system_prompt_with_memo(system_prompt, new_memo)

class ContextSummarizer:
    """
    上下文摘要器 - 使用0.5b模型对对话上下文进行摘要
    """
    
    def __init__(self, small_model_api_url: str = "http://localhost:11434/api/generate", model_name: str = "gemma2:2b"):
        self.small_model_api_url = small_model_api_url
        self.model_name = model_name
        logging.info("Context summarizer initialized")

    def summarize_context(self, context: str, detail_level: str = 'medium') -> str:
        """
        摘要上下文
        :param context: 需要摘要的上下文
        :param detail_level: 摘要详细程度
        :return: 摘要文本
        """
        try:
            # 根据详细程度调整摘要长度和提示词
            length_instruction = {
                'low': "用一两句话简单概括",
                'medium': "用简洁的几段话概括要点", 
                'high': "详细概括，保留重要细节"
            }.get(detail_level, "用简洁的几段话概括要点")
            
            # 构建摘要提示词
            prompt = f"""请根据以下上下文生成摘要：

上下文内容：
{context}

要求：
{length_instruction}

摘要："""
            
            # 根据详细程度设置模型参数
            temperature = {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7
            }.get(detail_level, 0.5)
            
            # 准备请求数据
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # 使用非流式响应
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": {
                        'low': 100,
                        'medium': 250,
                        'high': 500
                    }.get(detail_level, 250)
                }
            }
            
            # 发送请求到 Ollama API
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.small_model_api_url, json=data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "")
                
                logging.info(f"Context summarized (level: {detail_level}, length: {len(summary)})")
                return summary
            else:
                logging.error(f"Ollama API error: {response.status_code} - {response.text}")
                # 出错时返回原内容的简短版本
                max_length = {'low': 100, 'medium': 250, 'high': 500}.get(detail_level, 250)
                return context[:max_length] + "..." if len(context) > max_length else context
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Ollama API for summarization: {e}")
            # 出错时返回原内容的简短版本
            max_length = {'low': 100, 'medium': 250, 'high': 500}.get(detail_level, 250)
            return context[:max_length] + "..." if len(context) > max_length else context
        except Exception as e:
            logging.error(f"Error summarizing context: {e}")
            # 出错时返回原内容的简短版本
            max_length = {'low': 100, 'medium': 250, 'high': 500}.get(detail_level, 250)
            return context[:max_length] + "..." if len(context) > max_length else context