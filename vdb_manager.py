import logging
import os
from typing import List, Dict, Any, Optional
import requests
import json
from datetime import datetime, timedelta
import threading
import time
import hashlib

# 导入Qdrant客户端
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import PointStruct, Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    logging.warning("Qdrant client not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False

class VDBManager:
    """
    使用Qdrant向量数据库的管理模块
    负责：
    - 管理长期记忆存储
    - 提供RAG功能
    - 处理TTL过期
    - 弹幕内容筛选
    - 自动上下文摘要生成
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_api_url = config.get('ollama_api_url', 'http://localhost:11434')
        self.qdrant_client = None
        self.collection_name = "memory_collection"
        self.summarizer_model = self.config.get('summarizer_model', 'gemma2:2b')
        
        # 性能相关配置 - 从配置文件加载，提供默认值
        self.vdb_auto_run = self.config.get('vdb_auto_run', True)  # 是否自动运行数据库清理
        self.vdb_startup_run = self.config.get('vdb_startup_run', True)  # 是否在启动时运行清理
        self.vdb_check_interval = self.config.get('vdb_check_interval', 60)  # 检查间隔（分钟）

        # 初始化Qdrant客户端
        self._init_qdrant_client()

        # 如果启用了自动运行，则启动过期清理线程
        if self.vdb_auto_run:
            self.stop_expiry_event = threading.Event()
            self.expiry_thread = threading.Thread(target=self._expiry_worker, daemon=True)
            self.expiry_thread.start()
            logging.info("启动了自动数据库清理线程")
        
        # 如果设置了启动时运行，则立即执行一次清理
        if self.vdb_startup_run:
            threading.Thread(target=self._clean_expired_memories, daemon=True).start()
            logging.info("启动时执行数据库清理")

        # 如果启用了自动运行，则启动上下文摘要生成线程
        if self.vdb_auto_run:
            self.summary_thread = threading.Thread(target=self._context_summary_worker, daemon=True)
            self.summary_thread.start()
            logging.info("启动了自动上下文摘要线程")
        else:
            logging.info("上下文摘要功能已禁用")

        logging.info("VDB Manager with Qdrant initialized successfully")

    def _init_qdrant_client(self):
        """初始化Qdrant客户端"""
        try:
            if not QDRANT_AVAILABLE:
                logging.error("Qdrant client not available. Please install with: pip install qdrant-client")
                return

            # 从配置获取Qdrant连接信息
            qdrant_host = self.config.get('qdrant_host', 'localhost')
            qdrant_port = self.config.get('qdrant_port', 6333)

            # 创建Qdrant客户端
            self.qdrant_client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                timeout=30
            )

            # 检查集合是否存在，不存在则创建
            try:
                self.qdrant_client.get_collection(self.collection_name)
                logging.info(f"Connected to existing Qdrant collection: {self.collection_name}")
            except:
                # 创建新的向量集合 - 使用较小的向量维度以兼容更多模型
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # 使用384维向量
                )
                logging.info(f"Created new Qdrant collection: {self.collection_name}")

        except Exception as e:
            logging.error(f"Error initializing Qdrant client: {e}")
            logging.info("Falling back to simulated embedding approach")

    def _generate_embedding_with_ollama(self, text: str) -> Optional[List[float]]:
        """
        使用 Ollama API 生成文本嵌入
        尝试使用嵌入API，如果不可用则使用变通方法
        """
        if not self.qdrant_client:
            # 如果Qdrant不可用，使用原来的方法
            return self._generate_embedding_fallback(text)

        try:
            # 尝试使用Ollama嵌入API
            # 注意：Ollama的一些版本可能需要使用专门的嵌入模型
            embedding_model = self.config.get('embedding_model', 'nomic-embed-text')

            # 尝试调用Ollama嵌入API
            response = requests.post(
                f"{self.ollama_api_url}/api/embeddings",
                json={
                    "model": embedding_model,
                    "prompt": text
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                if embedding:
                    # 归一化向量到384维（Qdrant集合的维度）
                    if len(embedding) > 384:
                        embedding = embedding[:384]
                    elif len(embedding) < 384:
                        # 用零填充
                        embedding.extend([0.0] * (384 - len(embedding)))
                    return embedding
                else:
                    logging.warning("Ollama embeddings API returned empty embedding")

        except requests.exceptions.RequestException as e:
            logging.warning(f"Ollama embeddings API not available: {e}")
        except Exception as e:
            logging.warning(f"Error calling Ollama embeddings API: {e}")

        # 如果Ollama嵌入API不可用，使用变通方法
        return self._generate_embedding_fallback(text)

    def _generate_embedding_fallback(self, text: str) -> Optional[List[float]]:
        """
        使用变通方法生成嵌入向量（使用Ollama的chat API）
        """
        try:
            # 使用Ollama API获取文本的语义表示
            prompt = f"请将以下文本总结为一个简短的关键词或短语，用于向量检索：{text}"

            data = {
                "model": self.config.get('ollama_model', 'llama3.2'),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 50
                }
            }

            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(f"{self.ollama_api_url}/api/chat", json=data, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                summary = result.get("message", {}).get("content", text)

                # 将摘要转换为一个相对有意义的向量表示（使用哈希方法但结合语义）
                import hashlib
                hash_obj = hashlib.sha256(summary.encode('utf-8'))
                hash_hex = hash_obj.hexdigest()

                # 将十六进制哈希转换为浮点数列表
                embedding = []
                for i in range(0, len(hash_hex), 2):
                    hex_pair = hash_hex[i:i+2]
                    val = int(hex_pair, 16) / 255.0  # 归一化到 [0,1]
                    embedding.append(val)

                # 确保向量长度为384（Qdrant集合的维度）
                if len(embedding) > 384:
                    embedding = embedding[:384]
                elif len(embedding) < 384:
                    embedding.extend([0.0] * (384 - len(embedding)))

                return embedding
            else:
                logging.warning(f"Ollama API call for embedding fallback failed: {response.status_code}")
                # 退回到简单的哈希方法
                return self._generate_simple_hash_embedding(text)

        except Exception as e:
            logging.warning(f"Error in embedding fallback: {e}")
            # 最后的备选：简单哈希嵌入
            return self._generate_simple_hash_embedding(text)

    def _generate_simple_hash_embedding(self, text: str) -> Optional[List[float]]:
        """
        简单哈希嵌入方法
        """
        try:
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_hex = hash_obj.hexdigest()

            # 将十六进制哈希转换为浮点数列表
            embedding = []
            for i in range(0, len(hash_hex), 2):
                hex_pair = hash_hex[i:i+2]
                val = int(hex_pair, 16) / 255.0  # 归一化到 [0,1]
                embedding.append(val)

            # 确保向量长度为384（Qdrant集合的维度）
            if len(embedding) > 384:
                embedding = embedding[:384]
            elif len(embedding) < 384:
                embedding.extend([0.0] * (384 - len(embedding)))

            return embedding

        except Exception as e:
            logging.error(f"Error generating simple hash embedding: {e}")
            return None

    def add_memory(self, content: str, memory_type: str = 'weekly', metadata: Optional[Dict] = None):
        """
        添加记忆到Qdrant向量数据库
        :param content: 内容文本
        :param memory_type: 记忆类型 ('permanent', 'weekly', 'monthly')
        :param metadata: 额外元数据
        """
        try:
            if not self.qdrant_client:
                logging.error("Qdrant client not available")
                return False

            # 准备元数据
            timestamp = datetime.now().isoformat()
            ttl = self._get_ttl(memory_type)
            expiry_time = (datetime.now() + ttl).isoformat()

            metadata = metadata or {}
            metadata.update({
                'timestamp': timestamp,
                'expiry_time': expiry_time,
                'memory_type': memory_type,
                'content': content
            })

            # 生成唯一ID
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            # 生成嵌入向量
            embedding = self._generate_embedding_with_ollama(content)
            if embedding is None:
                logging.error("Failed to generate embedding for content")
                return False

            # 添加到Qdrant
            points = [PointStruct(
                id=content_hash,
                vector=embedding,
                payload=metadata
            )]

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logging.info(f"Added memory to Qdrant: {content[:50]}... (type: {memory_type})")
            return True

        except Exception as e:
            logging.error(f"Error adding memory to Qdrant: {e}")
            return False

    def search_memory(self, query: str, top_k: int = 5, memory_types: List[str] = None) -> List[Dict]:
        """
        从Qdrant搜索记忆
        :param query: 查询文本
        :param top_k: 返回结果数量
        :param memory_types: 记忆类型列表
        :return: 搜索结果列表
        """
        try:
            if not self.qdrant_client:
                logging.error("Qdrant client not available")
                return []

            # 生成查询嵌入向量
            query_embedding = self._generate_embedding_with_ollama(query)
            if query_embedding is None:
                logging.error("Failed to generate embedding for query")
                return []

            # 准备过滤条件
            filters = None
            if memory_types:
                filters = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="memory_type",
                            match=models.MatchAny(any=memory_types)
                        )
                    ]
                )

            # 执行向量搜索
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=filters,
                with_payload=True
            )

            results = []
            current_time = datetime.now()

            for hit in search_results:
                # 检查是否过期
                expiry_time_str = hit.payload.get('expiry_time', '')
                if expiry_time_str:
                    try:
                        expiry_time = datetime.fromisoformat(expiry_time_str.replace('Z', '+00:00'))
                        if expiry_time < current_time:
                            continue  # 跳过过期的记忆
                    except:
                        continue  # 时间格式错误，跳过

                results.append({
                    'content': hit.payload.get('content', ''),
                    'score': hit.score,
                    'metadata': hit.payload
                })

            return results

        except Exception as e:
            logging.error(f"Error searching memory in Qdrant: {e}")
            return []

    def filter_danmu(self, danmu_content: str) -> bool:
        """
        使用Ollama API过滤弹幕内容
        :param danmu_content: 弹幕内容
        :return: True表示通过过滤，False表示被过滤掉
        """
        try:
            # 使用Ollama API进行内容审核
            prompt = f"""请判断以下弹幕内容是否适当。仅回答"适当"或"不当"。
弹幕内容：{danmu_content}

判断结果："""

            data = {
                "model": self.config.get('ollama_model', 'llama3.2'),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 20
                }
            }

            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(f"{self.ollama_api_url}/api/chat", json=data, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("message", {}).get("content", "").strip().lower()

                # 如果AI认为内容不当，则过滤掉
                if "不当" in ai_response or "not appropriate" in ai_response or "inappropriate" in ai_response:
                    logging.info(f"Filtered inappropriate danmu (AI decision): {danmu_content}")
                    return False
                else:
                    return True
            else:
                logging.warning(f"Ollama API call failed for danmu filtering: {response.status_code}")
                # API调用失败时默认通过过滤，避免误杀
                return True

        except Exception as e:
            logging.error(f"Error filtering danmu: {e}")
            # 出错时默认通过过滤，避免误杀
            return True

    def _get_ttl(self, memory_type: str) -> timedelta:
        """获取记忆类型的TTL"""
        ttl_map = {
            'permanent': timedelta(days=365*100),  # 实际上永久
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=90)  # 3个月
        }
        return ttl_map.get(memory_type, timedelta(weeks=1))

    def _context_summary_worker(self):
        """上下文摘要生成工作线程"""
        # 启动时立即执行一次摘要生成（仅当启用自动运行时）
        if self.vdb_auto_run:
            self._generate_context_summary()

        while True:
            try:
                # 从配置中获取摘要间隔，如果没有配置则使用24小时（分钟转秒）
                summary_interval = self.config.get('context_summary_interval', 1440)  # 默认24小时(1440分钟)
                sleep_seconds = summary_interval * 60
                
                logging.info(f"上下文摘要生成完成，将在 {summary_interval} 分钟后再次执行")
                time.sleep(sleep_seconds)
                
                # 检查是否仍启用自动运行
                if self.vdb_auto_run:
                    self._generate_context_summary()
                else:
                    logging.info("上下文摘要功能已禁用，跳过执行")
                    # 即使禁用了，也继续循环，以便将来重新启用时能正常工作

            except Exception as e:
                logging.error(f"Error in context summary worker: {e}")
                # 出错时等待1小时后重试，避免频繁失败
                time.sleep(3600)

    def _generate_context_summary(self):
        """使用副模型对上下文进行摘要，并进行长期记忆管理"""
        try:
            if not self.qdrant_client:
                logging.warning("Qdrant client not available for context summarization")
                return

            # 优化：获取最近指定时间范围内的记忆，用于生成摘要
            # 从配置中获取时间范围，如果没有配置则使用24小时
            time_range_hours = self.config.get('context_summary_time_range', 24)
            current_time = datetime.now()
            since_time = current_time - timedelta(hours=time_range_hours)
            
            # 优化：限制获取的数量，避免内存占用过大
            limit_count = min(self.config.get('context_summary_limit', 500), 1000)  # 最多1000条

            # 获取最近的记忆
            recent_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit_count,
                with_payload=True
            )[0]

            # 优化：过滤出最近的记忆，并使用列表推导式提高效率
            recent_memories = []
            for point in recent_points:
                timestamp_str = point.payload.get('timestamp', '')
                if timestamp_str:
                    try:
                        # 优化：使用更高效的时间解析方式
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp >= since_time:
                            # 只存储必要的字段，减少内存使用
                            recent_memories.append({
                                'id': point.id,
                                'content': point.payload.get('content', ''),
                                'timestamp': timestamp,
                                'metadata': point.payload
                            })
                    except Exception as e:
                        logging.debug(f"时间格式解析错误，跳过该记录: {e}")
                        continue
            
            # 优化：按时间戳排序，优先处理最新的记忆
            recent_memories.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # 进一步限制内存使用，只保留最新的N条记忆
            max_recent_memories = min(self.config.get('max_recent_memories_for_summary', 300), 500)
            recent_memories = recent_memories[:max_recent_memories]

            # 如果没有最近的记忆，跳过摘要
            if not recent_memories:
                logging.info("No recent memories found, skipping context summary generation")
                return

            # 让副模型对近期记忆进行摘要
            summary_content = self._let_llm_decide_summary(recent_memories)

            if summary_content and summary_content.strip():
                # 创建摘要记忆
                self._create_summary_memory(summary_content, "auto", {})
                logging.info(f"Generated context summary with {len(recent_memories)} recent memories")
            else:
                logging.info(f"LLM decided no summary needed for {len(recent_memories)} recent memories")

            # 现在启动长期记忆管理循环
            self._run_long_term_memory_management()

        except Exception as e:
            logging.error(f"Error generating context summary: {e}")

    def _run_long_term_memory_management(self):
        """运行长期记忆管理循环，让副模型管理长期记忆"""
        try:
            # 获取所有记忆用于管理
            all_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=2000,  # 限制获取的数量
                with_payload=True
            )[0]

            # 构建记忆列表
            memory_list = []
            for point in all_points:
                memory_list.append({
                    'id': point.id,
                    'content': point.payload.get('content', ''),
                    'timestamp': point.payload.get('timestamp', ''),
                    'type': point.payload.get('memory_type', 'unknown'),
                    'permanent': point.payload.get('permanent', False),
                    'importance': point.payload.get('importance_level', 'medium')
                })

            if not memory_list:
                logging.info("No memories to manage, skipping long-term memory management")
                return

            # 构建系统提示词
            system_prompt = f"""
你是一个智能长期记忆管理器。你的任务是管理和优化长期记忆存储，使其更精炼、更有价值。你可以执行以下操作：
1. 删除 - 删除不再重要或过时的记忆
2. 整合 - 将多个相关记忆整合为一个综合记忆
3. 修改 - 更新记忆的内容，使其更准确或详细
4. 重新分类 - 更改记忆的类型或重要性等级

可用工具：
- delete_memory(id): 删除指定ID的记忆
- merge_memories(ids, new_content): 合并多个ID的记忆为一个新的记忆
- modify_memory(id, new_content): 修改指定ID的记忆内容
- update_metadata(id, metadata_updates): 更新记忆的元数据（类型、重要性等）
- stop(): 停止管理循环

请分析以下记忆列表，并根据重要性和相关性进行管理。重点关注：
- 重复或相似的内容可以整合
- 过时或不重要的记忆可以删除
- 有价值的细节可以保留和强化
- 持久性记忆(persistent)不应被删除，除非明确是错误的

记住：你可以进行多次操作，直到达到最佳的记忆组织状态。当完成管理时，请调用stop()指令。
"""

            # 开始管理循环
            continue_loop = True
            iteration_count = 0
            max_iterations = 10  # 防止无限循环

            while continue_loop and iteration_count < max_iterations:
                iteration_count += 1

                # 让副模型决定要执行的操作
                memory_info = "\n".join([
                    f"ID: {mem['id'][:8]}... | Type: {mem['type']} | Permanent: {mem['permanent']} | Importance: {mem['importance']} | Content: {mem['content'][:100]}..."
                    for mem in memory_list[:20]  # 只取前20个展示
                ])

                prompt = f"""
{system_prompt}

当前记忆列表：
{memory_info}

请告诉我你要执行什么操作：
"""

                try:
                    data = {
                        "model": self.summarizer_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": 0.4,
                            "top_p": 0.9,
                            "num_predict": 500
                        }
                    }

                    headers = {
                        "Content-Type": "application/json"
                    }

                    response = requests.post(f"{self.ollama_api_url}/api/chat", json=data, headers=headers, timeout=60)

                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result.get("message", {}).get("content", "")

                        # 解析AI的响应以决定执行什么操作
                        action_taken = self._execute_memory_management_action(ai_response, memory_list)

                        if action_taken == "stop":
                            continue_loop = False
                            logging.info("Long-term memory management completed by LLM")
                        elif not action_taken:
                            # 如果没有执行任何操作或操作不明确，也停止以避免无限循环
                            logging.info("No valid action taken, stopping long-term memory management")
                            continue_loop = False

                        # 更新记忆列表（重新获取）
                        all_points = self.qdrant_client.scroll(
                            collection_name=self.collection_name,
                            limit=2000,
                            with_payload=True
                        )[0]

                        memory_list = []
                        for point in all_points:
                            memory_list.append({
                                'id': point.id,
                                'content': point.payload.get('content', ''),
                                'timestamp': point.payload.get('timestamp', ''),
                                'type': point.payload.get('memory_type', 'unknown'),
                                'permanent': point.payload.get('permanent', False),
                                'importance': point.payload.get('importance_level', 'medium')
                            })

                    else:
                        logging.error(f"Error calling summarizer for memory management: {response.status_code}")
                        continue_loop = False

                except Exception as e:
                    logging.error(f"Error in long-term memory management loop: {e}")
                    continue_loop = False

        except Exception as e:
            logging.error(f"Error in long-term memory management: {e}")

    def _execute_memory_management_action(self, ai_response: str, memory_list: List[Dict]):
        """根据AI响应执行记忆管理动作"""
        # 参数验证
        if not ai_response or not isinstance(ai_response, str):
            logging.warning("Invalid AI response provided to _execute_memory_management_action")
            return False
        if memory_list and not isinstance(memory_list, list):
            logging.warning("Invalid memory_list provided to _execute_memory_management_action")
            return False
            
        try:
            # 检查是否要停止
            if "stop()" in ai_response or "停止" in ai_response or "完成" in ai_response:
                return "stop"

            # 查找要删除的记忆ID
            import re
            # 查找delete操作
            delete_pattern = r'delete_memory\(["\']([^"\']+)["\']\)'
            delete_matches = re.findall(delete_pattern, ai_response)

            for memory_id in delete_matches:
                if self._delete_memory_by_id(memory_id):
                    logging.info(f"Deleted memory with ID: {memory_id[:8]}...")

            # 查找修改操作
            modify_pattern = r'modify_memory\(["\']([^"\']+)["\'],\s*["\'](.+?)["\']\)'
            modify_matches = re.findall(modify_pattern, ai_response, re.DOTALL)

            for memory_id, new_content in modify_matches:
                if self._modify_memory_by_id(memory_id, new_content):
                    logging.info(f"Modified memory with ID: {memory_id[:8]}...")

            # 查找合并操作
            merge_pattern = r'merge_memories\(\[(.*?)\],\s*["\'](.+?)["\']\)'
            merge_matches = re.findall(merge_pattern, ai_response, re.DOTALL)

            for ids_str, new_content in merge_matches:
                # 解析ID列表
                try:
                    ids = [id.strip().strip('"\'') for id in ids_str.split(',') if id.strip().strip('"\'')]
                    if not ids:
                        logging.warning("No valid memory IDs found in merge operation")
                        continue
                        
                    # 删除旧记忆并创建新合并的记忆
                    deleted_ids = []
                    for old_id in ids:
                        if self._delete_memory_by_id(old_id):
                            deleted_ids.append(old_id)
                    
                    if deleted_ids:
                        # 创建新的合并记忆
                        self._create_summary_memory(new_content, "merged", {"source_ids": deleted_ids})
                        logging.info(f"Merged memories: {[id[:8]+'...' for id in deleted_ids]}")
                except Exception as merge_error:
                    logging.error(f"Error processing merge operation: {merge_error}", exc_info=True)

            # 查找更新元数据操作
            metadata_pattern = r'update_metadata\(["\']([^"\']+)["\'],\s*(\{.*?\})\)'
            metadata_matches = re.findall(metadata_pattern, ai_response, re.DOTALL)

            for memory_id, metadata_json_str in metadata_matches:
                import json
                try:
                    metadata_updates = json.loads(metadata_json_str)
                    if not isinstance(metadata_updates, dict):
                        logging.warning(f"Invalid metadata format for memory {memory_id[:8]}...")
                        continue
                        
                    if self._update_memory_metadata(memory_id, metadata_updates):
                        logging.info(f"Updated metadata for memory with ID: {memory_id[:8]}...")
                except json.JSONDecodeError as json_error:
                    logging.error(f"JSON decode error for metadata: {json_error}")
                except Exception as metadata_error:
                    logging.error(f"Error processing metadata update: {metadata_error}", exc_info=True)

            return len(delete_matches) + len(modify_matches) + len(merge_matches) + len(metadata_matches) > 0

        except Exception as e:
            logging.error(f"Error executing memory management action: {e}", exc_info=True)
            return False

    def _delete_memory_by_id(self, memory_id: str):
        """通过ID删除记忆"""
        # 参数验证
        if not memory_id or not isinstance(memory_id, str):
            logging.warning("Invalid memory ID provided to _delete_memory_by_id")
            return False
            
        try:
            if self.qdrant_client:
                # 验证ID格式或存在性
                existing_points = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[memory_id],
                    with_payload=False
                )
                
                if not existing_points:
                    logging.warning(f"Memory with ID {memory_id[:8]}... does not exist")
                    return False
                
                # 执行删除操作
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=[memory_id])
                )
                return True
            else:
                logging.warning("Qdrant client not initialized when trying to delete memory")
                return False
        except Exception as e:
            logging.error(f"Error deleting memory {memory_id[:8]}...: {e}", exc_info=True)
            return False

    def _modify_memory_by_id(self, memory_id: str, new_content: str):
        """通过ID修改记忆内容"""
        # 参数验证
        if not memory_id or not isinstance(memory_id, str):
            logging.warning("Invalid memory ID provided to _modify_memory_by_id")
            return False
        if not new_content or not isinstance(new_content, str):
            logging.warning("Invalid new content provided to _modify_memory_by_id")
            return False
            
        try:
            if self.qdrant_client:
                # 首先获取原有点的信息
                points = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[memory_id],
                    with_payload=True,
                    with_vectors=True
                )

                if not points:
                    logging.warning(f"Memory with ID {memory_id[:8]}... not found for modification")
                    return False
                    
                point = points[0]
                # 更新内容和时间戳
                point.payload['content'] = new_content
                point.payload['modified_at'] = datetime.now().isoformat()

                # 更新这个点
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=point.payload
                    )]
                )
                return True
            else:
                logging.warning("Qdrant client not initialized when trying to modify memory")
                return False
        except Exception as e:
            logging.error(f"Error modifying memory {memory_id[:8]}...: {e}", exc_info=True)
            return False

    def _update_memory_metadata(self, memory_id: str, metadata_updates: Dict):
        """更新记忆的元数据"""
        # 参数验证
        if not memory_id or not isinstance(memory_id, str):
            logging.warning("Invalid memory ID provided to _update_memory_metadata")
            return False
        if not metadata_updates or not isinstance(metadata_updates, dict):
            logging.warning("Invalid metadata updates provided to _update_memory_metadata")
            return False
            
        try:
            if self.qdrant_client:
                # 首先获取原有点的信息
                points = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[memory_id],
                    with_payload=True,
                    with_vectors=True
                )

                if not points:
                    logging.warning(f"Memory with ID {memory_id[:8]}... not found for metadata update")
                    return False
                    
                point = points[0]
                # 过滤掉可能导致问题的键值对
                safe_metadata = {k: v for k, v in metadata_updates.items() if k not in ['content', 'created_at', 'id']}
                point.payload.update(safe_metadata)
                point.payload['modified_at'] = datetime.now().isoformat()

                # 更新这个点
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=point.payload
                    )]
                )
                return True
            else:
                logging.warning("Qdrant client not initialized when trying to update memory metadata")
                return False
        except Exception as e:
            logging.error(f"Error updating metadata for memory {memory_id[:8]}...: {e}", exc_info=True)
            return False

    def _let_llm_decide_summary(self, recent_memories: List[Dict]) -> str:
        """
        让副LLM自行决定重要性并生成摘要
        :param recent_memories: 近期记忆列表
        :return: 摘要内容或空字符串（如果LLM认为不需要摘要）
        """
        try:
            # 构建输入给副LLM的内容
            memory_texts = []
            for i, memory in enumerate(recent_memories):
                memory_texts.append(f"{i+1}. {memory['content']}")

            # 构建提示词
            prompt = f"""
            请分析以下近期对话内容，决定是否需要生成摘要：

            近期内容：
            {''.join(memory_texts)}

            请做出以下判断：
            1. 这些内容中有哪些是真正重要的信息？
            2. 是否有必要生成摘要？（如果内容平淡无奇或重复， says no）
            3. 如果需要摘要，请生成简洁但保留关键信息的摘要。

            请按以下格式回复：
            是否需要摘要：[是/否]
            重要信息：[重要的要点，如果没有则写"无"]
            摘要：[如果需要则生成摘要，如果不需要则写"无"]
            """

            data = {
                "model": self.summarizer_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }

            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(f"{self.ollama_api_url}/api/chat", json=data, headers=headers, timeout=60)

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("message", {}).get("content", "")

                # 解析副LLM的响应
                need_summary = "是" in ai_response or "yes" in ai_response.lower()

                if need_summary:
                    # 尝试从响应中提取摘要部分
                    lines = ai_response.split('\n')
                    for line in lines:
                        if line.startswith('摘要：') or line.startswith('摘要:'):
                            summary = line[3:].strip()  # 去掉"摘要："前缀
                            if summary != "无" and summary.strip():
                                return summary

                return ""  # 不需要摘要或无法解析摘要

            else:
                logging.error(f"Error calling summarizer model: {response.status_code}")
                return ""

        except Exception as e:
            logging.error(f"Error in _let_llm_decide_summary: {e}")
            return ""

    def _calculate_importance_score(self, content: str) -> float:
        """计算记忆的重要性分数 (0-1) - 现在主要用于备用"""
        try:
            # 基于内容长度、关键词等因素计算重要性（保留作为备用方案）
            score = 0.1  # 基础分数

            # 基于关键词增加重要性
            important_keywords = ['重要', '关键', '紧急', '注意', '必须', '记得', '特殊', '特别', '重要信息']
            for keyword in important_keywords:
                if keyword in content:
                    score += 0.3

            # 基于内容长度调整
            if len(content) > 100:
                score += 0.2
            elif len(content) < 10:
                score -= 0.1
            else:
                score += 0.1

            # 确保分数在0-1之间
            return max(0.0, min(1.0, score))

        except:
            return 0.1  # 出错时返回基础分数

    def _create_summary_memory(self, content: str, importance_level: str, original_metadata: Dict):
        """使用副模型创建摘要记忆"""
        try:
            # 让副模型决定这个摘要是否值得长期保存
            decision_prompt = f"""
            请评估以下内容是否重要，值得长期保存：

            原始内容：{content}

            请回答：是否值得长期保存？[是/否]
            理由：[简要说明理由]
            """

            data = {
                "model": self.summarizer_model,
                "messages": [{"role": "user", "content": decision_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 100
                }
            }

            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(f"{self.ollama_api_url}/api/chat", json=data, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                decision_response = result.get("message", {}).get("content", "")

                # 解析副模型的决策
                should_save = "是" in decision_response or "需要" in decision_response or "值得" in decision_response

                if not should_save:
                    logging.info(f"LLM decided not to save summary (level: {importance_level}): content not deemed important")
                    return  # 不保存摘要

                # 如果决定保存，生成摘要
                prompt = f"""
                请对以下内容进行简洁但准确的摘要，保留关键信息：

                内容：{content}

                摘要：
                """

                summarize_data = {
                    "model": self.summarizer_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 200
                    }
                }

                summarize_response = requests.post(f"{self.ollama_api_url}/api/chat", json=summarize_data, headers=headers, timeout=60)

                if summarize_response.status_code == 200:
                    summarize_result = summarize_response.json()
                    summary = summarize_result.get("message", {}).get("content", "").strip()

                    # 保存摘要记忆，标记为永久记忆（不会过期）
                    summary_metadata = {
                        'original_content': content,
                        'summary': summary,
                        'importance_level': importance_level,
                        'created_at': datetime.now().isoformat(),
                        'permanent': True,  # 永不删除的标记
                        'content_type': 'summary'
                    }

                    # 将原始metadata中的关键信息合并
                    summary_metadata.update({k: v for k, v in original_metadata.items()
                                           if k not in ['content', 'expiry_time', 'timestamp']})

                    # 使用哈希生成ID（因为这是摘要，可能不需要单独的embedding）
                    content_hash = hashlib.md5((content + summary).encode('utf-8')).hexdigest()

                    # 生成嵌入向量（对于摘要）
                    embedding = self._generate_embedding_with_ollama(summary or content)
                    if embedding is None:
                        embedding = self._generate_simple_hash_embedding(summary or content)

                    if embedding:
                        # 添加到Qdrant，作为永久记忆
                        points = [PointStruct(
                            id=content_hash,
                            vector=embedding,
                            payload=summary_metadata
                        )]

                        self.qdrant_client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )

                        logging.info(f"Created summary memory (level: {importance_level}): {summary[:100]}...")

        except Exception as e:
            logging.error(f"Error creating summary memory: {e}")

    def add_memory(self, content: str, memory_type: str = 'weekly', metadata: Optional[Dict] = None):
        """
        添加记忆到Qdrant向量数据库
        :param content: 内容文本
        :param memory_type: 记忆类型 ('permanent', 'weekly', 'monthly')
        :param metadata: 额外元数据
        """
        try:
            if not self.qdrant_client:
                logging.error("Qdrant client not available")
                return False

            # 准备元数据
            timestamp = datetime.now().isoformat()
            # 永久记忆不设置过期时间
            expiry_time = None
            if memory_type != 'permanent':
                ttl = self._get_ttl(memory_type)
                expiry_time = (datetime.now() + ttl).isoformat()

            metadata = metadata or {}
            metadata.update({
                'timestamp': timestamp,
                'memory_type': memory_type,
                'content': content
            })

            # 永久记忆标记
            if memory_type == 'permanent':
                metadata['permanent'] = True

            # 只有非永久记忆才设置过期时间
            if expiry_time:
                metadata['expiry_time'] = expiry_time

            # 生成唯一ID
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            # 生成嵌入向量
            embedding = self._generate_embedding_with_ollama(content)
            if embedding is None:
                logging.error("Failed to generate embedding for content")
                return False

            # 添加到Qdrant
            points = [PointStruct(
                id=content_hash,
                vector=embedding,
                payload=metadata
            )]

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logging.info(f"Added memory to Qdrant: {content[:50]}... (type: {memory_type})")
            return True

        except Exception as e:
            logging.error(f"Error adding memory to Qdrant: {e}")
            return False

    def _clean_expired_memories(self):
        """清理过期记忆的实际实现方法"""
        try:
            if not self.qdrant_client:
                return

            current_time = datetime.now()

            # 获取所有记忆（通过scroll获取所有点）
            all_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000  # 假设不超过10000个点
            )[0]

            expired_ids = []

            for point in all_points:
                # 检查是否为永久记忆
                if point.payload.get('permanent', False):
                    continue  # 永久记忆不删除

                # 检查是否有过期时间
                expiry_time_str = point.payload.get('expiry_time', '')
                if expiry_time_str:
                    try:
                        expiry_time = datetime.fromisoformat(expiry_time_str.replace('Z', '+00:00'))
                        if expiry_time < current_time:
                            expired_ids.append(point.id)
                    except:
                        # 时间格式错误，也标记为过期
                        expired_ids.append(point.id)

            # 删除过期的记忆
            if expired_ids:
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=expired_ids
                    )
                )
                logging.info(f"Deleted {len(expired_ids)} expired memories from Qdrant")

        except Exception as e:
            logging.error(f"Error in cleaning expired memories: {e}")

    def _expiry_worker(self):
        """过期清理工作线程，支持配置化的检查间隔和停止控制"""
        # 初始化停止事件
        if not hasattr(self, 'stop_expiry_event'):
            self.stop_expiry_event = threading.Event()
        
        while not self.stop_expiry_event.is_set():
            try:
                # 执行过期清理
                self._clean_expired_memories()

                # 使用配置的检查间隔（分钟转换为秒）
                sleep_seconds = self.vdb_check_interval * 60
                logging.info(f"数据库清理完成，将在 {self.vdb_check_interval} 分钟后再次执行")
                
                # 使用事件等待而不是sleep，以便可以被及时中断
                if self.stop_expiry_event.wait(timeout=sleep_seconds):
                    break

            except Exception as e:
                logging.error(f"Error in expiry worker: {e}")
                # 出错时使用最小30分钟的间隔进行重试，避免频繁失败
                retry_interval = max(self.vdb_check_interval, 30)
                logging.info(f"出错后将在 {retry_interval} 分钟后重试")
                
                # 使用事件等待而不是sleep，以便可以被及时中断
                if self.stop_expiry_event.wait(timeout=retry_interval * 60):
                    break

    def update_system_prompt_with_memo(self, system_prompt: str, memo: str) -> str:
        """
        在系统提示词后添加备忘录
        :param system_prompt: 原始系统提示词
        :param memo: 备忘录内容
        :return: 更新后的系统提示词
        """
        if memo and memo.strip():
            updated_prompt = f"{system_prompt}\n\n备忘录: {memo}"
            logging.info("Updated system prompt with memo")
            return updated_prompt
        return system_prompt

    def save_context_to_file(self, context_messages: List[Dict], filepath: str = 'context_history.json'):
        """
        保存上下文到文件
        :param context_messages: 上下文消息列表
        :param filepath: 保存文件路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(context_messages, f, ensure_ascii=False, indent=2)
            logging.info(f"Context saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error saving context to file: {e}")
            return False

    def load_context_from_file(self, filepath: str = 'context_history.json') -> List[Dict]:
        """
        从文件加载上下文
        :param filepath: 上下文文件路径
        :return: 上下文消息列表
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    context_messages = json.load(f)
                logging.info(f"Context loaded from {filepath}")
                return context_messages
            else:
                logging.info(f"Context file {filepath} does not exist, returning empty context")
                return []
        except Exception as e:
            logging.error(f"Error loading context from file: {e}")
            return []

    def save_multiple_context_files(self, context_messages: List[Dict], system_prompt: str = ""):
        """
        保存多个上下文文件以实现不同策略
        :param context_messages: 上下文消息列表
        :param system_prompt: 系统提示词（始终保留在上下文中）
        """
        try:
            # 文件1：短期上下文 - 在摘要后清空
            short_term_context = []
            self.save_context_to_file(short_term_context, 'context_short_term.json')

            # 文件2：保留设定数量的上下文条数
            max_context_items = self.config.get('max_context_items', 50)  # 默认保留50条
            if len(context_messages) > max_context_items:
                medium_term_context = context_messages[-max_context_items:]
            else:
                medium_term_context = context_messages[:]

            # 确保系统提示词始终在上下文中
            if system_prompt and system_prompt not in [msg.get('content', '') for msg in medium_term_context if msg.get('role') == 'system']:
                # 如果系统提示词不在上下文中，添加它
                medium_term_context.insert(0, {
                    'role': 'system',
                    'content': system_prompt,
                    'timestamp': datetime.now().isoformat()
                })

            self.save_context_to_file(medium_term_context, 'context_medium_term.json')

            # 文件3：长期备份 - 永不清空
            # 加载现有长期上下文并添加新内容
            long_term_context = self.load_context_from_file('context_long_term.json')
            # 添加新消息（排除重复的系统消息）
            for msg in context_messages:
                if msg not in long_term_context:
                    long_term_context.append(msg)

            self.save_context_to_file(long_term_context, 'context_long_term.json')

            logging.info(f"Saved multiple context files: short_term (cleared), medium_term ({len(medium_term_context)} items), long_term ({len(long_term_context)} items)")
            return True

        except Exception as e:
            logging.error(f"Error saving multiple context files: {e}")
            return False

    def load_multiple_context_files(self) -> Dict[str, List[Dict]]:
        """
        加载多个上下文文件
        :return: 包含不同上下文文件内容的字典
        """
        try:
            context_data = {
                'short_term': self.load_context_from_file('context_short_term.json'),
                'medium_term': self.load_context_from_file('context_medium_term.json'),
                'long_term': self.load_context_from_file('context_long_term.json')
            }
            return context_data
        except Exception as e:
            logging.error(f"Error loading multiple context files: {e}")
            return {
                'short_term': [],
                'medium_term': [],
                'long_term': []
            }

    def get_effective_context(self, system_prompt: str = "", context_type: str = "medium") -> List[Dict]:
        """
        获取有效的上下文（确保系统提示词始终在上下文中）
        :param system_prompt: 系统提示词
        :param context_type: 上下文类型 ('short', 'medium', 'long')
        :return: 包含系统提示词的有效上下文
        """
        try:
            if context_type == "short":
                context = self.load_context_from_file('context_short_term.json')
            elif context_type == "long":
                context = self.load_context_from_file('context_long_term.json')
            else:  # medium
                context = self.load_context_from_file('context_medium_term.json')

            # 确保系统提示词始终在上下文中
            if system_prompt:
                # 检查是否已存在系统提示词
                has_system_prompt = any(
                    msg.get('role') == 'system' and system_prompt in msg.get('content', '')
                    for msg in context
                )

                if not has_system_prompt:
                    # 在上下文开头添加系统提示词
                    context.insert(0, {
                        'role': 'system',
                        'content': system_prompt,
                        'timestamp': datetime.now().isoformat()
                    })

            return context
        except Exception as e:
            logging.error(f"Error getting effective context: {e}")
            # 返回包含系统提示词的基本上下文
            if system_prompt:
                return [{
                    'role': 'system',
                    'content': system_prompt,
                    'timestamp': datetime.now().isoformat()
                }]
            else:
                return []

    def update_db_management_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新数据库管理设置
        :param settings: 包含要更新设置的字典
        :return: 更新后的设置字典
        """
        try:
            # 验证输入
            if not isinstance(settings, dict):
                logging.warning("Invalid settings type provided to update_db_management_settings")
                return self.get_db_management_settings()
            
            old_auto_run = self.vdb_auto_run
            old_check_interval = self.vdb_check_interval
            old_qdrant_host = self.config.get('qdrant_host', 'localhost')
            old_qdrant_port = self.config.get('qdrant_port', 6333)
            qdrant_config_changed = False
            
            # 更新自动运行设置
            if 'auto_run' in settings:
                self.vdb_auto_run = bool(settings['auto_run'])
                logging.info(f"Updated vdb_auto_run to: {self.vdb_auto_run}")
            
            # 更新启动时运行设置
            if 'startup_run' in settings:
                self.vdb_startup_run = bool(settings['startup_run'])
                logging.info(f"Updated vdb_startup_run to: {self.vdb_startup_run}")
            
            # 更新检查间隔设置（分钟）
            if 'check_interval' in settings:
                try:
                    interval = int(settings['check_interval'])
                    # 确保间隔在合理范围内（5分钟到24小时）
                    self.vdb_check_interval = max(5, min(interval, 1440))
                    logging.info(f"Updated vdb_check_interval to: {self.vdb_check_interval} minutes")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid check_interval value: {settings['check_interval']}")
            
            # 更新Qdrant主机地址
            if 'qdrant_host' in settings:
                new_host = settings['qdrant_host']
                if new_host != old_qdrant_host:
                    self.config['qdrant_host'] = new_host
                    qdrant_config_changed = True
                    logging.info(f"Updated qdrant_host to: {new_host}")
            
            # 更新Qdrant端口
            if 'qdrant_port' in settings:
                try:
                    new_port = int(settings['qdrant_port'])
                    if new_port != old_qdrant_port:
                        self.config['qdrant_port'] = new_port
                        qdrant_config_changed = True
                        logging.info(f"Updated qdrant_port to: {new_port}")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid qdrant_port value: {settings['qdrant_port']}")
            
            # 将设置保存到配置中
            self.config['vdb_auto_run'] = self.vdb_auto_run
            self.config['vdb_startup_run'] = self.vdb_startup_run
            self.config['vdb_check_interval'] = self.vdb_check_interval
            
            # 如果Qdrant配置发生了变化，重新初始化Qdrant客户端
            if qdrant_config_changed:
                self._reinitialize_qdrant_client()
            
            # 处理自动运行状态变化
            if not old_auto_run and self.vdb_auto_run:
                # 从禁用变为启用，启动新线程
                # 先停止现有的线程（如果有的话）
                if hasattr(self, 'stop_expiry_event'):
                    self.stop_expiry_event.set()
                
                # 创建新的停止事件
                self.stop_expiry_event = threading.Event()
                
                # 启动过期清理线程
                self.expiry_thread = threading.Thread(target=self._expiry_worker, daemon=True)
                self.expiry_thread.start()
                logging.info("Started new database cleanup thread")
                
                # 启动上下文摘要线程
                if not hasattr(self, 'summary_thread') or not self.summary_thread.is_alive():
                    self.summary_thread = threading.Thread(target=self._context_summary_worker, daemon=True)
                    self.summary_thread.start()
                    logging.info("Started new context summary thread")
            elif old_auto_run and not self.vdb_auto_run:
                # 从启用变为禁用，需要停止线程
                if hasattr(self, 'stop_expiry_event'):
                    self.stop_expiry_event.set()
                    logging.info("Stopped database cleanup thread")
                else:
                    logging.info("Disabled automatic database cleanup")
                
                # 停止上下文摘要线程
                # 注意：由于Python线程无法直接停止，我们只能依靠线程自行检查状态
                logging.info("Disabled context summary thread (will stop on next cycle)")
            elif self.vdb_auto_run:
                # 如果自动运行保持启用且间隔发生变化，重新启动线程以应用新间隔
                if hasattr(self, 'expiry_thread') and self.vdb_check_interval != old_check_interval:
                    # 停止现有线程
                    if hasattr(self, 'stop_expiry_event'):
                        self.stop_expiry_event.set()
                    
                    # 创建新的停止事件
                    self.stop_expiry_event = threading.Event()
                    
                    # 启动新线程
                    self.expiry_thread = threading.Thread(target=self._expiry_worker, daemon=True)
                    self.expiry_thread.start()
                    logging.info("Re-started database cleanup thread with new interval")
                
            return self.get_db_management_settings()
        except Exception as e:
            logging.error(f"Error updating DB management settings: {e}", exc_info=True)
            # 返回当前设置，不进行任何更改
            return self.get_db_management_settings()
    
    def _reinitialize_qdrant_client(self):
        """
        重新初始化Qdrant客户端
        当Qdrant配置（主机地址或端口）发生变化时调用
        """
        try:
            # 关闭现有的Qdrant客户端连接（如果存在）
            if hasattr(self, 'qdrant_client') and self.qdrant_client:
                # QdrantClient没有显式的关闭方法，但我们可以通过删除引用来释放资源
                del self.qdrant_client
                self.qdrant_client = None
                logging.info("Closed existing Qdrant client connection")
            
            # 从配置获取Qdrant连接信息
            qdrant_host = self.config.get('qdrant_host', 'localhost')
            qdrant_port = self.config.get('qdrant_port', 6333)
            
            # 创建新的Qdrant客户端
            self.qdrant_client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                timeout=30
            )
            
            # 检查集合是否存在，不存在则创建
            try:
                self.qdrant_client.get_collection(self.collection_name)
                logging.info(f"Connected to existing Qdrant collection: {self.collection_name}")
            except:
                # 创建新的向量集合 - 使用较小的向量维度以兼容更多模型
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # 使用384维向量
                )
                logging.info(f"Created new Qdrant collection: {self.collection_name}")
            
            logging.info(f"Reinitialized Qdrant client with host={qdrant_host}, port={qdrant_port}")
        except Exception as e:
            logging.error(f"Error reinitializing Qdrant client: {e}")
            logging.info("Falling back to simulated embedding approach")
            self.qdrant_client = None
    
    def get_db_management_settings(self) -> Dict[str, Any]:
        """
        获取当前数据库管理设置
        :return: 包含当前设置的字典
        """
        return {
            'auto_run': self.vdb_auto_run,
            'startup_run': self.vdb_startup_run,
            'check_interval': self.vdb_check_interval,
            'qdrant_host': self.config.get('qdrant_host', 'localhost'),
            'qdrant_port': self.config.get('qdrant_port', 6333)
        }
    
    def trigger_db_management(self) -> Dict[str, Any]:
        """
        手动触发数据库管理操作（清理过期记忆和生成摘要）
        :return: 操作结果信息
        """
        try:
            logging.info("Manual DB management triggered")
            
            # 启动一个新线程执行清理操作，避免阻塞
            thread = threading.Thread(target=self._execute_manual_db_management, daemon=True)
            thread.start()
            
            return {
                'status': 'success',
                'message': '数据库管理操作已在后台启动',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error triggering DB management: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'触发数据库管理失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def start_vdb_management(self) -> Dict[str, Any]:
        """
        供LLM工具调用启动VDB管理
        启用VDB自动管理功能，包括自动清理和上下文摘要
        :return: 操作结果信息
        """
        try:
            # 检查是否已经启用
            if self.vdb_auto_run:
                return {
                    'status': 'info',
                    'message': 'VDB自动管理功能已经启用',
                    'timestamp': datetime.now().isoformat()
                }
            
            # 启用自动运行
            self.vdb_auto_run = True
            self.config['vdb_auto_run'] = True
            
            # 先停止现有的线程（如果有的话）
            if hasattr(self, 'stop_expiry_event'):
                self.stop_expiry_event.set()
            
            # 创建新的停止事件
            self.stop_expiry_event = threading.Event()
            
            # 启动过期清理线程
            self.expiry_thread = threading.Thread(target=self._expiry_worker, daemon=True)
            self.expiry_thread.start()
            logging.info("Started new database cleanup thread")
            
            # 启动上下文摘要线程
            if not hasattr(self, 'summary_thread') or not self.summary_thread.is_alive():
                self.summary_thread = threading.Thread(target=self._context_summary_worker, daemon=True)
                self.summary_thread.start()
                logging.info("Started new context summary thread")
            
            return {
                'status': 'success',
                'message': 'VDB自动管理功能已成功启用',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error starting VDB management: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'启动VDB管理失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_manual_db_management(self):
        """
        执行手动触发的数据库管理操作
        这是一个内部方法，在单独的线程中运行
        """
        try:
            # 清理过期记忆
            self._clean_expired_memories()
            
            # 生成上下文摘要
            self._generate_context_summary()
            
            logging.info("Manual DB management completed")
        except Exception as e:
            logging.error(f"Error in manual DB management execution: {e}", exc_info=True)