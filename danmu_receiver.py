import asyncio
import json
import logging
from asyncio import Event
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Tuple, Callable, Dict, Any
import aiohttp
from reactivestreams.subscriber import Subscriber
from reactivestreams.subscription import Subscription
from rsocket.helpers import single_transport_provider
from rsocket.payload import Payload
from rsocket.rsocket_client import RSocketClient
from rsocket.streams.stream_from_async_generator import StreamFromAsyncGenerator

class DanmuReceiver:
    """
    弹幕接收器 - 连接到"让弹幕飞"服务
    """
    
    def __init__(self, config: Dict[str, Any], vdb_manager, message_handler: Callable[[str], None]):
        self.config = config
        self.vdb_manager = vdb_manager
        self.message_handler = message_handler
        self.websocket_uri = config.get('danmu_websocket_uri', 'ws://localhost:9898')
        self.task_ids = config.get('danmu_task_ids', [])
        self.running = False
        self.client = None
        
        # 订阅载荷
        self.subscribe_payload_json = {
            "data": {
                "taskIds": self.task_ids,
                "cmd": "SUBSCRIBE"
            }
        }

    class DanmuSubscriber(Subscriber):
        def __init__(self, wait_for_responder_complete: Event, vdb_manager, message_handler: Callable[[str], None]):
            super().__init__()
            self.subscription = None
            self._wait_for_responder_complete = wait_for_responder_complete
            self.vdb_manager = vdb_manager
            self.message_handler = message_handler

        def on_subscribe(self, subscription: Subscription):
            self.subscription = subscription
            self.subscription.request(0x7FFFFFFF)

        def on_next(self, value: Payload, is_complete=False):
            try:
                msg_dto = json.loads(value.data.decode('utf-8')) if isinstance(value.data, bytes) else json.loads(value.data)
                if not isinstance(msg_dto, dict):
                    return
                    
                msg_type = msg_dto.get('type')
                
                # 处理弹幕
                if msg_type == "DANMU":
                    msg = msg_dto['msg']
                    username = msg['username']
                    content = msg['content']
                    
                    logging.info(
                        f"{msg_dto['roomId']} 收到弹幕 {str(msg['badgeLevel']) + str(msg['badgeName']) if msg['badgeLevel'] != 0 else ''} {msg['username']}({str(msg['uid'])})：{msg['content']}"
                    )
                    
                    # 使用VDB管理器过滤弹幕
                    if self.vdb_manager.filter_danmu(content):
                        # 将弹幕内容交给消息处理器
                        self.message_handler(f"弹幕用户 {username} 说：{content}")
                    else:
                        logging.info(f"弹幕被过滤: {content}")
                        
                # 处理礼物
                elif msg_type == "GIFT":
                    msg = msg_dto['msg']
                    logging.info(
                        f"{msg_dto['roomId']} 收到礼物 {str(msg['badgeLevel']) + str(msg['badgeName']) if msg['badgeLevel'] != 0 else ''} {msg['username']}({str(msg['uid'])}) {str(msg['data']['action']) if msg.get('data') is not None and msg.get('data').get('action') is not None else '赠送'} {msg['giftName']}({str(msg['giftId'])})x{str(msg['giftCount'])}({str(msg['giftPrice'])})"
                    )
                    
                    # 可以根据礼物触发特定响应
                    gift_message = f"感谢 {msg['username']} 赠送的 {msg['giftName']} x{msg['giftCount']}!"
                    self.message_handler(gift_message)
                    
                else:
                    logging.info("收到消息 " + json.dumps(msg_dto))
                    
                if is_complete:
                    self._wait_for_responder_complete.set()
                    
            except Exception as e:
                logging.error(f"Error processing danmu message: {e}")

        def on_error(self, exception: Exception):
            logging.error('Error from danmu server: ' + str(exception))
            self._wait_for_responder_complete.set()

        def on_complete(self):
            logging.info('Danmu server connection completed')
            self._wait_for_responder_complete.set()

    @asynccontextmanager
    async def connect(self):
        """
        创建一个Client，建立连接并return
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.websocket_uri) as websocket:
                    async with RSocketClient(
                            single_transport_provider(
                                type.__dict__['TransportAioHttpClient'](websocket=websocket)  # 修正导入
                            ),
                            keep_alive_period=timedelta(seconds=30),
                            max_lifetime_period=timedelta(days=1)
                    ) as client:
                        self.client = client
                        yield client
        except Exception as e:
            logging.error(f"Error connecting to danmu server: {e}")
            raise

    async def start_receiving(self):
        """开始接收弹幕"""
        self.running = True
        logging.info(f"Starting to receive danmu from {self.websocket_uri} with task IDs: {self.task_ids}")
        
        while self.running:
            try:
                # 1 建立连接
                async with self.connect() as client:
                    # 阻塞等待Channel关闭事件
                    channel_completion_event = Event()

                    # 定义Client向Channel发送消息的Publisher
                    async def generator() -> AsyncGenerator[Tuple[Payload, bool], None]:
                        # 2 发送订阅Task的请求
                        yield Payload(
                            data=json.dumps(self.subscribe_payload_json["data"]).encode()
                        ), False
                        # 发送了一条订阅消息后直接暂停发送即可
                        await Event().wait()

                    stream = StreamFromAsyncGenerator(generator)
                    
                    # Client请求一个Channel
                    requested = client.request_channel(Payload(), stream)
                    
                    # 3 订阅Channel，DanmuSubscriber用于处理Server通过Channel回复的消息
                    subscriber = self.DanmuSubscriber(
                        channel_completion_event, 
                        self.vdb_manager, 
                        self.message_handler
                    )
                    requested.subscribe(subscriber)
                    
                    await channel_completion_event.wait()
                    
            except Exception as e:
                logging.error(f"Error in danmu receiving loop: {e}")
                # 等待一段时间后重试
                await asyncio.sleep(5)

    def stop_receiving(self):
        """停止接收弹幕"""
        self.running = False
        if self.client:
            try:
                self.client.close()
            except:
                pass

# 修正TransportAioHttpClient的导入问题
try:
    from rsocket.transports.aiohttp_websocket import TransportAioHttpClient
except ImportError:
    # 如果导入失败，定义一个临时类
    class TransportAioHttpClient:
        def __init__(self, websocket):
            self.websocket = websocket

# 使用正确导入的类
class DanmuReceiverFixed(DanmuReceiver):
    @asynccontextmanager
    async def connect(self):
        """
        创建一个Client，建立连接并return
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.websocket_uri) as websocket:
                    async with RSocketClient(
                            single_transport_provider(TransportAioHttpClient(websocket=websocket)),
                            keep_alive_period=timedelta(seconds=30),
                            max_lifetime_period=timedelta(days=1)
                    ) as client:
                        self.client = client
                        yield client
        except Exception as e:
            logging.error(f"Error connecting to danmu server: {e}")
            raise