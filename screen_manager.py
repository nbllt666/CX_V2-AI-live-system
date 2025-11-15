import logging
import pyautogui
import time
from typing import Dict, Any, Optional
from PIL import Image
import os
import threading
import queue

class ScreenManager:
    """
    屏幕管理器 - 处理屏幕截图和点击功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.screenshot_dir = config.get('screenshot_dir', './screenshots')
        
        # 确保截图目录存在
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # 连续操作模式相关
        self.continuous_mode = False
        self.continuous_operations_queue = queue.Queue()
        
        logging.info("Screen manager initialized")

    def take_screenshot(self, filename: Optional[str] = None) -> str:
        """
        截取屏幕截图
        :param filename: 截图文件名（可选）
        :return: 截图文件路径
        """
        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
            
            filepath = os.path.join(self.screenshot_dir, filename)
            
            # 截取全屏
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            
            logging.info(f"Screenshot saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error taking screenshot: {e}")
            raise

    def click_position(self, x: int, y: int) -> bool:
        """
        点击指定位置
        :param x: X坐标
        :param y: Y坐标
        :return: 成功与否
        """
        try:
            # 移动到指定位置并点击
            pyautogui.click(x, y)
            
            logging.info(f"Clicked position: ({x}, {y})")
            return True
            
        except Exception as e:
            logging.error(f"Error clicking position ({x}, {y}): {e}")
            return False

    def drag_from_to(self, x1: int, y1: int, x2: int, y2: int, duration: float = 1.0) -> bool:
        """
        从一个位置拖动到另一个位置
        :param x1, y1: 起始位置
        :param x2, y2: 结束位置
        :param duration: 拖动持续时间
        :return: 成功与否
        """
        try:
            pyautogui.drag(x2-x1, y2-y1, duration=duration, button='left')
            
            logging.info(f"Dragged from ({x1}, {y1}) to ({x2}, {y2}) in {duration}s")
            return True
            
        except Exception as e:
            logging.error(f"Error dragging from ({x1}, {y1}) to ({x2}, {y2}): {e}")
            return False

    def start_continuous_mode(self):
        """开始连续操作模式"""
        self.continuous_mode = True
        logging.info("Started continuous mode")
        
        # 启动连续操作处理线程
        self.continuous_thread = threading.Thread(target=self._process_continuous_operations, daemon=True)
        self.continuous_thread.start()

    def stop_continuous_mode(self):
        """停止连续操作模式"""
        self.continuous_mode = False
        logging.info("Stopped continuous mode")

    def _process_continuous_operations(self):
        """处理连续操作的后台线程"""
        while self.continuous_mode:
            try:
                # 等待下一个操作指令
                operation = self.continuous_operations_queue.get(timeout=1)
                if operation is None:
                    break
                
                # 执行操作
                op_type = operation.get('type')
                if op_type == 'click':
                    x, y = operation.get('x', 0), operation.get('y', 0)
                    self.click_position(x, y)
                    
                    # 如果在连续模式下，执行完操作后发送截图给主模型
                    if self.continuous_mode:
                        try:
                            screenshot_path = self.take_screenshot()
                            # 这里应该调用主模型处理截图
                            # self.main_model.process_message("请根据屏幕截图继续操作", image_path=screenshot_path)
                        except Exception as e:
                            logging.error(f"Error taking screenshot after operation: {e}")
                
                self.continuous_operations_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in continuous operations: {e}")
                continue

    def add_continuous_operation(self, operation: Dict[str, Any]):
        """添加连续操作到队列"""
        if self.continuous_mode:
            try:
                self.continuous_operations_queue.put(operation)
                logging.info(f"Added continuous operation: {operation}")
            except Exception as e:
                logging.error(f"Error adding continuous operation: {e}")
        else:
            logging.warning("Continuous mode not active, operation ignored")

class ScreenCapture:
    """
    屏幕捕获类 - 专门用于实时屏幕捕获
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capture_enabled = config.get('screen_capture_enabled', True)
        self.capture_interval = config.get('screen_capture_interval', 5.0)  # 秒
        
    def capture_region(self, region: tuple = None) -> Image.Image:
        """
        捕获屏幕区域
        :param region: 区域 (left, top, width, height)，如果为None则捕获全屏
        :return: PIL图像对象
        """
        try:
            if region:
                # 捕获指定区域
                screenshot = pyautogui.screenshot(region=region)
            else:
                # 捕获全屏
                screenshot = pyautogui.screenshot()
            
            return screenshot
            
        except Exception as e:
            logging.error(f"Error capturing screen region: {e}")
            # 返回一个空图像作为替代
            return Image.new('RGB', (100, 100), color='black')

    def get_screen_size(self) -> tuple:
        """获取屏幕尺寸"""
        width, height = pyautogui.size()
        return width, height

    def enable_capture(self, enabled: bool):
        """启用/禁用屏幕捕获"""
        self.capture_enabled = enabled
        logging.info(f"Screen capture {'enabled' if enabled else 'disabled'}")