import logging
import logging.handlers
import sys
import os
import threading
import time
from typing import Dict, Any, Callable
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class HotReloadHandler(FileSystemEventHandler):
    """热重载处理器"""
    
    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self.callback = callback
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            # 等待文件写入完成
            time.sleep(0.5)
            self.callback(event.src_path)

class HotReloader:
    """热重载管理器"""
    
    def __init__(self, paths_to_watch: list, callback: Callable[[str], None]):
        self.paths_to_watch = paths_to_watch
        self.callback = callback
        self.observer = Observer()
        
        # 设置热重载处理器
        for path in paths_to_watch:
            self.observer.schedule(
                HotReloadHandler(self.callback), 
                path, 
                recursive=True
            )
    
    def start(self):
        """启动热重载监听"""
        self.observer.start()
        logging.info(f"Hot reload started for paths: {self.paths_to_watch}")
    
    def stop(self):
        """停止热重载监听"""
        self.observer.stop()
        self.observer.join()
        logging.info("Hot reload stopped")

class RobustManager:
    """健壮性管理器 - 处理各种可能的错误和异常"""
    
    def __init__(self):
        pass
    
    def safe_execute(self, func, *args, **kwargs):
        """
        安全执行函数，捕获所有异常
        :param func: 要执行的函数
        :param args: 函数参数
        :param kwargs: 函数关键字参数
        :return: (success: bool, result: Any, error: str)
        """
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logging.error(error_msg)
            return False, None, error_msg
    
    def retry_execute(self, func, max_retries=3, delay=1, *args, **kwargs):
        """
        带重试的函数执行
        :param func: 要执行的函数
        :param max_retries: 最大重试次数
        :param delay: 重试间隔（秒）
        :param args: 函数参数
        :param kwargs: 函数关键字参数
        :return: 函数执行结果
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:  # 最后一次尝试
                    logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying in {delay}s...")
                time.sleep(delay)

class LoggingSetup:
    """日志设置"""
    
    def __init__(self, log_dir: str = './logs', log_level: int = logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        # 创建日志格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 设置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # 清除现有的处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建文件处理器 - 使用轮转日志
        log_file = os.path.join(self.log_dir, 'ai_live.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB, 保留5个备份
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到根记录器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("Logging setup completed")

class TTLFilter:
    """TTL过滤器 - 用于过滤过期数据"""
    
    def __init__(self):
        self.data_store = {}  # 存储数据及其过期时间
    
    def add_item(self, key: str, value: Any, ttl_seconds: int):
        """添加带TTL的项目"""
        import time
        expiry_time = time.time() + ttl_seconds
        self.data_store[key] = {
            'value': value,
            'expiry': expiry_time
        }
    
    def get_item(self, key: str) -> Any:
        """获取项目，如果过期则返回None"""
        import time
        if key in self.data_store:
            item = self.data_store[key]
            if time.time() < item['expiry']:
                return item['value']
            else:
                # 过期，删除项目
                del self.data_store[key]
        return None
    
    def cleanup_expired(self):
        """清理过期项目"""
        import time
        current_time = time.time()
        expired_keys = [
            key for key, item in self.data_store.items()
            if current_time >= item['expiry']
        ]
        
        for key in expired_keys:
            del self.data_store[key]
        
        return len(expired_keys)

def setup_error_handling():
    """设置全局错误处理"""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """处理未捕获的异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许键盘中断正常退出
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception

def ensure_robust_service(service_func, service_name: str):
    """确保服务的健壮性"""
    def wrapper(*args, **kwargs):
        max_retries = 5
        retry_delay = 5  # 秒
        
        for attempt in range(max_retries):
            try:
                return service_func(*args, **kwargs)
            except Exception as e:
                logging.error(f"{service_name} failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying {service_name} in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.critical(f"{service_name} failed after {max_retries} attempts")
                    raise
    return wrapper

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_counts = {}
        self.max_errors = 10  # 连续错误阈值
    
    def record_error(self, error_type: str):
        """记录错误"""
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
        else:
            self.error_counts[error_type] = 1
    
    def should_shutdown(self, error_type: str) -> bool:
        """检查是否应该关闭服务"""
        return self.error_counts.get(error_type, 0) >= self.max_errors
    
    def reset_error_count(self, error_type: str):
        """重置错误计数"""
        if error_type in self.error_counts:
            self.error_counts[error_type] = 0

# 初始化错误处理
setup_error_handling()