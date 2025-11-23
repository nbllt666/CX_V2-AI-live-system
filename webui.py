# 尝试导入eventlet，如果失败则提供友好提示
try:
    import eventlet
    eventlet.monkey_patch()
    print("Eventlet 已成功导入并应用monkey_patch")
except ImportError:
    print("警告: Eventlet 导入失败。将使用默认的Flask服务器而不是Socket.IO增强的服务器。")
    print("请确保已正确安装eventlet: pip install eventlet")
    # 继续执行，Flask仍然可以工作，只是可能性能较差

import os
import sys
import json
import logging
import threading
import time
import socketio
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from config_manager import ConfigManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask和Socket.IO实例
app = Flask(__name__, template_folder='templates')
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# 配置上传文件夹和允许的文件类型
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局变量
config = {}
system_status = {
    'status': 'stopped',  # stopped, running, restarting
    'platform': None,
    'room_id': None,
    'connection_time': None
}
logs = []

# 模拟系统控制函数
class SystemController:
    def __init__(self):
        self.running = False
        self.thread = None
        # 尝试导入AILiveSystem
        try:
            from main import AILiveSystem
            self.ai_system_class = AILiveSystem
        except ImportError:
            self.ai_system_class = None
            logger.warning("无法导入AILiveSystem类")
        
        # 存储系统实例
        self.ai_system_instance = None
    
    def start(self):
        logger.info("系统启动中...")
        try:
            # 如果已经有运行的实例，先停止它
            if self.ai_system_instance and hasattr(self.ai_system_instance, 'running') and self.ai_system_instance.running:
                self.ai_system_instance.stop()
            
            # 创建新的系统实例
            if self.ai_system_class:
                # 加载配置
                config_path = 'config/config.json'
                local_config = {}
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        local_config = json.load(f)
                except FileNotFoundError:
                    logger.error(f"配置文件未找到: {config_path}")
                    return False
                except Exception as e:
                    logger.error(f"加载配置文件失败: {e}")
                    return False
                
                # 创建AI系统实例
                self.ai_system_instance = self.ai_system_class(local_config)
                
                # 启动系统
                self.ai_system_instance.start()
                self.running = True
                
                system_status['status'] = 'running'
                system_status['connection_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                sio.emit('status_update', system_status)
                logger.info("系统启动成功")
                return True
            else:
                logger.error("AILiveSystem类不可用")
                return False
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            return False
    
    def stop(self):
        logger.info("系统停止中...")
        try:
            if self.ai_system_instance and hasattr(self.ai_system_instance, 'running') and self.ai_system_instance.running:
                self.ai_system_instance.stop()
            
            self.running = False
            system_status['status'] = 'stopped'
            sio.emit('status_update', system_status)
            logger.info("系统停止成功")
            return True
        except Exception as e:
            logger.error(f"系统停止失败: {e}")
            return False
    
    def restart(self):
        logger.info("系统重启中...")
        system_status['status'] = 'restarting'
        sio.emit('status_update', system_status)
        
        # 先停止
        self.stop()
        # 等待一段时间确保完全停止
        time.sleep(2)
        # 再启动
        success = self.start()
        if success:
            logger.info("系统重启成功")
        else:
            logger.error("系统重启失败")
        return success
    
    def preheat_models(self):
        """预热模型功能"""
        logger.info("开始预热模型...")
        try:
            # 加载配置
            config_path = 'config/config.json'
            local_config = {}
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    local_config = json.load(f)
            except FileNotFoundError:
                logger.error(f"配置文件未找到: {config_path}")
                return False
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
                return False
            
            # 预热SenseVoice模型
            try:
                from audio_recorder import AudioRecorder
                recorder = AudioRecorder(local_config)
                recorder.preheat_sense_voice()
                logger.info("SenseVoice模型预热完成")
            except Exception as e:
                logger.error(f"SenseVoice模型预热失败: {e}")
            
            # 预热Ollama模型
            try:
                ollama_api_url = local_config.get('ollama_api_url', 'http://localhost:11434/api/chat')
                ollama_model = local_config.get('ollama_model', 'llama3.2')
                
                # 发送一个简单的请求到Ollama API来预热模型
                import requests
                import json
                
                payload = {
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
                
                response = requests.post(ollama_api_url, json=payload, timeout=30)
                if response.status_code == 200:
                    logger.info(f"Ollama模型 {ollama_model} 预热完成")
                else:
                    logger.error(f"Ollama模型预热失败，状态码: {response.status_code}")
            except Exception as e:
                logger.error(f"Ollama模型预热失败: {e}")
            
            logger.info("模型预热完成")
            return True
        except Exception as e:
            logger.error(f"模型预热过程中出现错误: {e}")
            return False

# 创建系统控制器实例
system_controller = SystemController()

# WebUI控制器类
class WebUIController:
    def __init__(self, vdb_manager=None):
        # 使用与main.py一致的配置文件路径
        self.config_path = 'config/config.json'
        self.vdb_manager = vdb_manager
        # 初始化配置管理器
        self.config_manager = ConfigManager(self.config_path)
        # 如果vdb_manager未提供，则尝试初始化一个
        if self.vdb_manager is None:
            try:
                from vdb_manager import VDBManager
                # 加载配置
                local_config = self._load_config()
                if local_config:
                    self.vdb_manager = VDBManager(local_config)
            except Exception as e:
                logger.warning(f"无法初始化VDB管理器: {e}")
        self._load_config()
    
    def load_config(self):
        """加载配置"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'r', encoding='utf-8') as f:
                global config
                config = json.load(f)
            logger.info(f"配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            return None
    
    def save_config(self, new_config):
        """保存配置"""
        try:
            # 使用配置管理器保存配置
            if self.config_manager.save_config(new_config):
                global config
                config = new_config
                logger.info(f"配置保存成功: {self.config_path}")
                # 通知其他组件配置已更新
                sio.emit('config_updated_broadcast', new_config)
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            return False
    
    def update_status(self, new_status):
        """更新系统状态"""
        global system_status
        system_status.update(new_status)
        sio.emit('status_update', system_status)
        return True
    
    def add_log_message(self, message, level='info'):
        """添加日志消息"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        logs.append(log_entry)
        # 保持日志数量限制
        if len(logs) > 1000:
            logs.pop(0)
        
        # 通过Socket.IO发送日志
        sio.emit('log_update', log_entry)
        return True
    
    def _load_config(self):
        """初始化加载配置"""
        global config
        try:
            # 使用配置管理器加载和验证配置
            config = self.config_manager.get_config_with_validation()
            logger.info(f"配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"配置加载失败: {str(e)}")
            config = {}
            return config

# 创建WebUI控制器实例
# 注意：当独立运行webui.py时，vdb_manager为None
# 当从main.py运行时，会在main.py中重新初始化带vdb_manager的WebUIController
webui_controller = WebUIController()

# 日志处理程序，将日志发送到前端
class SocketIOLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname.lower(),
                'message': self.format(record)
            }
            sio.emit('log_update', log_entry)
        except Exception as e:
            print(f"发送日志到前端失败: {e}")

# 添加自定义日志处理器
socketio_handler = SocketIOLogHandler()
socketio_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
logger.addHandler(socketio_handler)

# Flask路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    # 确保返回一个对象而不是null
    try:
        # 确保config是一个字典对象
        if config is None:
            return jsonify({})
        return jsonify(config if isinstance(config, dict) else {})
    except Exception as e:
        logger.error(f"获取配置时出错: {e}")
        return jsonify({})

@app.route('/api/config', methods=['POST'])
def update_config():
    new_config = request.json
    if webui_controller.save_config(new_config):
        return jsonify({'success': True, 'message': '配置保存成功'})
    else:
        return jsonify({'success': False, 'message': '配置保存失败'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(system_status)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    # 返回最近的日志
    limit = request.args.get('limit', 100, type=int)
    return jsonify(logs[-limit:])

@app.route('/api/system/start', methods=['POST'])
def start_system():
    if system_controller.start():
        return jsonify({'success': True, 'message': '系统启动成功'})
    else:
        return jsonify({'success': False, 'message': '系统启动失败'}), 500

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    if system_controller.stop():
        return jsonify({'success': True, 'message': '系统停止成功'})
    else:
        return jsonify({'success': False, 'message': '系统停止失败'}), 500

@app.route('/api/system/restart', methods=['POST'])
def restart_system():
    if system_controller.restart():
        return jsonify({'success': True, 'message': '系统重启成功'})
    else:
        return jsonify({'success': False, 'message': '系统重启失败'}), 500

@app.route('/api/config/reload', methods=['POST'])
def reload_config():
    """重新加载配置文件"""
    try:
        # 重新加载配置
        global config
        config = webui_controller.config_manager.get_config_with_validation()
        # 通知前端配置已更新
        sio.emit('config_updated_broadcast', config)
        return jsonify({'success': True, 'message': '配置重新加载成功', 'config': config})
    except Exception as e:
        logger.error(f"重新加载配置失败: {str(e)}")
        return jsonify({'success': False, 'message': f'配置重新加载失败: {str(e)}'}), 500

@app.route('/api/send_message', methods=['POST'])
def send_message():
    try:
        message = request.json.get('message', '')
        if not message:
            return jsonify({'success': False, 'message': '消息不能为空'}), 400
        
        # 这里模拟发送消息
        logger.info(f"发送消息: {message}")
        webui_controller.add_log_message(f"[发送] {message}")
        
        # 模拟回复
        reply = f"收到消息: {message}"
        webui_controller.add_log_message(f"[回复] {reply}")
        
        return jsonify({'success': True, 'message': '消息发送成功', 'reply': reply})
    except Exception as e:
        logger.error(f"发送消息失败: {e}")
        return jsonify({'success': False, 'message': f'消息发送失败: {str(e)}'}), 500

@app.route('/api/upload_config', methods=['POST'])
def upload_config():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有文件部分'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
                if webui_controller.save_config(new_config):
                    return jsonify({'success': True, 'message': '配置文件上传并应用成功'})
                else:
                    return jsonify({'success': False, 'message': '配置应用失败'}), 500
        except Exception as e:
            logger.error(f"处理上传的配置文件失败: {e}")
            return jsonify({'success': False, 'message': f'配置文件解析失败: {str(e)}'}), 400
    else:
        return jsonify({'success': False, 'message': '不支持的文件类型，只允许.json文件'}), 400

@app.route('/api/download_config', methods=['GET'])
def download_config():
    # 确保返回正确的配置文件路径
    return send_from_directory(os.path.dirname(webui_controller.config_path), 
                             os.path.basename(webui_controller.config_path), 
                             as_attachment=True)

@app.route('/api/control', methods=['POST'])
def control_system():
    """统一控制系统功能"""
    try:
        data = request.get_json()
        action = data.get('action')
        
        if not action:
            return jsonify({'success': False, 'message': '未指定操作'}), 400
            
        # 根据不同的操作类型进行处理
        if action == 'start':
            if system_controller.start():
                return jsonify({'success': True, 'message': '系统启动成功'})
            else:
                return jsonify({'success': False, 'message': '系统启动失败'}), 500
        elif action == 'stop':
            if system_controller.stop():
                return jsonify({'success': True, 'message': '系统停止成功'})
            else:
                return jsonify({'success': False, 'message': '系统停止失败'}), 500
        elif action == 'restart':
            if system_controller.restart():
                return jsonify({'success': True, 'message': '系统重启成功'})
            else:
                return jsonify({'success': False, 'message': '系统重启失败'}), 500
        elif action == 'control_melotts':
            # 处理MeloTTS控制命令
            try:
                melotts_action = data.get('melotts_action')
                if not melotts_action:
                    return jsonify({'success': False, 'message': '未指定MeloTTS操作'}), 400
                    
                # 获取当前配置
                global config
                current_config = config
                
                # 获取MeloTTS API URL
                api_url = current_config.get('melotts_api_url', 'http://127.0.0.1:8000')
                
                if melotts_action == 'start':
                    # 启动MeloTTS服务
                    try:
                        import requests
                        start_url = f"{api_url}/start"
                        response = requests.post(start_url, timeout=10)
                        if response.status_code == 200:
                            current_config['use_molotts'] = True
                            webui_controller.save_config(current_config)
                            logger.info("MeloTTS 服务已启用")
                            return jsonify({'success': True, 'message': 'MeloTTS 服务已启用'})
                        else:
                            logger.error(f"MeloTTS 启动失败，状态码: {response.status_code}")
                            return jsonify({'success': False, 'message': f'MeloTTS 启动失败，状态码: {response.status_code}'}), 500
                    except Exception as e:
                        logger.error(f"MeloTTS 启动异常: {e}")
                        return jsonify({'success': False, 'message': f'MeloTTS 启动异常: {str(e)}'}), 500
                    
                elif melotts_action == 'stop':
                    # 停止MeloTTS服务
                    try:
                        import requests
                        stop_url = f"{api_url}/stop"
                        response = requests.post(stop_url, timeout=10)
                        if response.status_code == 200:
                            current_config['use_molotts'] = False
                            webui_controller.save_config(current_config)
                            logger.info("MeloTTS 服务已禁用")
                            return jsonify({'success': True, 'message': 'MeloTTS 服务已禁用'})
                        else:
                            logger.error(f"MeloTTS 停止失败，状态码: {response.status_code}")
                            return jsonify({'success': False, 'message': f'MeloTTS 停止失败，状态码: {response.status_code}'}), 500
                    except Exception as e:
                        logger.error(f"MeloTTS 停止异常: {e}")
                        return jsonify({'success': False, 'message': f'MeloTTS 停止异常: {str(e)}'}), 500
                    
                elif melotts_action == 'warmup':
                    # 预热MeloTTS模型
                    logger.info('开始预热MeloTTS模型...')
                    try:
                        import requests
                        import json
                        
                        test_url = f"{api_url}/tts"
                        
                        # 准备测试数据来预热模型
                        test_data = {
                            "text": "系统预热测试",
                            "speaker_id": int(current_config.get('molotts_speaker_id', 0)),
                            "sdp_ratio": float(current_config.get('molotts_sdp_ratio', 0.2)),
                            "noise_scale": float(current_config.get('molotts_noise_scale', 0.6)),
                            "noise_scale_w": float(current_config.get('molotts_noise_scale_w', 0.8)),
                            "speed": float(current_config.get('molotts_speed', 1.0))
                        }
                        
                        # 发送预热请求
                        response = requests.post(test_url, json=test_data, timeout=30)
                        if response.status_code == 200:
                            logger.info('MeloTTS 模型预热成功')
                            return jsonify({'success': True, 'message': 'MeloTTS 模型预热成功'})
                        else:
                            logger.warning(f'MeloTTS 模型预热请求失败，状态码: {response.status_code}')
                            return jsonify({'success': True, 'message': f'MeloTTS 预热请求完成，状态码: {response.status_code}'})
                    except Exception as e:
                        logger.error(f'MeloTTS 模型预热失败: {str(e)}')
                        return jsonify({'success': False, 'message': f'MeloTTS 预热失败: {str(e)}'})
                        
                else:
                    return jsonify({
                        'success': False, 
                        'message': f'不支持的MeloTTS操作: {melotts_action}'
                    }), 400
                    
            except Exception as e:
                logger.error(f"控制MeloTTS服务失败: {e}")
                return jsonify({'success': False, 'message': f'控制MeloTTS服务失败: {str(e)}'}), 500
        elif action == 'send_message':
            # 处理发送消息命令
            message = data.get('message', '')
            if not message:
                return jsonify({'success': False, 'message': '消息不能为空'}), 400
            
            # 记录消息
            logger.info(f"发送消息: {message}")
            webui_controller.add_log_message(f"[发送] {message}")
            
            # 将消息发送到AI系统
            try:
                if hasattr(system_controller, 'ai_system') and system_controller.ai_system:
                    # 使用AI系统的process_message方法处理消息
                    response = system_controller.ai_system.process_message(message)
                    reply = response.get('response', f"收到消息: {message}")
                    webui_controller.add_log_message(f"[回复] {reply}")
                    
                    return jsonify({
                        'success': True, 
                        'message': '消息发送成功', 
                        'reply': reply
                    })
                else:
                    # 如果AI系统未初始化，返回模拟回复
                    reply = f"收到消息: {message}"
                    webui_controller.add_log_message(f"[回复] {reply}")
                    
                    return jsonify({
                        'success': True, 
                        'message': '消息发送成功', 
                        'reply': reply
                    })
            except Exception as e:
                logger.error(f"处理消息失败: {e}")
                reply = f"处理消息时出错: {str(e)}"
                webui_controller.add_log_message(f"[错误] {reply}")
                
                return jsonify({
                    'success': False, 
                    'message': '消息处理失败', 
                    'reply': reply
                }), 500
        elif action == 'start_recording':
            # 处理开始录音命令
            logger.info("收到开始录音命令")
            webui_controller.add_log_message("[控制] 开始录音")
            
            # 触发实际的录音功能
            try:
                if hasattr(system_controller, 'ai_system') and system_controller.ai_system:
                    # 获取AI系统的音频录制器
                    audio_recorder = getattr(system_controller.ai_system, 'audio_recorder', None)
                    if audio_recorder:
                        audio_recorder.start_manual_recording()
                        return jsonify({'success': True, 'message': '开始录音'})
                    else:
                        return jsonify({'success': False, 'message': '音频录制器未初始化'}), 500
                else:
                    return jsonify({'success': False, 'message': 'AI系统未初始化'}), 500
            except Exception as e:
                logger.error(f"开始录音失败: {e}")
                return jsonify({'success': False, 'message': f'开始录音失败: {str(e)}'}), 500
        elif action == 'stop_recording':
            # 处理停止录音命令
            logger.info("收到停止录音命令")
            webui_controller.add_log_message("[控制] 停止录音")
            
            # 停止录音功能
            try:
                if hasattr(system_controller, 'ai_system') and system_controller.ai_system:
                    # 获取AI系统的音频录制器
                    audio_recorder = getattr(system_controller.ai_system, 'audio_recorder', None)
                    if audio_recorder:
                        # 调用专门的停止录音方法
                        audio_recorder.stop_manual_recording()
                        logger.info("录音已停止")
                        return jsonify({'success': True, 'message': '录音已停止'})
                    else:
                        return jsonify({'success': False, 'message': '音频录制器未初始化'}), 500
                else:
                    return jsonify({'success': False, 'message': 'AI系统未初始化'}), 500
            except Exception as e:
                logger.error(f"停止录音失败: {e}")
                return jsonify({'success': False, 'message': f'停止录音失败: {str(e)}'}), 500
        else:
            return jsonify({'success': False, 'message': f'不支持的操作: {action}'}), 400
            
    except Exception as e:
        logger.error(f"系统控制失败: {e}")
        return jsonify({'success': False, 'message': f'系统控制失败: {str(e)}'}), 500

@app.route('/api/control/vdb', methods=['POST'])
def control_vdb():
    """控制VDB管理功能"""
    try:
        data = request.get_json()
        action = data.get('action')
        
        if not action:
            return jsonify({'success': False, 'message': '未指定操作'}), 400
            
        # 使用WebUIController中的VDB管理器实例
        vdb_manager = webui_controller.vdb_manager
        
        if action == 'get_status':
            # 获取VDB当前状态和配置
            if vdb_manager:
                settings = vdb_manager.get_db_management_settings()
                return jsonify({
                    'success': True,
                    'vdb_auto_run': settings.get('auto_run', True),
                    'vdb_startup_run': settings.get('startup_run', True),
                    'vdb_check_interval': settings.get('check_interval', 60),
                    'enable_llm_vdb_control': config.get('enable_llm_vdb_control', True)
                })
            else:
                # 如果VDB管理器未初始化，返回默认值
                return jsonify({
                    'success': True,
                    'vdb_auto_run': True,
                    'vdb_startup_run': True,
                    'vdb_check_interval': 60,
                    'enable_llm_vdb_control': config.get('enable_llm_vdb_control', True)
                })
        elif action == 'update_settings' or action == 'set_config':
            # 更新VDB设置
            if action == 'update_settings':
                settings = data.get('settings', {})
            else:  # set_config
                # 从前端发送的config对象中提取设置
                config_data = data.get('config', {})
                # 映射前端字段名到VDB管理器期望的字段名
                settings = {
                    'auto_run': config_data.get('vdb_auto_run', True),
                    'startup_run': config_data.get('vdb_startup_run', True),
                    'check_interval': config_data.get('vdb_check_interval', 60),
                    'qdrant_host': config_data.get('qdrant_host', 'localhost'),
                    'qdrant_port': config_data.get('qdrant_port', 6333)
                }
                
                # 更新enable_llm_vdb_control配置
                enable_llm_vdb_control = config_data.get('enable_llm_vdb_control')
                if enable_llm_vdb_control is not None:
                    config['enable_llm_vdb_control'] = enable_llm_vdb_control
                    # 保存配置到文件
                    webui_controller.save_config(config)
            
            if vdb_manager:
                updated_settings = vdb_manager.update_db_management_settings(settings)
                return jsonify({
                    'success': True,
                    'message': 'VDB设置更新成功',
                    'settings': updated_settings
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'VDB管理器未初始化'
                }), 500
        elif action == 'trigger_cleanup' or action == 'manual_clean':
            # 手动触发数据库清理
            if vdb_manager:
                result = vdb_manager.trigger_db_management()
                return jsonify(result)
            else:
                return jsonify({
                    'success': False,
                    'message': 'VDB管理器未初始化'
                }), 500
        else:
            return jsonify({'success': False, 'message': f'不支持的操作: {action}'}), 400
            
    except Exception as e:
        logger.error(f"VDB控制失败: {e}")
        return jsonify({'success': False, 'message': f'VDB控制失败: {str(e)}'}), 500

# Socket.IO事件处理
@sio.event
def connect(sid, environ):
    logger.info(f"客户端连接: {sid}")
    # 发送初始数据
    sio.emit('initial_data', {
        'config': config,
        'status': system_status,
        'logs': logs[-100:]
    }, room=sid)

@sio.event
def disconnect(sid):
    logger.info(f"客户端断开连接: {sid}")

@sio.event
def config_updated(sid, data):
    """处理从客户端更新的配置"""
    if webui_controller.save_config(data):
        sio.emit('config_update_success', room=sid)
    else:
        sio.emit('config_update_error', {'message': '配置保存失败'}, room=sid)

@sio.event
def system_command(sid, data):
    """处理系统命令"""
    command = data.get('command')
    result = {'success': False, 'message': '未知命令'}
    
    if command == 'start':
        if system_controller.start():
            result = {'success': True, 'message': '系统启动成功'}
        else:
            result = {'success': False, 'message': '系统启动失败'}
    elif command == 'stop':
        if system_controller.stop():
            result = {'success': True, 'message': '系统停止成功'}
        else:
            result = {'success': False, 'message': '系统停止失败'}
    elif command == 'restart':
        if system_controller.restart():
            result = {'success': True, 'message': '系统重启成功'}
        else:
            result = {'success': False, 'message': '系统重启失败'}
    
    sio.emit('system_command_result', result, room=sid)

# 辅助函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 定时检查和更新系统状态
def status_checker():
    while True:
        time.sleep(5)
        try:
            # 这里可以添加实际的系统状态检查逻辑
            # 例如检查连接状态、资源使用情况等
            pass
        except Exception as e:
            logger.error(f"状态检查失败: {e}")

# 启动状态检查线程
status_thread = threading.Thread(target=status_checker, daemon=True)
status_thread.start()

if __name__ == '__main__':
    logger.info("启动WebUI服务...")
    # 确保templates文件夹存在
    if not os.path.exists('templates'):
        os.makedirs('templates')
        # 创建一个基础的index.html模板
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>晨曦V2 AI直播系统控制面板</title>
    <style>
            body {
                font-family: 'Microsoft YaHei', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
            }
            .panel {
                background-color: white;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .panel h2 {
                color: #4CAF50;
                margin-top: 0;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            .form-group input, .form-group select, .form-group textarea {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            .form-group textarea {
                height: 100px;
                resize: vertical;
            }
            .button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
            }
            .button:hover {
                background-color: #45a049;
            }
            .button.danger {
                background-color: #f44336;
            }
            .button.danger:hover {
                background-color: #da190b;
            }
            .button.warning {
                background-color: #ff9800;
            }
            .button.warning:hover {
                background-color: #e68a00;
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            .status.running {
                background-color: #4CAF50;
                color: white;
            }
            .status.stopped {
                background-color: #f44336;
                color: white;
            }
            .status.restarting {
                background-color: #ff9800;
                color: white;
            }
            .logs {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                background-color: #f9f9f9;
                font-family: monospace;
                font-size: 12px;
            }
            .log-entry {
                margin-bottom: 5px;
                padding: 3px 0;
                border-bottom: 1px solid #eee;
            }
            .log-entry .timestamp {
                color: #666;
                margin-right: 10px;
            }
            .log-entry .level {
                display: inline-block;
                width: 60px;
                text-align: center;
                padding: 2px 5px;
                border-radius: 3px;
                font-size: 10px;
                margin-right: 10px;
            }
            .log-entry .level.info {
                background-color: #2196F3;
                color: white;
            }
            .log-entry .level.warning {
                background-color: #ff9800;
                color: white;
            }
            .log-entry .level.error {
                background-color: #f44336;
                color: white;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 2px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                background-color: #f1f1f1;
                border: 1px solid #ddd;
                border-bottom: none;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
            }
            .tab.active {
                background-color: white;
                border-bottom: 2px solid white;
                margin-bottom: -2px;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            .file-upload {
                border: 2px dashed #ddd;
                padding: 20px;
                text-align: center;
                border-radius: 5px;
                cursor: pointer;
                transition: border-color 0.3s;
            }
            .file-upload:hover {
                border-color: #4CAF50;
            }
            .file-upload input {
                display: none;
            }
            
            /* 开关控件样式 */
            .switch-group {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            .switch-label {
                margin-right: 10px;
                white-space: nowrap;
                font-weight: bold;
                flex: 1;
            }
            .switch {
                position: relative;
                display: inline-block;
                width: 50px;
                height: 24px;
            }
            .switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
                border-radius: 24px;
            }
            .slider:before {
                position: absolute;
                content: "";
                height: 18px;
                width: 18px;
                left: 3px;
                bottom: 3px;
                background-color: white;
                transition: .4s;
                border-radius: 50%;
            }
            input:checked + .slider {
                background-color: #2196F3;
            }
            input:checked + .slider:before {
                transform: translateX(26px);
            }
        </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>晨曦V2 AI直播系统控制面板</h1>
            <p>版本: v2.0 | 实时控制和配置管理</p>
        </div>
        
        <!-- 系统状态 -->
        <div class="panel">
            <h2>系统状态</h2>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p><strong>当前状态:</strong> <span id="system-status" class="status stopped">已停止</span></p>
                    <p><strong>平台:</strong> <span id="platform">未设置</span></p>
                    <p><strong>房间ID:</strong> <span id="room-id">未设置</span></p>
                    <p><strong>连接时间:</strong> <span id="connection-time">无</span></p>
                </div>
                <div>
                    <button id="start-btn" class="button">启动系统</button>
                    <button id="stop-btn" class="button danger">停止系统</button>
                    <button id="restart-btn" class="button warning">重启系统</button>
                </div>
            </div>
        </div>
        
        <!-- 标签页 -->
        <div class="tabs">
            <div class="tab active" data-tab="config">配置管理</div>
            <div class="tab" data-tab="logs">系统日志</div>
            <div class="tab" data-tab="message">消息发送</div>
            <div class="tab" data-tab="file">配置文件</div>
        </div>
        
        <!-- 配置管理 -->
        <div id="config-content" class="tab-content active">
            <div class="grid">
                <!-- 基础配置 -->
                <div class="panel">
                    <h2>基础配置</h2>
                    <div class="form-group">
                        <label for="ollama_api_url">Ollama API URL</label>
                        <input type="text" id="ollama_api_url" placeholder="http://localhost:11434">
                    </div>
                    <div class="form-group">
                        <label for="ollama_model">Ollama 模型</label>
                        <input type="text" id="ollama_model" placeholder="llama3.2">
                    </div>
                    <div class="form-group">
                        <label for="system_prompt">系统提示词</label>
                        <textarea id="system_prompt" placeholder="你是一个AI助手..."></textarea>
                    </div>
                    <div class="form-group">
                        <label for="trigger_key">触发键</label>
                        <input type="text" id="trigger_key" placeholder="t">
                    </div>
                    <div class="form-group">
                        <label for="stop_trigger_key">停止触发键</label>
                        <input type="text" id="stop_trigger_key" placeholder="s">
                    </div>
                </div>
                
                <!-- 记忆和向量数据库配置 -->
                <div class="panel">
                    <h2>记忆与向量数据库配置</h2>
                    <div class="switch-group">
                        <label class="switch-label">启用记忆功能 (启用系统记忆能力):</label>
                        <label class="switch">
                            <input type="checkbox" id="memory_enabled">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div class="form-group">
                        <label for="summarizer_model">摘要模型</label>
                        <input type="text" id="summarizer_model" placeholder="gemma2:2b">
                    </div>
                    <div class="form-group">
                        <label for="embedding_model">嵌入模型</label>
                        <input type="text" id="embedding_model" placeholder="nomic-embed-text">
                    </div>
                    <div class="form-group">
                        <label for="qdrant_host">Qdrant 主机</label>
                        <input type="text" id="qdrant_host" placeholder="localhost">
                    </div>
                    <div class="form-group">
                        <label for="qdrant_port">Qdrant 端口</label>
                        <input type="number" id="qdrant_port" placeholder="6333">
                    </div>
                </div>
                
                <!-- 上下文管理配置 -->
                <div class="panel">
                    <h2>上下文管理</h2>
                    <div class="form-group">
                        <label for="max_context_items">最大上下文项数</label>
                        <input type="number" id="max_context_items" placeholder="50">
                    </div>
                    <div class="form-group">
                        <label for="context_summary_time_range">上下文摘要时间范围(小时)</label>
                        <input type="number" id="context_summary_time_range" placeholder="24">
                    </div>
                    <div class="form-group">
                        <label for="context_summary_interval">上下文摘要间隔(分钟)</label>
                        <input type="number" id="context_summary_interval" placeholder="1440">
                    </div>
                    <div class="form-group">
                        <label for="context_summary_limit">上下文摘要限制(字符)</label>
                        <input type="number" id="context_summary_limit" placeholder="500">
                    </div>
                    <div class="form-group">
                        <label for="max_recent_memories_for_summary">摘要用最大近期记忆数</label>
                        <input type="number" id="max_recent_memories_for_summary" placeholder="300">
                    </div>
                </div>
                
                <!-- VDB管理配置 -->
                <div class="panel">
                    <h2>向量数据库管理</h2>
                    <div class="switch-group">
                        <label class="switch-label">自动运行数据库管理 (定期检查并更新):</label>
                        <label class="switch">
                            <input type="checkbox" id="vdb_auto_run">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div class="switch-group">
                        <label class="switch-label">启动时运行数据库管理 (服务启动时执行):</label>
                        <label class="switch">
                            <input type="checkbox" id="vdb_startup_run">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div class="form-group">
                        <label for="vdb_check_interval">检查间隔(秒)</label>
                        <input type="number" id="vdb_check_interval" placeholder="60">
                    </div>
                </div>
                
                <!-- MeloTTS配置 -->
                <div class="panel">
                    <h2>MeloTTS配置</h2>
                    <div class="switch-group">
                        <label class="switch-label">使用MeloTTS (文本到语音合成):</label>
                        <label class="switch">
                            <input type="checkbox" id="use_molotts">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div class="form-group">
                        <label for="molotts_speaker_id">Speaker ID</label>
                        <input type="number" id="molotts_speaker_id" placeholder="0">
                    </div>
                    <div class="form-group">
                        <label for="molotts_sdp_ratio">SDP Ratio</label>
                        <input type="number" step="0.1" id="molotts_sdp_ratio" placeholder="0.2">
                    </div>
                    <div class="form-group">
                        <label for="molotts_noise_scale">Noise Scale</label>
                        <input type="number" step="0.1" id="molotts_noise_scale" placeholder="0.6">
                    </div>
                    <div class="form-group">
                        <label for="molotts_noise_scale_w">Noise Scale W</label>
                        <input type="number" step="0.1" id="molotts_noise_scale_w" placeholder="0.8">
                    </div>
                    <div class="form-group">
                        <label for="molotts_speed">Speed</label>
                        <input type="number" step="0.1" id="molotts_speed" placeholder="1.0">
                    </div>
                </div>
                
                <!-- 音频设备配置 -->
                <div class="panel">
                    <h2>音频设备配置</h2>
                    <div class="form-group">
                        <label for="audio_input_device">音频输入设备ID</label>
                        <input type="number" id="audio_input_device" placeholder="0">
                    </div>
                    <div class="form-group">
                        <label for="context_file_path">上下文文件路径</label>
                        <input type="text" id="context_file_path" placeholder="context_medium_term.json">
                    </div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <button id="save-config-btn" class="button">保存配置</button>
                <button id="reset-config-btn" class="button warning">重置表单</button>
            </div>
        </div>
        
        <!-- 系统日志 -->
        <div id="logs-content" class="tab-content">
            <div class="panel">
                <h2>系统日志</h2>
                <div class="logs" id="logs-container">
                    <div class="log-entry">
                        <span class="timestamp">2024-01-01 00:00:00</span>
                        <span class="level info">INFO</span>
                        <span class="message">系统启动</span>
                    </div>
                </div>
                <div style="margin-top: 10px; text-align: right;">
                    <button id="clear-logs-btn" class="button warning">清空日志</button>
                </div>
            </div>
        </div>
        
        <!-- 消息发送 -->
        <div id="message-content" class="tab-content">
            <div class="panel">
                <h2>直接消息发送</h2>
                <div class="form-group">
                    <label for="message-input">消息内容</label>
                    <textarea id="message-input" placeholder="输入要发送的消息..."></textarea>
                </div>
                <div style="text-align: center; margin-top: 20px;">
                    <button id="send-message-btn" class="button">发送消息</button>
                </div>
                <div id="message-result" style="margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 4px; display: none;">
                    <h3>回复结果</h3>
                    <div id="reply-content"></div>
                </div>
            </div>
        </div>
        
        <!-- 配置文件管理 -->
        <div id="file-content" class="tab-content">
            <div class="grid">
                <div class="panel">
                    <h2>上传配置文件</h2>
                    <div id="file-upload" class="file-upload">
                        <input type="file" id="config-file" accept=".json">
                        <p>点击或拖拽配置文件到此处上传</p>
                        <p style="font-size: 12px; color: #666;">仅支持 .json 格式文件</p>
                    </div>
                    <div id="upload-status" style="margin-top: 10px; text-align: center; display: none;"></div>
                </div>
                <div class="panel">
                    <h2>下载配置文件</h2>
                    <p>点击下方按钮下载当前配置文件：</p>
                    <div style="text-align: center; margin-top: 20px;">
                        <button id="download-config-btn" class="button">下载配置</button>
                    </div>
                    <p style="margin-top: 20px; font-size: 12px; color: #666;">
                        提示：下载的配置文件可用于备份或在其他实例上导入。
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        // Socket.IO连接
        const socket = io();
        
        // 初始数据
        socket.on('initial_data', function(data) {
            loadConfig(data.config);
            updateStatus(data.status);
            loadLogs(data.logs);
        });
        
        // 状态更新
        socket.on('status_update', function(status) {
            updateStatus(status);
        });
        
        // 日志更新
        socket.on('log_update', function(log) {
            addLogEntry(log);
        });
        
        // 配置更新成功
        socket.on('config_update_success', function() {
            alert('配置保存成功！');
        });
        
        // 配置更新失败
        socket.on('config_update_error', function(data) {
            alert('配置保存失败：' + data.message);
        });
        
        // 系统命令结果
        socket.on('system_command_result', function(result) {
            if (result.success) {
                alert(result.message);
            } else {
                alert('操作失败：' + result.message);
            }
        });
        
        // 加载配置到表单
        function loadConfig(config) {
            // 基础配置
            document.getElementById('ollama_api_url').value = config.ollama_api_url || 'http://localhost:11434/api/chat';
            document.getElementById('ollama_model').value = config.ollama_model || 'llama3.2';
            document.getElementById('system_prompt').value = config.system_prompt || '你是一个AI直播助手。';
            document.getElementById('trigger_key').value = config.trigger_key || 'Alt';
            document.getElementById('stop_trigger_key').value = config.stop_trigger_key || 'Escape';
            
            // 记忆与向量数据库配置
            document.getElementById('memory_enabled').checked = config.memory_enabled !== undefined ? config.memory_enabled : true;
            document.getElementById('summarizer_model').value = config.summarizer_model || 'gemma2:2b';
            document.getElementById('embedding_model').value = config.embedding_model || 'nomic-embed-text';
            document.getElementById('qdrant_host').value = config.qdrant_host || 'localhost';
            document.getElementById('qdrant_port').value = config.qdrant_port || 6333;
            document.getElementById('max_context_items').value = config.max_context_items || 50;
            
            // 上下文管理配置
            document.getElementById('context_summary_time_range').value = config.context_summary_time_range || 24;
            document.getElementById('context_summary_interval').value = config.context_summary_interval || 1440;
            document.getElementById('context_summary_limit').value = config.context_summary_limit || 500;
            document.getElementById('max_recent_memories_for_summary').value = config.max_recent_memories_for_summary || 300;
            document.getElementById('context_file_path').value = config.context_file_path || 'context_history.json';
            
            // 向量数据库管理配置
            document.getElementById('vdb_auto_run').checked = config.vdb_auto_run !== undefined ? config.vdb_auto_run : true;
            document.getElementById('vdb_startup_run').checked = config.vdb_startup_run !== undefined ? config.vdb_startup_run : true;
            document.getElementById('vdb_check_interval').value = config.vdb_check_interval || 60;
            
            // MeloTTS配置
            document.getElementById('use_molotts').checked = config.use_molotts !== undefined ? config.use_molotts : false;
            document.getElementById('molotts_speaker_id').value = config.molotts_speaker_id || 'zh_female_1';
            document.getElementById('molotts_sdp_ratio').value = config.molotts_sdp_ratio || 0.2;
            document.getElementById('molotts_noise_scale').value = config.molotts_noise_scale || 0.667;
            document.getElementById('molotts_noise_scale_w').value = config.molotts_noise_scale_w || 0.8;
            document.getElementById('molotts_speed').value = config.molotts_speed || 1.0;
            
            // 音频设备配置
            document.getElementById('audio_input_device').value = config.audio_input_device || '';
            
            // 功能开关配置 - 添加安全检查
            if (document.getElementById('enable_sound_effects')) document.getElementById('enable_sound_effects').checked = config.enable_sound_effects !== undefined ? config.enable_sound_effects : true;
            if (document.getElementById('enable_music')) document.getElementById('enable_music').checked = config.enable_music !== undefined ? config.enable_music : true;
            if (document.getElementById('enable_ai_drawing')) document.getElementById('enable_ai_drawing').checked = config.enable_ai_drawing !== undefined ? config.enable_ai_drawing : true;
            if (document.getElementById('enable_screen_click')) document.getElementById('enable_screen_click').checked = config.enable_screen_click !== undefined ? config.enable_screen_click : true;
            if (document.getElementById('enable_continuous_talk')) document.getElementById('enable_continuous_talk').checked = config.enable_continuous_talk !== undefined ? config.enable_continuous_talk : false;
            if (document.getElementById('enable_wake_sleep')) document.getElementById('enable_wake_sleep').checked = config.enable_wake_sleep !== undefined ? config.enable_wake_sleep : false;
            
            // SenseVoice配置 - 添加安全检查
            if (document.getElementById('sense_voice_api_url')) document.getElementById('sense_voice_api_url').value = config.sense_voice_api_url || 'http://127.0.0.1:8877/api/v1/asr';
            if (document.getElementById('sense_voice_api_key')) document.getElementById('sense_voice_api_key').value = config.sense_voice_api_key || '';
        }
        
        // 更新系统状态显示
        function updateStatus(status) {
            const statusEl = document.getElementById('system-status');
            statusEl.textContent = status.status === 'running' ? '运行中' : status.status === 'stopped' ? '已停止' : '重启中';
            statusEl.className = 'status ' + status.status;
            
            document.getElementById('platform').textContent = status.platform || '未设置';
            document.getElementById('room-id').textContent = status.room_id || '未设置';
            document.getElementById('connection-time').textContent = status.connection_time || '无';
        }
        
        // 加载日志
        function loadLogs(logs) {
            const logsContainer = document.getElementById('logs-container');
            logsContainer.innerHTML = '';
            logs.forEach(log => {
                addLogEntry(log);
            });
            // 滚动到底部
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        // 添加日志条目
        function addLogEntry(log) {
            const logsContainer = document.getElementById('logs-container');
            const logEl = document.createElement('div');
            logEl.className = 'log-entry';
            logEl.innerHTML = `
                <span class="timestamp">${log.timestamp}</span>
                <span class="level ${log.level}">${log.level.toUpperCase()}</span>
                <span class="message">${log.message}</span>
            `;
            logsContainer.appendChild(logEl);
            // 滚动到底部
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        // 标签页切换
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', function() {
                // 移除所有活跃状态
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // 添加活跃状态
                this.classList.add('active');
                const tabId = this.getAttribute('data-tab');
                document.getElementById(tabId + '-content').classList.add('active');
            });
        });
        
        // 保存配置
        document.getElementById('save-config-btn').addEventListener('click', function() {
            // 构建完整的配置对象，包含所有功能开关
            const newConfig = {
                // 基础配置
                ollama_api_url: document.getElementById('ollama_api_url').value || 'http://localhost:11434/api/chat',
                ollama_model: document.getElementById('ollama_model').value || 'llama3.2',
                system_prompt: document.getElementById('system_prompt').value || '你是一个AI直播助手。',
                trigger_key: document.getElementById('trigger_key').value || 'Alt',
                stop_trigger_key: document.getElementById('stop_trigger_key').value || 'Escape',
                
                // 记忆与向量数据库配置
                memory_enabled: document.getElementById('memory_enabled').checked,
                summarizer_model: document.getElementById('summarizer_model').value || 'gemma2:2b',
                embedding_model: document.getElementById('embedding_model').value || 'nomic-embed-text',
                qdrant_host: document.getElementById('qdrant_host').value || 'localhost',
                qdrant_port: parseInt(document.getElementById('qdrant_port').value) || 6333,
                max_context_items: parseInt(document.getElementById('max_context_items').value) || 50,
                
                // 上下文管理配置
                context_summary_time_range: parseInt(document.getElementById('context_summary_time_range').value) || 24,
                context_summary_interval: parseInt(document.getElementById('context_summary_interval').value) || 1440,
                context_summary_limit: parseInt(document.getElementById('context_summary_limit').value) || 500,
                max_recent_memories_for_summary: parseInt(document.getElementById('max_recent_memories_for_summary').value) || 300,
                context_file_path: document.getElementById('context_file_path').value || 'context_history.json',
                
                // 向量数据库管理配置
                vdb_auto_run: document.getElementById('vdb_auto_run').checked,
                vdb_startup_run: document.getElementById('vdb_startup_run').checked,
                vdb_check_interval: parseInt(document.getElementById('vdb_check_interval').value) || 60,
                
                // MeloTTS配置
                use_molotts: document.getElementById('use_molotts').checked,
                molotts_speaker_id: document.getElementById('molotts_speaker_id').value || 'zh_female_1',
                molotts_sdp_ratio: parseFloat(document.getElementById('molotts_sdp_ratio').value) || 0.2,
                molotts_noise_scale: parseFloat(document.getElementById('molotts_noise_scale').value) || 0.667,
                molotts_noise_scale_w: parseFloat(document.getElementById('molotts_noise_scale_w').value) || 0.8,
                molotts_speed: parseFloat(document.getElementById('molotts_speed').value) || 1.0,
                
                // 音频设备配置
                audio_input_device: document.getElementById('audio_input_device').value || '',
                
                // 功能开关配置
                enable_sound_effects: document.getElementById('enable_sound_effects')?.checked || true,
                enable_music: document.getElementById('enable_music')?.checked || true,
                enable_ai_drawing: document.getElementById('enable_ai_drawing')?.checked || true,
                enable_screen_click: document.getElementById('enable_screen_click')?.checked || true,
                enable_continuous_talk: document.getElementById('enable_continuous_talk')?.checked || false,
                enable_wake_sleep: document.getElementById('enable_wake_sleep')?.checked || false,
                
                // SenseVoice配置
                sense_voice_api_url: document.getElementById('sense_voice_api_url')?.value || 'http://127.0.0.1:8877/api/v1/asr',
                sense_voice_api_key: document.getElementById('sense_voice_api_key')?.value || ''
            };
            
            // 添加错误处理，确保数值类型正确
            Object.keys(newConfig).forEach(key => {
                if (typeof newConfig[key] === 'number' && isNaN(newConfig[key])) {
                    // 设置为默认值
                    switch(key) {
                        case 'qdrant_port': newConfig[key] = 6333; break;
                        case 'max_context_items': newConfig[key] = 50; break;
                        case 'context_summary_time_range': newConfig[key] = 24; break;
                        case 'context_summary_interval': newConfig[key] = 1440; break;
                        case 'context_summary_limit': newConfig[key] = 500; break;
                        case 'max_recent_memories_for_summary': newConfig[key] = 300; break;
                        case 'vdb_check_interval': newConfig[key] = 60; break;
                        case 'molotts_sdp_ratio': newConfig[key] = 0.2; break;
                        case 'molotts_noise_scale': newConfig[key] = 0.667; break;
                        case 'molotts_noise_scale_w': newConfig[key] = 0.8; break;
                        case 'molotts_speed': newConfig[key] = 1.0; break;
                    }
                }
            });
            
            socket.emit('config_updated', newConfig);
        });
        
        // 重置配置表单
        document.getElementById('reset-config-btn').addEventListener('click', function() {
            fetch('/api/config')
                .then(response => response.json())
                .then(config => {
                    loadConfig(config);
                })
                .catch(error => {
                    console.error('重置配置失败:', error);
                });
        });
        
        // 系统控制按钮
        document.getElementById('start-btn').addEventListener('click', function() {
            socket.emit('system_command', { command: 'start' });
        });
        
        document.getElementById('stop-btn').addEventListener('click', function() {
            socket.emit('system_command', { command: 'stop' });
        });
        
        document.getElementById('restart-btn').addEventListener('click', function() {
            socket.emit('system_command', { command: 'restart' });
        });
        
        // 清空日志
        document.getElementById('clear-logs-btn').addEventListener('click', function() {
            document.getElementById('logs-container').innerHTML = '';
        });
        
        // 发送消息
        document.getElementById('send-message-btn').addEventListener('click', function() {
            const message = document.getElementById('message-input').value.trim();
            if (!message) {
                alert('请输入消息内容');
                return;
            }
            
            fetch('/api/send_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const resultDiv = document.getElementById('message-result');
                    document.getElementById('reply-content').textContent = data.reply;
                    resultDiv.style.display = 'block';
                } else {
                    alert('发送失败：' + data.message);
                }
            })
            .catch(error => {
                console.error('发送消息失败:', error);
                alert('发送失败，请检查网络连接');
            });
        });
        
        // 文件上传
        const fileUpload = document.getElementById('file-upload');
        const configFile = document.getElementById('config-file');
        const uploadStatus = document.getElementById('upload-status');
        
        fileUpload.addEventListener('click', function() {
            configFile.click();
        });
        
        fileUpload.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#4CAF50';
        });
        
        fileUpload.addEventListener('dragleave', function() {
            this.style.borderColor = '#ddd';
        });
        
        fileUpload.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#ddd';
            
            if (e.dataTransfer.files.length) {
                configFile.files = e.dataTransfer.files;
                handleFileUpload();
            }
        });
        
        configFile.addEventListener('change', handleFileUpload);
        
        function handleFileUpload() {
            const file = configFile.files[0];
            if (!file) return;
            
            if (!file.name.endsWith('.json')) {
                uploadStatus.textContent = '错误：请上传.json格式文件';
                uploadStatus.style.color = '#f44336';
                uploadStatus.style.display = 'block';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            uploadStatus.textContent = '上传中...';
            uploadStatus.style.color = '#2196F3';
            uploadStatus.style.display = 'block';
            
            fetch('/api/upload_config', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    uploadStatus.textContent = '上传成功！配置已应用';
                    uploadStatus.style.color = '#4CAF50';
                    // 重新加载配置
                    fetch('/api/config')
                        .then(response => response.json())
                        .then(config => {
                            loadConfig(config);
                        });
                } else {
                    uploadStatus.textContent = '上传失败：' + data.message;
                    uploadStatus.style.color = '#f44336';
                }
            })
            .catch(error => {
                uploadStatus.textContent = '上传失败，请检查网络连接';
                uploadStatus.style.color = '#f44336';
            });
        }
        
        // 下载配置
        document.getElementById('download-config-btn').addEventListener('click', function() {
            window.location.href = '/api/download_config';
        });
    </script>
</body>
</html>''')
    
# 供外部调用的WebUI启动函数
def run_webui(host='0.0.0.0', port=5000):
    """启动WebUI服务
    
    Args:
        host: 主机地址，默认为'0.0.0.0'
        port: 端口号，默认为5000
    """
    try:
        logger.info(f"WebUI服务启动在 http://{host}:{port}")
        # 检查是否成功导入了eventlet
        if 'eventlet' in sys.modules:
            # 使用eventlet服务器启动Flask应用，这样才能正确支持WebSocket
            import eventlet
            eventlet.wsgi.server(eventlet.listen((host, port)), app)
        else:
            # 如果eventlet不可用，回退到标准的Flask服务器（可能无法正常使用WebSocket功能）
            logger.warning("eventlet模块不可用，将使用标准Flask服务器启动，WebSocket功能可能无法正常工作")
            app.run(host=host, port=port, debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"启动WebUI服务失败: {e}")
        sys.exit(1)

# 启动服务器 - 仅当直接运行webui.py时执行
if __name__ == "__main__":
    run_webui(port=5001)  # 使用5001端口以避免冲突