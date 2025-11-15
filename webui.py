#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebUI for 晨曦V2 AI直播系统
使用 Flask 创建 Web 界面来控制 晨曦V2 AI直播系统
"""
import json
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import time
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebUIController:
    """WebUI 控制器，管理与主应用的交互"""

    def __init__(self, ai_system=None):
        self.config_file = 'config/config.json'
        self.ai_system = ai_system  # 添加对AI系统实例的引用
        self.status = {
            'running': False,
            'last_update': datetime.now().isoformat(),
            'log_messages': []
        }
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            # 确保上下文文件路径设置为中期上下文文件
            if 'context_file_path' not in self.config or self.config['context_file_path'] == 'context_history.json':
                self.config['context_file_path'] = 'context_medium_term.json'

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}

    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def update_status(self, running_status):
        """更新运行状态"""
        self.status['running'] = running_status
        self.status['last_update'] = datetime.now().isoformat()

    def add_log_message(self, message, level='INFO'):
        """添加日志消息"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.status['log_messages'].append(log_entry)

        # 限制日志数量，保留最新的100条
        if len(self.status['log_messages']) > 100:
            self.status['log_messages'] = self.status['log_messages'][-100:]

        # 通过 WebSocket 发送日志更新
        socketio.emit('log_update', log_entry)

# 创建控制器实例
controller = WebUIController()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def config_api():
    """配置 API 端点"""
    if request.method == 'GET':
        return jsonify(controller.config)
    elif request.method == 'POST':
        try:
            new_config = request.json
            # 确保上下文文件路径设置为中期上下文文件
            if 'context_file_path' not in new_config or new_config['context_file_path'] == 'context_history.json':
                new_config['context_file_path'] = 'context_medium_term.json'

            controller.config.update(new_config)
            if controller.save_config():
                socketio.emit('config_updated', controller.config)
                return jsonify({'success': True, 'message': '配置已保存'})
            else:
                return jsonify({'success': False, 'message': '保存配置失败'})
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status', methods=['GET'])
def status_api():
    """状态 API 端点"""
    return jsonify(controller.status)

@app.route('/api/control', methods=['POST'])
def control_api():
    """控制 API 端点"""
    try:
        data = request.json
        action = data.get('action')

        if action == 'start':
            # 如果有 AI 系统实例，尝试启动它
            if controller.ai_system:
                try:
                    if not controller.ai_system.running:
                        controller.ai_system.start()
                        controller.update_status(True)
                        controller.add_log_message('AI 系统已启动', 'INFO')
                        socketio.emit('status_update', controller.status)
                        return jsonify({'success': True, 'message': 'AI 系统已启动'})
                    else:
                        return jsonify({'success': False, 'message': 'AI 系统已在运行'})
                except Exception as e:
                    controller.add_log_message(f'启动 AI 系统失败: {str(e)}', 'ERROR')
                    return jsonify({'success': False, 'message': f'启动失败: {str(e)}'})
            else:
                # 没有 AI 系统实例，仅更新状态
                controller.update_status(True)
                controller.add_log_message('应用程序状态已更新为运行', 'INFO')
                socketio.emit('status_update', controller.status)
                return jsonify({'success': True, 'message': '状态已更新为运行（无AI系统实例）'})

        elif action == 'stop':
            # 如果有 AI 系统实例，尝试停止它
            if controller.ai_system:
                try:
                    if controller.ai_system.running:
                        controller.ai_system.stop()
                        controller.update_status(False)
                        controller.add_log_message('AI 系统已停止', 'INFO')
                        socketio.emit('status_update', controller.status)
                        return jsonify({'success': True, 'message': 'AI 系统已停止'})
                    else:
                        return jsonify({'success': False, 'message': 'AI 系统未在运行'})
                except Exception as e:
                    controller.add_log_message(f'停止 AI 系统失败: {str(e)}', 'ERROR')
                    return jsonify({'success': False, 'message': f'停止失败: {str(e)}'})
            else:
                # 没有 AI 系统实例，仅更新状态
                controller.update_status(False)
                controller.add_log_message('应用程序状态已更新为停止', 'INFO')
                socketio.emit('status_update', controller.status)
                return jsonify({'success': True, 'message': '状态已更新为停止（无AI系统实例）'})

        elif action == 'restart':
            # 重启 AI 系统
            if controller.ai_system:
                try:
                    if controller.ai_system.running:
                        # 先停止系统
                        controller.ai_system.stop()
                        time.sleep(1)  # 等待1秒确保系统完全停止

                    # 然后重新启动系统
                    controller.ai_system.start()
                    controller.update_status(True)
                    controller.add_log_message('AI 系统已重启', 'INFO')
                    socketio.emit('status_update', controller.status)
                    return jsonify({'success': True, 'message': 'AI 系统已重启'})
                except Exception as e:
                    controller.add_log_message(f'重启 AI 系统失败: {str(e)}', 'ERROR')
                    return jsonify({'success': False, 'message': f'重启失败: {str(e)}'})
            else:
                return jsonify({'success': False, 'message': '无AI系统实例可重启'})

        elif action == 'update_key_config':
            # 更新按键配置
            if controller.ai_system:
                try:
                    # 获取新的按键配置
                    new_trigger_key = data.get('trigger_key')
                    new_stop_trigger_key = data.get('stop_trigger_key')

                    if new_trigger_key:
                        controller.config['trigger_key'] = new_trigger_key
                        controller.ai_system.config['trigger_key'] = new_trigger_key
                    if new_stop_trigger_key:
                        controller.config['stop_trigger_key'] = new_stop_trigger_key
                        controller.ai_system.config['stop_trigger_key'] = new_stop_trigger_key

                    # 保存配置到文件
                    controller.save_config()

                    controller.add_log_message(f'按键配置已更新: 触发键={new_trigger_key}, 停止键={new_stop_trigger_key}', 'INFO')
                    socketio.emit('config_updated', controller.config)
                    return jsonify({'success': True, 'message': '按键配置已更新'})
                except Exception as e:
                    controller.add_log_message(f'更新按键配置失败: {str(e)}', 'ERROR')
                    return jsonify({'success': False, 'message': f'更新按键配置失败: {str(e)}'})
            else:
                return jsonify({'success': False, 'message': '无AI系统实例'})

        elif action == 'send_message':
            # 向 LLM 发送消息
            if controller.ai_system:
                try:
                    message = data.get('message')
                    if not message:
                        return jsonify({'success': False, 'message': '消息内容不能为空'})

                    # 使用系统方法发送消息（绕过ASR）
                    controller.ai_system.add_manual_message(message)

                    controller.add_log_message(f'直接发送消息: {message[:50]}...', 'INFO')
                    return jsonify({'success': True, 'message': '消息已发送'})
                except Exception as e:
                    controller.add_log_message(f'发送消息失败: {str(e)}', 'ERROR')
                    return jsonify({'success': False, 'message': f'发送消息失败: {str(e)}'})
            else:
                return jsonify({'success': False, 'message': 'AI系统不可用'})

        else:
            return jsonify({'success': False, 'message': f'未知操作: {action}'})

    except Exception as e:
        logger.error(f"Error in control API: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/logs')
def logs_api():
    """日志 API 端点"""
    return jsonify(controller.status['log_messages'])

@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    logger.info('WebUI client connected')
    emit('config_update', controller.config)
    emit('status_update', controller.status)
    # 发送最近的日志
    for log_entry in controller.status['log_messages'][-20:]:  # 发送最新的20条日志
        emit('log_update', log_entry)

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    logger.info('WebUI client disconnected')

# 创建模板目录和静态目录
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

if not os.path.exists(template_dir):
    os.makedirs(template_dir)

if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# We no longer need to embed the HTML template since we created a separate file

def run_webui(host='0.0.0.0', port=5000):
    """运行 WebUI"""
    logger.info(f"Starting WebUI on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)

if __name__ == '__main__':
    # 启动 WebUI 服务器
    run_webui()