"""
Steam 数据采集系统 - Flask 后端 API
提供游戏数据采集、评论收集、模型训练、数据清洗的 RESTful API
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import sys
import os
import threading
import uuid
from datetime import datetime
import traceback

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 初始化 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'steam-data-collector-secret-key'
CORS(app)  # 允许跨域请求
socketio = SocketIO(app, cors_allowed_origins="*")

# 导入路由蓝图
from routes.collection import collection_bp
from routes.reviews import reviews_bp
from routes.training import training_bp
from routes.cleaning import cleaning_bp

# 全局任务存储
tasks = {}
task_lock = threading.Lock()


class TaskManager:
    """任务管理器 - 管理后台异步任务"""
    
    def __init__(self):
        self.tasks = {}
    
    def create_task(self, task_type: str) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            'id': task_id,
            'type': task_type,
            'status': 'pending',
            'progress': 0,
            'message': '任务已创建',
            'created_at': datetime.now().isoformat(),
            'logs': [],
            'result': None,
            'error': None
        }
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
            # 通过 WebSocket 推送更新
            socketio.emit('task_update', self.tasks[task_id], room=task_id)
    
    def add_log(self, task_id: str, message: str, level: str = 'info'):
        """添加任务日志"""
        if task_id in self.tasks:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message
            }
            self.tasks[task_id]['logs'].append(log_entry)
            socketio.emit('task_log', log_entry, room=task_id)
    
    def get_task(self, task_id: str):
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def delete_task(self, task_id: str):
        """删除任务"""
        if task_id in self.tasks:
            del self.tasks[task_id]


# 初始化任务管理器
task_manager = TaskManager()


# ============================================================================
# WebSocket 事件处理
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print(f'客户端已连接: {request.sid}')


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    print(f'客户端已断开: {request.sid}')


@socketio.on('join_task')
def handle_join_task(data):
    """加入任务房间以接收实时更新"""
    task_id = data.get('task_id')
    if task_id:
        from flask_socketio import join_room
        join_room(task_id)
        emit('joined', {'task_id': task_id})


# ============================================================================
# API 路由 - 健康检查
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'message': 'Steam Data Collector API is running',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task)


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def cancel_task(task_id):
    """取消任务"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    
    task_manager.update_task(task_id, status='cancelled', message='任务已取消')
    return jsonify({'message': '任务已取消'})


# ============================================================================
# 错误处理
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API 端点不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


# ============================================================================
# 注册蓝图
# ============================================================================

app.register_blueprint(collection_bp)
app.register_blueprint(reviews_bp)
app.register_blueprint(training_bp)
app.register_blueprint(cleaning_bp)


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    print('=' * 80)
    print('Steam 数据采集系统 - 后端 API 服务')
    print('=' * 80)
    print(f'启动时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'监听地址: http://localhost:5000')
    print(f'WebSocket: ws://localhost:5000')
    print('API 端点:')
    print('  - GET  /api/health                     健康检查')
    print('  - POST /api/collect/start              开始游戏数据采集')
    print('  - POST /api/reviews/start              开始评论采集')
    print('  - POST /api/train/start                开始模型训练')
    print('  - POST /api/clean/start                开始数据清洗')
    print('=' * 80)
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
