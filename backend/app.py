"""
衣鱼 (Silverfish) - 小说人物关系解析后端
Flask API 服务
"""
import os
import sys
import uuid
import logging
import threading
import time
from datetime import datetime

# 确保能导入同目录模块
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# 加载 .env（用绝对路径，避免 Flask reloader 下 __file__ 解析异常）
_ENV_PATH = os.path.join(os.path.dirname(_HERE), '.env')
load_dotenv(_ENV_PATH)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('silverfish')

app = Flask(__name__)
# 在读取请求体前限制上传体积，避免异常大文件占满内存。
app.config['MAX_CONTENT_LENGTH'] = max(
    1024 * 1024,
    int(os.getenv('MAX_UPLOAD_BYTES', str(12 * 1024 * 1024)))
)
CORS(app)

# 会话存储（单机发布版使用内存；多进程部署应换成 Redis）
sessions = {}
sessions_lock = threading.RLock()
SESSION_TTL_SECONDS = max(60, int(os.getenv('SESSION_TTL_SECONDS', '3600')))
SESSION_STALE_SECONDS = max(SESSION_TTL_SECONDS, int(os.getenv('SESSION_STALE_SECONDS', '86400')))
SESSION_CLEANUP_INTERVAL = max(10, int(os.getenv('SESSION_CLEANUP_INTERVAL', '300')))


def _cleanup_expired_sessions(now=None):
    """删除已结束的过期会话，以及异常滞留的超时会话。"""
    now = now or time.time()
    terminal_states = {'completed', 'failed'}
    with sessions_lock:
        expired_ids = [
            session_id
            for session_id, session in sessions.items()
            if (
                session['status'] in terminal_states
                and now - session['updated_at'] >= SESSION_TTL_SECONDS
            ) or now - session['created_at'] >= SESSION_STALE_SECONDS
        ]
        for session_id in expired_ids:
            sessions.pop(session_id, None)
    if expired_ids:
        logger.info(f"已清理 {len(expired_ids)} 个过期分析会话")
    return len(expired_ids)


def _session_cleanup_worker():
    while True:
        time.sleep(SESSION_CLEANUP_INTERVAL)
        try:
            _cleanup_expired_sessions()
        except Exception:
            logger.exception("清理过期会话失败")


threading.Thread(
    target=_session_cleanup_worker,
    name='session-cleanup',
    daemon=True,
).start()


def _friendly_analysis_error(exc):
    """把内部异常映射成可操作、但不泄露服务细节的用户提示。"""
    message = str(exc).lower()
    if 'chunk_overlap' in message or 'chunk_size' in message:
        return '文本分块配置无效：请确保 0 <= CHUNK_OVERLAP < CHUNK_SIZE。'
    if any(token in message for token in ('api key', 'authentication', 'unauthorized', '401')):
        return 'LLM 认证失败，请检查 API Key 配置。'
    if any(token in message for token in ('rate limit', '频率', '限流', '429')):
        return 'LLM 请求过于频繁，请稍后重试或降低并发数。'
    if any(token in message for token in ('timeout', 'timed out', '超时')):
        return 'LLM 请求超时，请检查网络后重试。'
    return '分析失败，请稍后重试；详细原因已记录在后端日志中。'

# ==================== 路由 ====================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    启动文本分析。支持：
    - 文件上传 multipart/form-data (field: file, .txt)
    - JSON {"text": "..."}
    """
    try:
        # 检查 API Key
        api_key = os.getenv('LLM_API_KEY', '')
        if not api_key or api_key == '你的_API_KEY':
            return jsonify({
                'success': False,
                'error': 'LLM_API_KEY 未配置。请在项目根目录创建 .env 文件。'
            }), 400

        text_content = ''
        source_name = '粘贴文本'

        # 处理文件上传
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and file.filename.lower().endswith(('.txt', '.md')):
                source_name = os.path.basename(file.filename)
                content_bytes = file.read()
                # 多编码尝试
                for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin-1']:
                    try:
                        text_content = content_bytes.decode(enc)
                        logger.info(f"使用 {enc} 编码读取上传文件")
                        break
                    except UnicodeDecodeError:
                        continue
                if not text_content:
                    text_content = content_bytes.decode('utf-8', errors='ignore')
                logger.info(f"文件读取完成, 长度: {len(text_content)}")
            else:
                return jsonify({'success': False, 'error': '仅支持 .txt 或 .md 文本文件'}), 400

        # 处理 JSON 文本
        elif request.is_json:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({'success': False, 'error': 'JSON 请求必须是包含 text 字段的对象'}), 400
            text_content = data.get('text', '')
            source_name = str(data.get('title') or source_name)[:120]

        if not isinstance(text_content, str):
            return jsonify({'success': False, 'error': 'text 字段必须是字符串'}), 400

        if not text_content or len(text_content.strip()) == 0:
            return jsonify({'success': False, 'error': '文本内容不能为空'}), 400

        if len(text_content) > 3000000:
            return jsonify({'success': False, 'error': '文本过长，目前仅支持 300 万字以内'}), 400

        # 创建会话
        session_id = str(uuid.uuid4())[:8]
        now = time.time()
        with sessions_lock:
            sessions[session_id] = {
                'status': 'processing',
                'progress': 0,
                'message': '开始分析...',
                'result': None,
                'error': None,
                'created_at': now,
                'updated_at': now,
                'created_at_iso': datetime.now().isoformat(),
                'source_name': source_name,
            }

        # 异步启动分析
        thread = threading.Thread(
            target=_run_analysis,
            args=(session_id, text_content, source_name),
            daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'session_id': session_id,
            'status_url': f'/api/status/{session_id}'
        })

    except ValueError as ve:
        logger.error(f"配置错误: {ve}", exc_info=True)
        return jsonify({'success': False, 'error': '请求配置无效，请检查输入和 LLM 配置。'}), 400
    except Exception as e:
        logger.error(f"分析请求失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '服务器处理请求失败，请稍后重试。'}), 500


@app.route('/api/status/<session_id>', methods=['GET'])
def get_status(session_id):
    """查询分析进度"""
    with sessions_lock:
        session = sessions.get(session_id)
        session = dict(session) if session else None
    if not session:
        return jsonify({'success': False, 'error': '会话不存在'}), 404

    return jsonify({
        'success': True,
        'session_id': session_id,
        'status': session['status'],
        'progress': session['progress'],
        'message': session['message'],
        'result': session['result'],
        'error': session['error']
    })


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    api_key = os.getenv('LLM_API_KEY', '')
    return jsonify({
        'success': True,
        'status': 'running',
        'llm_configured': bool(api_key) and api_key != '你的_API_KEY'
    })


# ==================== 分析逻辑 ====================

def _run_analysis(session_id, text, source_name='粘贴文本'):
    """在后台线程中运行完整分析"""
    with sessions_lock:
        session = sessions.get(session_id)
    if not session:
        logger.warning(f"会话 {session_id} 在分析开始前已不存在")
        return

    def on_progress(pct, msg):
        with sessions_lock:
            session['progress'] = pct
            session['message'] = msg
            session['updated_at'] = time.time()
            if pct < 90:
                session['status'] = 'processing'
            elif pct < 100:
                session['status'] = 'aggregating'

    try:
        # 延迟导入，避免启动时 LLM 初始化失败
        from extractor import RelationshipExtractor

        extractor = RelationshipExtractor()
        result = extractor.analyze(text, on_progress=on_progress)
        result.setdefault('overview', {})['source_name'] = source_name

        with sessions_lock:
            session['status'] = 'completed'
            session['progress'] = 100
            session['message'] = '分析完成'
            session['result'] = result
            session['updated_at'] = time.time()

        logger.info(f"会话 {session_id} 分析完成: {len(result['nodes'])} 人物, {len(result['links'])} 关系")

    except Exception as e:
        logger.error(f"会话 {session_id} 分析失败: {e}", exc_info=True)
        friendly_error = _friendly_analysis_error(e)
        with sessions_lock:
            session['status'] = 'failed'
            session['error'] = friendly_error
            session['message'] = friendly_error
            session['updated_at'] = time.time()


# ==================== 启动 ====================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Novel Constellation 后端服务启动")
    logger.info(f"LLM: {os.getenv('LLM_BASE_URL', 'N/A')} / {os.getenv('LLM_MODEL_NAME', 'N/A')}")
    api_key = os.getenv('LLM_API_KEY', '')
    logger.info(f"API Key 已配置: {bool(api_key) and api_key != '你的_API_KEY'}")
    logger.info("=" * 50)
    # 关闭 reloader，避免 debug 模式下产生孤儿进程占用端口
    # 本地发布包不启用 Flask 调试器，避免暴露调试页面和额外开销。
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
