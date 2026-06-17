from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import re
import json
import hashlib
import time
import logging
from logging.handlers import RotatingFileHandler

from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from tools.tool_registry import get_tools

# 配置日志
LOG_FILE = "logs/app.log"
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("med_agent")
logger.setLevel(logging.INFO)

# 控制台输出（保留）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 文件输出（带轮转）
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5           # 保留5个备份
)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

app = FastAPI(title="医疗Agent API", description="基于RAG的多工具医学问答助手", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# Init Agent Components
# =========================
TOOLS = get_tools()
planner = Planner()
executor = Executor(TOOLS)
synthesizer = Synthesizer(api_key=os.getenv("DASHSCOPE_API_KEY"))

class ChatRequest(BaseModel):
    question: str
    history: list = []

# =========================
# 会话状态管理（多候选人选择）
# =========================
pending_selection = {}  # {session_id: {"candidates": [...], "action": "generate_report"}}
current_patient = {}    # {session_id: "张三"}

def get_session_id(request: Request) -> str:
    """生成会话ID（基于IP + User-Agent）"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("User-Agent", "")[:30]
    return f"{client_ip}_{user_agent}"

def set_current_patient(session_id: str, name: str):
    """设置当前会话的操作患者"""
    current_patient[session_id] = name

def get_current_patient(session_id: str) -> str:
    """获取当前会话的操作患者"""
    return current_patient.get(session_id)

def clear_current_patient(session_id: str):
    """清除当前会话的操作患者"""
    if session_id in current_patient:
        del current_patient[session_id]

# =========================
# 简单内存缓存（带 TTL）
# =========================
class SimpleCache:
    def __init__(self):
        self._cache = {}

    def get(self, key):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['time'] < entry['ttl']:
                return entry['value']
            else:
                del self._cache[key]
        return None

    def set(self, key, value, ttl=3600):
        self._cache[key] = {'value': value, 'time': time.time(), 'ttl': ttl}

    def clear(self):
        self._cache.clear()

cache = SimpleCache()

def should_cache(question: str, history: list) -> bool:
    """判断当前问题是否适合缓存"""
    pronouns = ["它", "他", "她", "这个", "那个", "刚才", "上述", "以上", "以下"]
    if any(word in question for word in pronouns):
        logger.info(f"[Cache] 包含指代词，不缓存: {question}")
        return False
    if not history or len(history) <= 1:
        return True
    return False

def get_cache_key(question: str) -> str:
    """缓存 key 只基于问题本身"""
    return hashlib.md5(question.encode()).hexdigest()

# =========================
# 核心问答逻辑（独立函数，用于缓存）
# =========================
async def process_question(question: str, history: list) -> dict:
    """执行完整的问答流程，返回最终响应字典"""
    stripped = question.strip()
    greetings = ["你好", "您好", "hi", "hello", "在吗", "在不在", "你好呀"]
    if stripped in greetings or stripped.lower() in greetings:
        return {
            "success": True,
            "result": {
                "answer": "您好！我是医学问答助手，您可以向我咨询药物、指南、文献或风险相关问题。",
                "tools_used": [],
                "plan": {"question": question, "tools": []},
                "tool_results": []
            },
            "trace": {"executor": []}
        }

    # 纯中文姓名预检
    if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', stripped):
        from tools.tool_registry import get_tools
        tools = get_tools()
        patient_tool = tools.get("patient")
        if patient_tool:
            info = patient_tool.recall(stripped)
            if info:
                return {
                    "success": True,
                    "result": {
                        "answer": f"📋 患者 {stripped} 的档案：\n{info}",
                        "tools_used": ["patient"],
                        "plan": {"question": question, "tools": ["patient"]},
                        "tool_results": []
                    },
                    "trace": {"executor": []}
                }

    # 构建上下文（历史 + 患者档案预加载）
    if history:
        recent = history[-5:]
        filtered = []
        for msg in recent:
            if msg["role"] == "user" and msg["content"] in ["你好", "您好", "hi", "hello"]:
                continue
            filtered.append(msg)
        if filtered:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in filtered])
            augmented_question = f"对话历史：\n{history_str}\n当前问题：{question}"
        else:
            augmented_question = question
    else:
        augmented_question = question

    # 患者档案预加载
    patient_context = ""
    if "患者" in question and "记住" not in question and "记录" not in question and "追加" not in question and "补充" not in question:
        name_match = re.search(r'患者\s*([\u4e00-\u9fa5]{2,4})', question)
        if name_match:
            name = name_match.group(1)
            from tools.tool_registry import get_tools
            tools = get_tools()
            patient_tool = tools.get("patient")
            if patient_tool:
                info = patient_tool.recall(name)
                if info:
                    patient_context = f"【患者档案】姓名：{name}，{info}\n"
                    logger.info(f"[Patient] 已加载患者 {name} 档案")
        else:
            name_match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*的?信息', question)
            if name_match and "记住" not in question and "追加" not in question:
                name = name_match.group(1)
                from tools.tool_registry import get_tools
                tools = get_tools()
                patient_tool = tools.get("patient")
                if patient_tool:
                    info = patient_tool.recall(name)
                    if info:
                        patient_context = f"【患者档案】姓名：{name}，{info}\n"
                        logger.info(f"[Patient] 已加载患者 {name} 档案")

    if patient_context:
        final_augmented = patient_context + augmented_question
    else:
        final_augmented = augmented_question

    # 调用 Planner
    plan = planner.run(question)
    tool_list = plan.get("tools", [])
    if not tool_list:
        tool_list = ["drug", "guideline", "literature", "risk"]

    # 异步执行工具
    tool_results = await executor.run(tool_list, final_augmented)

    # 合成答案
    final_answer = synthesizer.run(final_augmented, tool_results)

    return {
        "success": True,
        "result": {
            "answer": final_answer,
            "tools_used": tool_list,
            "plan": plan,
            "tool_results": tool_results
        },
        "trace": {"executor": tool_results}
    }

# =========================
# API 端点
# =========================
@app.post("/v1/ask")
@app.post("/ask")
async def ask(req: ChatRequest, request: Request):
    question = req.question.strip()
    session_id = get_session_id(request)

    # =========================
    # 检查是否有待处理的多候选人选择
    # =========================
    if session_id in pending_selection:
        selection = pending_selection[session_id]
        candidates = selection.get("candidates", [])

        # 如果用户输入是数字
        if question.isdigit():
            idx = int(question) - 1
            if 0 <= idx < len(candidates):
                candidate = candidates[idx]
                # 清理状态
                del pending_selection[session_id]
                # 使用选中的患者生成评估表
                from tools.report_tool import ReportTool
                report_tool = ReportTool()
                result = report_tool._generate_from_candidate(candidate)
                # 将结果包装成标准响应格式
                return {
                    "success": result.get("success", True),
                    "result": {
                        "answer": result.get("answer", ""),
                        "tools_used": ["report"],
                        "plan": {"question": question, "tools": ["report"]},
                        "tool_results": []
                    },
                    "trace": {"executor": []}
                }
            else:
                # 无效数字，重新显示候选列表
                return {
                    "success": False,
                    "result": {
                        "answer": f"❌ 无效选择，请输入 1-{len(candidates)} 之间的数字。",
                        "tools_used": [],
                        "plan": {"question": question, "tools": []},
                        "tool_results": []
                    },
                    "trace": {"executor": []}
                }

        # 如果用户输入包含姓名，尝试精确匹配
        name_match = re.search(r'([\u4e00-\u9fa5]{2,4})', question)
        if name_match:
            name = name_match.group(1)
            for c in candidates:
                if c.get("name") == name:
                    # 精确匹配到姓名，使用该候选
                    del pending_selection[session_id]
                    from tools.report_tool import ReportTool
                    report_tool = ReportTool()
                    result = report_tool._generate_from_candidate(c)
                    return {
                        "success": result.get("success", True),
                        "result": {
                            "answer": result.get("answer", ""),
                            "tools_used": ["report"],
                            "plan": {"question": question, "tools": ["report"]},
                            "tool_results": []
                        },
                        "trace": {"executor": []}
                    }

        # 如果用户输入的是身份证号，更新当前患者并生成
        id_match = re.fullmatch(r'\s*(\d{17}[\dXx])\s*', question)
        if id_match:
            # 由 patient_tool 处理，继续正常流程
            pass

        # 如果用户输入其他内容，继续正常流程

    # =========================
    # 正常问答流程
    # =========================
    cache_key = get_cache_key(question)

    if should_cache(question, req.history):
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"[Cache] 命中缓存，问题：{question}")
            return cached_result
        logger.warning(f"[Cache] 未命中缓存，执行完整流程，问题：{question}")
        result = await process_question(question, req.history)
        cache.set(cache_key, result, ttl=3600)
        return result
    else:
        logger.warning(f"[Cache] 不适合缓存，执行完整流程，问题：{question}")
        return await process_question(question, req.history)

# =========================
# 健康检查
# =========================
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# =========================
# Prometheus 监控
# =========================
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
ERROR_COUNT = Counter('http_errors_total', 'Total HTTP errors', ['method', 'endpoint', 'status'])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(latency)
    if response.status_code >= 400:
        ERROR_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY), media_type="text/plain")

# =========================
# 报告下载接口
# =========================
from fastapi.responses import FileResponse

@app.get("/reports/{filename}")
async def download_report(filename: str):
    file_path = f"reports/{filename}"
    if not os.path.exists(file_path):
        return {"error": "文件不存在"}
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename
    )