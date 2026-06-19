from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import re
import json
import hashlib
import time
import logging
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from typing import List, Optional

from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from agents.consult_graph import ConsultGraph
from tools.tool_registry import get_tools

# =========================
# 日志配置
# =========================
LOG_FILE = "logs/app.log"
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("med_agent")
logger.setLevel(logging.INFO)

class TenantFilter(logging.Filter):
    def filter(self, record):
        try:
            from chat import get_current_tenant
            record.tenant_id = get_current_tenant()
        except:
            record.tenant_id = "default"
        return True

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10*1024*1024,
    backupCount=5
)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(tenant_id)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addFilter(TenantFilter())
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================
# 响应模型（用于 Swagger 文档）
# =========================
class ToolResult(BaseModel):
    success: bool
    id: str
    source: str
    answer: str
    debug: dict
    trace: list
    timestamp: float

class ResultData(BaseModel):
    answer: str
    tools_used: List[str]
    plan: dict
    tool_results: List[ToolResult]

class AskResponse(BaseModel):
    success: bool
    result: ResultData
    trace: dict

class ApprovalItem(BaseModel):
    id: str
    title: str
    requester: str
    created_at: str

class ApprovalsResponse(BaseModel):
    count: int
    items: List[ApprovalItem]

# =========================
# FastAPI App
# =========================
tags_metadata = [
    {"name": "问答", "description": "医学问答相关接口（快速问答、智能问诊）"},
    {"name": "审批", "description": "审批管理相关接口（列表、通过、驳回）"},
    {"name": "报告", "description": "报告生成与下载相关接口"},
    {"name": "运维", "description": "系统运维相关接口（健康检查、监控指标）"}
]

app = FastAPI(
    title="医疗Agent API",
    description="基于RAG的多工具医学问答助手，支持快速问答、智能问诊、评估表生成、多租户审批",
    version="3.0.0",
    openapi_tags=tags_metadata
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# 租户上下文
# =========================
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="default")

def get_current_tenant() -> str:
    return tenant_id_var.get()

@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    tenant_id = request.headers.get("X-Tenant-ID", "default")
    token = tenant_id_var.set(tenant_id)
    try:
        response = await call_next(request)
        return response
    finally:
        tenant_id_var.reset(token)

# =========================
# 全局用户状态（用于审批/患者操作）
# =========================
current_session_user = None

# =========================
# 组件初始化
# =========================
TOOLS = get_tools()
planner = Planner()
executor = Executor(TOOLS)
synthesizer = Synthesizer(api_key=os.getenv("DASHSCOPE_API_KEY"))
consult_graph = ConsultGraph()

class ChatRequest(BaseModel):
    question: str
    history: list = []

# =========================
# 缓存
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
    pronouns = ["它", "他", "她", "这个", "那个", "刚才", "上述", "以上", "以下"]
    if any(word in question for word in pronouns):
        logger.info(f"[Cache] 包含指代词，不缓存: {question}")
        return False
    if not history or len(history) <= 1:
        return True
    return False

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()

# =========================
# 核心问答逻辑
# =========================
async def process_question(question: str, history: list) -> dict:
    """执行完整的问答流程（不含拦截）"""
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
# 快速问答端点 (/ask)
# =========================
@app.post(
    "/v1/ask",
    tags=["问答"],
    summary="快速问答（V1）",
    description="单轮快速问答，适用于明确、完整的医学问题。支持多工具协同（drug/guideline/literature/risk/patient）。",
    response_model=AskResponse
)
@app.post(
    "/ask",
    tags=["问答"],
    summary="快速问答（V1）",
    description="单轮快速问答，适用于明确、完整的医学问题。支持多工具协同（drug/guideline/literature/risk/patient）。",
    response_model=AskResponse
)
async def ask(req: ChatRequest, request: Request):
    global current_session_user
    question = req.question.strip()
    history = req.history

    # 1. 身份声明
    if re.match(r'^用户[：:]', question):
        user_match = re.search(r'用户[：:]\s*(\S+)', question)
        if user_match:
            current_session_user = user_match.group(1)
            logger.info(f"[身份声明] 当前用户设置为: {current_session_user}")
            return {
                "success": True,
                "result": {
                    "answer": f"✅ 已识别当前用户：{current_session_user}",
                    "tools_used": [],
                    "plan": {"question": question, "tools": []},
                    "tool_results": []
                },
                "trace": {"executor": []}
            }

    # 2. 审批指令拦截（不经过 process_question）
    approval_keywords = ["待审批", "审批通过", "驳回", "已通过", "已驳回", "全部列表", "审批列表"]
    if any(kw in question for kw in approval_keywords):
        print(f"[Chat] 拦截到审批指令，直接处理: {question}")
        from tools.approval_tool import ApprovalTool
        approval_tool = ApprovalTool()
        result = approval_tool.run(question)  # 传入原始问题，不带历史
        return {
            "success": True,
            "result": {
                "answer": result.get("answer", ""),
                "tools_used": ["approval"],
                "plan": {"question": question, "tools": ["approval"]},
                "tool_results": [result]
            },
            "trace": {"executor": [result]}
        }

    # 3. 患者操作拦截
    if re.search(r'记住患者|记录患者|追加患者|补充患者', question, re.IGNORECASE):
        print(f"[Chat] 拦截到患者操作: {question}")
        from tools.patient_tool import PatientTool
        patient_tool = PatientTool()
        # 直接解析
        name_match = re.search(r'(?:记住患者|记录患者|追加患者|补充患者)\s*([\u4e00-\u9fa5]{2,4})\s*[:：]?\s*(.+)', question)
        if name_match:
            name = name_match.group(1)
            info = name_match.group(2).strip()
            result = patient_tool.remember(name, info, append=False)
            return {
                "success": True,
                "result": {
                    "answer": result.get("answer", ""),
                    "tools_used": ["patient"],
                    "plan": {"question": question, "tools": ["patient"]},
                    "tool_results": [result]
                },
                "trace": {"executor": [result]}
            }
        else:
            return {
                "success": False,
                "result": {
                    "answer": "❌ 无法解析患者信息，请使用格式：记住患者 张三：60岁，男，肚子痛",
                    "tools_used": [],
                    "plan": {"question": question, "tools": []},
                    "tool_results": []
                },
                "trace": {"executor": []}
            }

    # 4. 正常问答（走缓存 + process_question）
    cache_key = get_cache_key(question)
    if should_cache(question, history):
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"[Cache] 命中缓存，问题：{question}")
            return cached_result
        logger.warning(f"[Cache] 未命中缓存，执行完整流程，问题：{question}")
        result = await process_question(question, history)
        cache.set(cache_key, result, ttl=3600)
        return result
    else:
        logger.warning(f"[Cache] 不适合缓存，执行完整流程，问题：{question}")
        return await process_question(question, history)

# =========================
# 智能问诊端点 (/consult)
# =========================
@app.post(
    "/consult",
    tags=["问答"],
    summary="智能问诊（V2）",
    description="基于 LangGraph 的多轮交互式问诊。Agent 会主动追问缺失信息（年龄、过敏史、用药史），收集完整后给出个性化用药建议。",
    response_model=AskResponse
)
async def consult(req: ChatRequest):
    global current_session_user
    question = req.question.strip()
    history = req.history

    # 身份声明
    if re.match(r'^用户[：:]', question):
        user_match = re.search(r'用户[：:]\s*(\S+)', question)
        if user_match:
            current_session_user = user_match.group(1)
            logger.info(f"[Consult] 身份声明，当前用户设置为: {current_session_user}")
            return {
                "success": True,
                "result": {
                    "answer": f"✅ 已识别当前用户：{current_session_user}",
                    "tools_used": [],
                    "plan": {"question": question, "tools": []},
                    "tool_results": []
                },
                "trace": {"executor": []}
            }

    # 患者操作拦截
    if re.search(r'记住患者|记录患者|追加患者|补充患者', question, re.IGNORECASE):
        print(f"[Consult] 拦截到患者操作: {question}")
        from tools.patient_tool import PatientTool
        patient_tool = PatientTool()
        name_match = re.search(r'(?:记住患者|记录患者|追加患者|补充患者)\s*([\u4e00-\u9fa5]{2,4})\s*[:：]?\s*(.+)', question)
        if name_match:
            name = name_match.group(1)
            info = name_match.group(2).strip()
            result = patient_tool.remember(name, info, append=False)
            return {
                "success": True,
                "result": {
                    "answer": result.get("answer", ""),
                    "tools_used": ["patient"],
                    "plan": {"question": question, "tools": ["patient"]},
                    "tool_results": [result]
                },
                "trace": {"executor": [result]}
            }
        else:
            return {
                "success": False,
                "result": {
                    "answer": "❌ 无法解析患者信息，请使用格式：记住患者 张三：60岁，男，肚子痛",
                    "tools_used": [],
                    "plan": {"question": question, "tools": []},
                    "tool_results": []
                },
                "trace": {"executor": []}
            }

    # 审批指令拦截
    approval_keywords = ["待审批", "审批通过", "驳回", "已通过", "已驳回", "全部列表", "审批列表"]
    if any(kw in question for kw in approval_keywords):
        print(f"[Consult] 拦截到审批指令: {question}")
        from tools.approval_tool import ApprovalTool
        approval_tool = ApprovalTool()
        result = approval_tool.run(question)
        return {
            "success": True,
            "result": {
                "answer": result.get("answer", ""),
                "tools_used": ["approval"],
                "plan": {"question": question, "tools": ["approval"]},
                "tool_results": [result]
            },
            "trace": {"executor": [result]}
        }

    # 正常进入 LangGraph 问诊
    print(f"[Consult] 正常进入 LangGraph 问诊流程")
    try:
        answer = consult_graph.run(question, history)
        return {
            "success": True,
            "result": {
                "answer": answer,
                "tools_used": [],
                "plan": {"question": question, "mode": "consult"},
                "tool_results": []
            },
            "trace": {"executor": []}
        }
    except Exception as e:
        return {
            "success": False,
            "result": {
                "answer": f"问诊处理失败：{str(e)}",
                "tools_used": [],
                "plan": {"question": question, "mode": "consult"},
                "tool_results": []
            },
            "trace": {"executor": []}
        }

# =========================
# 审批列表接口（用于前端侧边栏）
# =========================
@app.get(
    "/approvals",
    tags=["审批"],
    summary="获取待审批列表",
    description="查询当前用户的待审批项列表，供前端侧边栏轮询使用。支持多租户隔离，只返回当前租户的数据。",
    response_model=ApprovalsResponse
)
async def get_approvals():
    from tools.tool_registry import get_tools
    from chat import current_session_user

    print(f"[DEBUG] /approvals - current_session_user: {current_session_user}")
    user = current_session_user if current_session_user else "current_user"
    print(f"[DEBUG] /approvals - 查询用户: {user}")

    tools = get_tools()
    approval_tool = tools.get("approval")
    if approval_tool:
        items = approval_tool.list_pending_by_user(user)
        print(f"[DEBUG] /approvals - 查询到 {len(items)} 条记录")
        return {"count": len(items), "items": items}
    return {"count": 0, "items": []}

# =========================
# 健康检查
# =========================
@app.get(
    "/health",
    tags=["运维"],
    summary="健康检查",
    description="检查服务是否正常运行，返回固定状态 'ok'。"
)
async def health_check():
    return {"status": "ok"}

# =========================
# Prometheus 监控
# =========================
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'tenant_id'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint', 'tenant_id'])
ERROR_COUNT = Counter('http_errors_total', 'Total HTTP errors', ['method', 'endpoint', 'status', 'tenant_id'])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    tenant_id = get_current_tenant()
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        tenant_id=tenant_id
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
        tenant_id=tenant_id
    ).observe(latency)
    if response.status_code >= 400:
        ERROR_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            tenant_id=tenant_id
        ).inc()
    return response

@app.get(
    "/metrics",
    tags=["运维"],
    summary="Prometheus 监控指标",
    description="暴露 Prometheus 格式的监控指标（请求数、延迟、错误数等），支持租户维度拆分。"
)
async def metrics():
    return Response(generate_latest(REGISTRY), media_type="text/plain")

# =========================
# 报告下载
# =========================
@app.get(
    "/reports/{filename}",
    tags=["报告"],
    summary="下载评估表",
    description="根据文件名下载生成的评估表 Word 文档（.docx 格式）。",
    responses={
        200: {"description": "文件下载成功", "content": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document": {}}},
        404: {"description": "文件不存在"}
    }
)
async def download_report(filename: str):
    file_path = f"reports/{filename}"
    if not os.path.exists(file_path):
        return {"error": "文件不存在"}
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename
    )