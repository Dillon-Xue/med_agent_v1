from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, re, json, hashlib, time, logging, pymysql, asyncio, httpx, requests
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from typing import List, Optional
from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from agents.consult_graph import ConsultGraph
from tools.tool_registry import get_tools
# =========================
# Trace 存储（内存）
# =========================
trace_store = {}
trace_lock = asyncio.Lock()

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

class UploadResponse(BaseModel):
    success: bool
    filename: str
    content: str
    preview: str          # 前 200 字符预览
    source: str
    error: Optional[str] = None

class HistoryItem(BaseModel):
    id: int
    role: str
    content: str
    tools_used: Optional[List[str]] = None
    file_name: Optional[str] = None
    conversation_type: str
    created_at: str

class HistoryResponse(BaseModel):
    success: bool
    items: List[HistoryItem]
    count: int

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
async def process_question(question: str, history: list, trace_callback=None) -> dict:
    print(f"[process_question] 收到的 trace_callback 是否为 None: {trace_callback is None}")
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
    plan = planner.run(question, trace_callback=trace_callback)
    tool_list = plan.get("tools", [])
    if not tool_list:
        tool_list = ["drug", "guideline", "literature", "risk"]

    # 异步执行工具
    tool_results = await executor.run(tool_list, final_augmented, trace_callback=trace_callback)

    # 合成答案
    final_answer = synthesizer.run(final_augmented, tool_results, trace_callback=trace_callback)

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


def save_conversation(session_id: str, role: str, content: str, 
                      tools_used: list = None, file_name: str = None,
                      conversation_type: str = "quick", tenant_id: str = None):
    """保存单条对话记录到 conversations 表"""
    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "yourpassword"),
            database=os.getenv("DB_NAME", "patient_db"),
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        
        tools_json = json.dumps(tools_used) if tools_used else None
        if tenant_id is None:
            tenant_id = get_current_tenant()
        
        cursor.execute(
            """INSERT INTO conversations 
               (session_id, tenant_id, role, content, tools_used, file_name, conversation_type) 
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (session_id, tenant_id, role, content, tools_json, file_name, conversation_type)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"保存对话历史失败: {e}")


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
    print(f"[ASK] ==== 请求开始 ====")
    print(f"[ASK] 所有 headers: {request.headers}")
    print(f"[ASK] X-Trace-ID: {request.headers.get('X-Trace-ID')}")
    global current_session_user
    question = req.question.strip()
    history = req.history
    session_id = request.headers.get("X-Session-ID")
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

        # 获取会话信息
        session_id = request.headers.get("X-Session-ID")
        conv_type = request.headers.get("X-Conversation-Type", "unknown")
        print(f"[Chat] 审批拦截 - session_id: {session_id}, conv_type: {conv_type}")

        from tools.approval_tool import ApprovalTool
        approval_tool = ApprovalTool()
        result = approval_tool.run(question)  # 传入原始问题，不带历史
        # 🆕 保存审批助手的对话记录
        if session_id:
            print("[Chat] 审批拦截 - 开始保存对话")
            try:
                save_conversation(session_id, "user", question,
                                conversation_type="approval")
                answer = result.get("answer", "")
                save_conversation(session_id, "assistant", answer,
                                conversation_type="approval")
                print("[Chat] 审批拦截 - 保存完成")
            except Exception as e:
                print(f"[Chat] 审批拦截 - 保存失败: {e}")
        else:
            print("[Chat] 审批拦截 - session_id 为空，跳过保存")
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
    trace_id = request.headers.get("X-Trace-ID") or request.headers.get("x-trace-id")  # 前端生成的 trace_session_id
    print(f"[Trace] 收到 trace_id: {trace_id}")
    
    # 🆕 定义 trace_callback 函数
    def trace_callback(step_type: str, data: dict):
        print(f"[Trace] 收到 step: {step_type}, data: {data}")
        if not trace_id:
            return
        try:
            print(f"[Trace] 正在记录 step: {step_type}")
            if trace_id in trace_store:
                trace_store[trace_id]["steps"].append({
                    "step_type": step_type,
                    "timestamp": time.time(),
                    "data": data
                })
            print(f"[Trace] step 记录完成: {step_type}")
        except Exception as e:
            print(f"[Trace] 回调失败: {e}")

    # 🆕 启动 trace
    if trace_id:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "http://localhost:8000/trace/start",
                    json={"session_id": trace_id, "question": question},
                    timeout=2.0
                )
            print(f"[Trace] 启动成功: {trace_id}")
        except Exception as e:
            print(f"[Trace] 启动失败: {e}")

    # ===== 执行问答，获取 result =====
    result = None
    
    if should_cache(question, history):
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"[Cache] 命中缓存，问题：{question}")
            result = cached_result
            # 🆕 缓存命中时也记录 trace（仅记录命中状态）
            if trace_id:
                try:
                    if trace_id in trace_store:
                        trace_store[trace_id]["steps"].append({
                            "step_type": "cache",
                            "timestamp": time.time(),
                            "data": {"status": "hit", "question": question}
                        })
                except Exception as e:
                    print(f"[Trace] 缓存命中记录失败: {e}")
        else:
            logger.warning(f"[Cache] 未命中缓存，执行完整流程，问题：{question}")
            result = await process_question(question, history, trace_callback=trace_callback if trace_id else None)
            cache.set(cache_key, result, ttl=3600)
    else:
        logger.warning(f"[Cache] 不适合缓存，执行完整流程，问题：{question}")
        result = await process_question(question, history, trace_callback=trace_callback if trace_id else None)

    # ===== 🆕 统一保存历史记录（所有分支都执行） =====
    session_id = request.headers.get("X-Session-ID")
    conv_type = request.headers.get("X-Conversation-Type", "quick")
    if session_id and result and result.get("success"):
        try:
            save_conversation(session_id, "user", question, conversation_type=conv_type)
            answer = result.get("result", {}).get("answer", "")
            tools_used = result.get("result", {}).get("tools_used", [])
            save_conversation(session_id, "assistant", answer, tools_used=tools_used, conversation_type=conv_type)
            logger.info(f"[历史记录] 已保存会话 {session_id}，类型：{conv_type}")
        except Exception as e:
            logger.error(f"[历史记录] 保存失败: {e}")

    return result

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
async def consult(req: ChatRequest, request: Request):
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

        # 🆕 保存审批助手的对话记录
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            save_conversation(session_id, "user", question,
                            conversation_type="approval")
            answer = result.get("answer", "")
            save_conversation(session_id, "assistant", answer,
                            conversation_type="approval")

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
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            # 保存用户问题
            save_conversation(session_id, "user", question, 
                            conversation_type="consult")
            # 保存助手回答
            save_conversation(session_id, "assistant", answer, 
                            conversation_type="consult")
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


from fastapi import File, UploadFile, Form
from typing import Optional
from tools.file_tool import FileTool

# 在 app 初始化后，初始化 FileTool
file_tool = FileTool(api_key=os.getenv("DASHSCOPE_API_KEY"))

@app.post(
    "/upload",
    tags=["文件"],
    summary="上传并解析文件（图片/PDF）",
    description="上传图片或 PDF 文件，系统自动识别并提取其中的医疗信息（患者姓名、年龄、诊断、药品等），返回纯文本内容供后续对话使用。",
    response_model=UploadResponse
)
async def upload_file(
    file: UploadFile = File(..., description="支持 PNG、JPG、PDF 格式，最大 10MB"),
    module: str = Form("consult", description="调用模块：quick | consult | approval")
):
    """
    上传文件并解析内容
    
    - **图片 (PNG/JPG)**：调用 Qwen-VL 多模态模型识别文字和医疗信息
    - **PDF**：提取全文文本，自动截断超长内容
    """
    # 1. 文件大小校验（10MB）
    content_bytes = await file.read()
    if len(content_bytes) > 10 * 1024 * 1024:
        return UploadResponse(
            success=False,
            filename=file.filename,
            content="",
            preview="",
            source="file",
            error="文件大小超过 10MB 限制"
        )
    
    # 2. 调用 FileTool 解析
    result = file_tool.run(content_bytes, file.filename, module)
    
    if not result.get("success"):
        return UploadResponse(
            success=False,
            filename=file.filename,
            content="",
            preview="",
            source="file",
            error=result.get("answer", "解析失败")
        )
    
    content = result.get("answer", "")
    preview = content[:200] + ("..." if len(content) > 200 else "")
    
    return UploadResponse(
        success=True,
        filename=file.filename,
        content=content,
        preview=preview,
        source="file"
    )

@app.get(
    "/history",
    tags=["会话"],
    summary="获取会话历史",
    description="根据 session_id 拉取最近 50 条对话记录，用于刷新页面后恢复聊天上下文",
    response_model=HistoryResponse
)
async def get_history(session_id: str, limit: int = 50, conversation_type: str = None):
    """
    获取指定会话的历史记录
    
    - **session_id**: 会话标识（前端生成，存储在 localStorage）
    - **limit**: 返回条数，默认 50
    - **conversation_type**: 可选过滤（quick / consult / approval）
    """
    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "yourpassword"),
            database=os.getenv("DB_NAME", "patient_db"),
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        
        sql = """SELECT id, role, content, tools_used, file_name, conversation_type, created_at 
                 FROM conversations 
                 WHERE session_id = %s AND tenant_id = %s"""
        params = [session_id, get_current_tenant()]
        
        if conversation_type:
            sql += " AND conversation_type = %s"
            params.append(conversation_type)
        
        sql += " ORDER BY created_at ASC LIMIT %s"
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            tools = json.loads(row[3]) if row[3] else []
            items.append({
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "tools_used": tools,
                "file_name": row[4],
                "conversation_type": row[5],
                "created_at": row[6].strftime("%Y-%m-%d %H:%M:%S") if row[6] else ""
            })
        
        return {"success": True, "items": items, "count": len(items)}
    except Exception as e:
        logger.error(f"获取会话历史失败: {e}")
        return {"success": False, "items": [], "count": 0, "error": str(e)}
# =========================
# Trace 存储（内存，生产环境可改为 Redis）
# =========================
from pydantic import BaseModel

class TraceStep(BaseModel):
    step_type: str  # planner / executor / retriever / synthesizer
    timestamp: float
    data: dict

class TraceData(BaseModel):
    session_id: str
    question: str
    steps: List[TraceStep]
    start_time: float
    end_time: Optional[float] = None

class TraceStartRequest(BaseModel):
    session_id: str
    question: str

@app.post("/trace/start")
async def start_trace(req: TraceStartRequest):
    async with trace_lock:
        trace_store[req.session_id] = {
            "session_id": req.session_id,
            "question": req.question,
            "steps": [],
            "start_time": time.time(),
            "end_time": None
        }
    return {"status": "ok"}

@app.post("/trace/step")
async def add_trace_step(session_id: str, step_type: str, data: dict):
    """添加一个追踪步骤"""
    if session_id not in trace_store:
        return {"status": "error", "message": "trace not found"}
    
    async with trace_lock:
        trace_store[session_id]["steps"].append({
            "step_type": step_type,
            "timestamp": time.time(),
            "data": data
        })
    return {"status": "ok"}

@app.get("/trace/{session_id}")
async def get_trace(session_id: str):
    """获取完整的追踪链路"""
    if session_id not in trace_store:
        return {"success": False, "error": "trace not found"}
    
    trace = trace_store[session_id].copy()
    trace["steps"] = trace["steps"]
    return {"success": True, "data": trace}

@app.delete("/trace/{session_id}")
async def clear_trace(session_id: str):
    """清除追踪数据"""
    async with trace_lock:
        if session_id in trace_store:
            del trace_store[session_id]
    return {"status": "ok"}