from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import re
import json
import hashlib
import time

from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from tools.tool_registry import get_tools

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
    # 包含指代词 → 不缓存
    pronouns = ["它", "他", "她", "这个", "那个", "刚才", "上述", "以上", "以下"]
    if any(word in question for word in pronouns):
        print(f"[Cache] 包含指代词，不缓存: {question}")
        return False
    
    # 历史为空或只有一条用户问题 → 可以缓存
    if not history or len(history) <= 1:
        return True
    
    # 其他情况默认不缓存（保守策略）
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
        recent = history[-4:]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
        augmented_question = f"对话历史：\n{history_str}\n当前问题：{question}"
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
                    print(f"[Patient] 已加载患者 {name} 档案")
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
                        print(f"[Patient] 已加载患者 {name} 档案")

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
async def ask(req: ChatRequest):
    # 生成缓存 key（只基于问题）
    cache_key = get_cache_key(req.question)
    
    if should_cache(req.question, req.history):
        cached_result = cache.get(cache_key)
        if cached_result:
            print(f"[Cache] 命中缓存，问题：{req.question}")
            return cached_result
        print(f"[Cache] 未命中缓存，执行完整流程，问题：{req.question}")
        result = await process_question(req.question, req.history)
        cache.set(cache_key, result, ttl=3600)
        return result
    else:
        # 不适合缓存，直接执行完整流程
        print(f"[Cache] 不适合缓存，执行完整流程，问题：{req.question}")
        return await process_question(req.question, req.history)