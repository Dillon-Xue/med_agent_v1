from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import re

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

@app.post("/v1/ask")
@app.post("/ask")
def ask(req: ChatRequest):
    question = req.question
    history = req.history
    stripped = question.strip()

    # =========================
    # 🚀 纯中文姓名预检（2-4个中文字符，无其他内容）
    # =========================
    if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', stripped):
        from tools.tool_registry import get_tools
        tools = get_tools()
        patient_tool = tools.get("patient")
        if patient_tool:
            info = patient_tool.recall(stripped)
            if info:
                # 找到患者，直接返回档案
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
            # 未找到患者，不返回，继续走正常流程
            # 这样“布洛芬”会继续走 drug 工具

    # =========================
    # 构建上下文（历史 + 患者档案预加载）
    # =========================
    if history:
        recent = history[-4:]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
        augmented_question = f"对话历史：\n{history_str}\n当前问题：{question}"
    else:
        augmented_question = question

    # 患者档案预加载（用于“患者张三的信息”、“张三的信息”等）
    patient_context = ""
    # 排除“记住”、“追加”等指令，避免干扰
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
            # 匹配“XXX的信息”格式
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

    # =========================
    # 调用 Planner/Executor/Synthesizer
    # =========================
    plan = planner.run(question)
    tool_list = plan.get("tools", [])
    if not tool_list:
        tool_list = ["drug", "guideline", "literature", "risk"]

    tool_results = executor.run(tool_list, final_augmented)
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