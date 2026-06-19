import os
import re
import json
import asyncio
import concurrent.futures
import traceback
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from tools.tool_registry import get_tools
from openai import OpenAI

# 全局调试开关
DEBUG = True

class ConsultGraph:
    def __init__(self):
        self.tools = get_tools()
        self.planner = Planner()
        self.executor = Executor(self.tools)
        self.synthesizer = Synthesizer(api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("analyze_gap", self._analyze_gap)
        graph.add_node("ask_missing", self._ask_missing)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("synthesize", self._synthesize)

        graph.set_entry_point("analyze_gap")

        graph.add_conditional_edges(
            "analyze_gap",
            self._should_ask_or_execute,
            {
                "ask": "ask_missing",
                "execute": "execute_tools"
            }
        )
        graph.add_edge("ask_missing", END)
        graph.add_edge("execute_tools", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _log(self, *args, **kwargs):
        if DEBUG:
            print("[DEBUG]", *args, **kwargs)

    def _extract_info_from_text(self, text: str) -> dict:
        """使用 LLM 从文本中提取结构化患者信息"""
        # 检查是否为询问
        inquiry_patterns = [
            r'可以\s*(吃|服用|用)\s*([\u4e00-\u9fa5]{2,})',
            r'能\s*(吃|服用|用)\s*([\u4e00-\u9fa5]{2,})',
            r'可\s*以\s*(吃|服用|用)\s*([\u4e00-\u9fa5]{2,})'
        ]
        is_inquiry = any(re.search(p, text) for p in inquiry_patterns)

        try:
            prompt = f"""
    请从以下用户输入中提取患者的健康信息，返回 JSON 格式。
    输入：{text}

    需要提取的字段：
    - name: 患者姓名（如果有“我是XX”或“我叫XX”则提取，否则为 null）
    - age: 年龄（如“30岁”，如果没有则为 null）
    - allergy: 过敏史（如果有过敏信息，提取具体过敏原，如“西红柿”；如果明确说“无过敏”则为“无”；如果没有提及则为 null）
    - medication: 当前正在服用的药物（如果用户明确说“在服用XX”，提取药物名称；如果明确说“未服药”则为“无”；如果没有提及则为 null）

    重要规则：
    1. 如果用户是在询问“可以吃XX吗？”、“能吃XX吗？”，这不算“正在服用”，medication 应为 null。
    2. 如果用户说“在服用硝苯地平”，则 medication 为“硝苯地平”。
    3. 如果用户说“80岁，无过敏史，在服用硝苯地平”，则 age 为“80岁”，allergy 为“无”，medication 为“硝苯地平”。

    只输出 JSON，不要其他内容。
    """
            resp = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            # 提取 JSON 块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            return {
                "姓名": data.get("name"),
                "年龄": data.get("age"),
                "过敏史": data.get("allergy"),
                "用药史": data.get("medication")
            }
        except Exception as e:
            print(f"[ConsultGraph] LLM 提取失败，降级到正则: {e}")
            return self._extract_info_with_regex(text)

    def _extract_info_with_regex(self, text: str) -> dict:
        """降级方案：使用正则提取"""
        info = {"姓名": None, "年龄": None, "过敏史": None, "用药史": None}
        # 姓名
        name_match = re.search(r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})', text)
        if name_match:
            info["姓名"] = name_match.group(1)
        # 年龄
        age_match = re.search(r'(\d{1,3})\s*岁', text)
        if age_match:
            info["年龄"] = age_match.group(1) + "岁"
        # 过敏史
        if "过敏" in text:
            if re.search(r'(无|没有|否认)\s*过敏', text):
                info["过敏史"] = "无"
            else:
                info["过敏史"] = "有过敏史"
        # 用药史（简单处理）
        if "服用" in text or "吃" in text:
            drug_match = re.search(r'服用\s*([\u4e00-\u9fa5]{2,6})', text)
            if drug_match:
                info["用药史"] = drug_match.group(1)
        return info

    def _get_patient_info_from_db(self, name: str) -> dict:
        from tools.tool_registry import get_tools
        tools = get_tools()
        patient_tool = tools.get("patient")
        if patient_tool:
            stored = patient_tool.recall(name)
            if stored:
                info_text = stored.get("info", "")
                self._log("从数据库读取到:", info_text)
                extracted = self._extract_info_from_text(info_text)
                extracted["全文档案"] = info_text
                return extracted
        return {}

    def _analyze_gap(self, state: AgentState) -> AgentState:
        self._log("========== _analyze_gap 开始 ==========")
        history = state.get("history", [])
        patient_info = state.get("patient_info", {})
        current_patient = state.get("current_patient", "")

        # 从 history 中提取姓名
        all_user_text = ""
        for msg in history:
            if msg.get("role") == "user":
                all_user_text += " " + msg.get("content", "")
        name_match = re.search(r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
        if name_match:
            current_patient = name_match.group(1)

        # 如果 patient_info 为空，从数据库加载
        if current_patient and not patient_info:
            db_info = self._get_patient_info_from_db(current_patient)
            if db_info:
                for key in ["年龄", "过敏史", "用药史"]:
                    if db_info.get(key):
                        patient_info[key] = db_info[key]
                if db_info.get("全文档案"):
                    patient_info["全文档案"] = db_info["全文档案"]

        # 从历史中提取新信息
        extracted = self._extract_info_from_text(all_user_text)
        self._log("从历史提取的新信息:", extracted)
        if extracted.get("年龄"):
            patient_info["年龄"] = extracted["年龄"]
        if extracted.get("过敏史"):
            patient_info["过敏史"] = extracted["过敏史"]
        if extracted.get("用药史"):
            patient_info["用药史"] = extracted["用药史"]

        # 确定缺失信息
        missing = []
        if "年龄" not in patient_info:
            missing.append("年龄")
        if "过敏史" not in patient_info:
            missing.append("过敏史")
        if "用药史" not in patient_info:
            missing.append("用药史")

        state["current_patient"] = current_patient
        state["patient_info"] = patient_info
        state["missing_info"] = missing
        state["iteration"] = state.get("iteration", 0) + 1
        state["max_iterations"] = state.get("max_iterations", 5)

        self._log("最终 patient_info:", patient_info)
        self._log("最终 missing_info:", missing)
        self._log("========== _analyze_gap 结束 ==========")
        return state

    def _should_ask_or_execute(self, state: AgentState) -> str:
        if state["missing_info"] and state["iteration"] <= state.get("max_iterations", 5):
            return "ask"
        return "execute"

    def _ask_missing(self, state: AgentState) -> AgentState:
        missing = state["missing_info"]
        if not missing:
            state["final_answer"] = "请描述您的具体症状，我来帮您分析。"
            return state

        questions = {
            "年龄": "请问您的年龄是？（例如：30岁）",
            "过敏史": "请问您是否有药物过敏史？（例如：无过敏或对青霉素过敏）",
            "用药史": "请问您目前是否在服用其他药物？（例如：未服用或服用感冒灵）"
        }
        ask_text = "为了给您更精准的建议，请补充以下信息：\n"
        for item in missing:
            ask_text += "- " + questions.get(item, item) + "\n"
        state["final_answer"] = ask_text
        return state

    def _execute_tools(self, state: AgentState) -> AgentState:
        # 从历史中提取原始用药咨询问题
        question = state["question"] 
        patient_info = state.get("patient_info", {})
        current_patient = state.get("current_patient", "")
        history = state.get("history", [])
        parts = []
        if patient_info.get("年龄"):
            parts.append("年龄：" + patient_info["年龄"])
        if patient_info.get("过敏史"):
            parts.append("过敏史：" + patient_info["过敏史"])
        if patient_info.get("用药史"):
            parts.append("用药史：" + patient_info["用药史"])
        if patient_info.get("全文档案"):
            parts.append("既往档案：" + patient_info["全文档案"])
        
        if parts:
            enhanced = "患者信息：" + "，".join(parts) + "。问题：" + question
        else:
            enhanced = question
        
        plan = self.planner.run(enhanced)  # 使用增强问题

        original_question = None
        for msg in history:
            if msg.get("role") == "user":
                original_question = msg.get("content")
                break
        if not original_question:
            original_question = state["question"]


        if current_patient and not patient_info.get("全文档案"):
            db_info = self._get_patient_info_from_db(current_patient)
            if db_info.get("全文档案"):
                patient_info["全文档案"] = db_info["全文档案"]

        parts = []
        if patient_info.get("年龄"):
            parts.append("年龄：" + patient_info["年龄"])
        if patient_info.get("过敏史"):
            parts.append("过敏史：" + patient_info["过敏史"])
        if patient_info.get("用药史"):
            parts.append("用药史：" + patient_info["用药史"])
        if patient_info.get("全文档案"):
            parts.append("既往档案：" + patient_info["全文档案"])

        info_str = "，".join(parts) if parts else ""
        enhanced = "患者信息：" + info_str + "。问题：" + original_question if parts else original_question

        plan = self.planner.run(enhanced)
        tool_list = plan.get("tools", ["drug", "guideline", "literature", "risk"])
        # 移除 patient 工具，避免干扰
        tool_list = [t for t in tool_list if t != "patient"]

        # 执行工具（异步）
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.executor.run(tool_list, enhanced))
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self.executor.run(tool_list, enhanced))
        except Exception as e:
            print("[ConsultGraph] 执行工具出错:", e)
            results = []

        if not isinstance(results, list):
            results = []

        state["tool_results"] = results
        return state

    def _synthesize(self, state: AgentState) -> AgentState:
        print(f"[ConsultGraph._synthesize] 开始执行")
        question = state["question"]
        results = state.get("tool_results", [])
        if not isinstance(results, list):
            results = []
        answer = self.synthesizer.run(question, results)
        print(f"[ConsultGraph._synthesize] 合成答案完成")
        '''
        current_patient = state.get("current_patient")
        patient_info = state.get("patient_info", {})
        if current_patient and patient_info:
            info_parts = []
            if patient_info.get("年龄"):
                info_parts.append(str(patient_info["年龄"]))
            if patient_info.get("过敏史"):
                info_parts.append("过敏史：" + str(patient_info["过敏史"]))
            if patient_info.get("用药史"):
                info_parts.append("用药史：" + str(patient_info["用药史"]))
            info_text = "，".join(info_parts)
            if info_text:
                from tools.tool_registry import get_tools
                tools = get_tools()
                patient_tool = tools.get("patient")
                if patient_tool:
                    patient_tool.remember(current_patient, info_text, append=False)
                    print(f"[ConsultGraph] 已更新患者 {current_patient} 的档案：{info_text}")
        '''
        print(f"[ConsultGraph._synthesize] 自动保存已禁用，跳过")
        state["final_answer"] = answer
        return state

    def run(self, question: str, history: list = None) -> str:
        self._log("========== run 开始 ==========")
        self._log("问题:", question)
        self._log("历史:", history)

        all_user_text = ""
        for msg in (history or []):
            if msg.get("role") == "user":
                all_user_text += " " + msg.get("content", "")
        all_user_text += " " + question
        self._log("所有用户文本:", all_user_text)

        name_match = re.search(r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
        name = name_match.group(1) if name_match else None
        self._log("提取姓名:", name)

        patient_info = {}
        if name:
            db_info = self._get_patient_info_from_db(name)
            if db_info:
                for key in ["年龄", "过敏史", "用药史"]:
                    if db_info.get(key):
                        patient_info[key] = db_info[key]
                if db_info.get("全文档案"):
                    patient_info["全文档案"] = db_info["全文档案"]
                self._log("从数据库加载档案:", patient_info)

        extracted = self._extract_info_from_text(all_user_text)
        if extracted.get("年龄"):
            patient_info["年龄"] = extracted["年龄"]
        if extracted.get("过敏史"):
            patient_info["过敏史"] = extracted["过敏史"]
        if extracted.get("用药史"):
            patient_info["用药史"] = extracted["用药史"]

        missing = []
        if "年龄" not in patient_info:
            missing.append("年龄")
        if "过敏史" not in patient_info:
            missing.append("过敏史")
        if "用药史" not in patient_info:
            missing.append("用药史")

        initial_state = {
            "question": question,
            "history": history or [],
            "current_patient": name or "",
            "patient_info": patient_info,
            "missing_info": missing,
            "tool_results": [],
            "final_answer": "",
            "iteration": 0,
            "max_iterations": 5
        }
        self._log("初始状态:", initial_state)
        self._log("========== run 结束 ==========")

        try:
            result = self.graph.invoke(initial_state)
            return result.get("final_answer", "未能生成回答")
        except Exception as e:
            print("ConsultGraph 运行异常:")
            traceback.print_exc()
            return "问诊过程出错：" + str(e)