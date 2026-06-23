import os
import re
from agents.llm_planner import LLMPlanner


class Planner:
    def __init__(self, use_llm: bool = True, trace_callback=None):
        self.rules = {
            "drug": ["成分", "说明书", "副作用", "禁忌", "用法", "剂量"],
            "guideline": ["指南", "治疗", "推荐", "临床", "诊疗"],
            "literature": ["论文", "研究", "文献", "机制", "试验"],
            "risk": ["相互作用", "不良反应", "禁忌", "风险", "安全", "冲突", "合用", "联合用药", "危险"],
            "patient": ["患者", "记住", "记录", "档案", "病历", "姓名", "信息", "追加", "补充", "病号", "增加"],
            "report": ["评估表", "生成报告", "生成评估表", "评估报告", "生成病历", "生成档案", "生成记录"]
        }
        self.use_llm = use_llm
        self.trace_callback = trace_callback
        if self.use_llm:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY not set")
            self.llm_planner = LLMPlanner(api_key)

    def rule_based(self, question: str) -> list:
        tools = set()
        stripped = question.strip()

        # 1. 纯姓名检测（2-4个中文字符）
        if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', stripped):
            tools.add("patient")

        # 2. 使用正则检测 patient 相关关键词
        if re.search(r'患者|记住|记录|档案|病历|信息|追加|补充|病号|增加', question):
            tools.add("patient")

        # 3. 使用正则检测 report 相关关键词
        if re.search(r'评估表|生成报告|生成评估表|评估报告|生成病历|生成档案|生成记录', question):
            tools.add("report")

        # 4. 匹配其他工具的关键词（跳过 patient 和 report）
        for tool, keywords in self.rules.items():
            if tool in ["patient", "report"]:
                continue
            for kw in keywords:
                if kw in question:
                    tools.add(tool)

        # 5. 默认工具
        if not tools:
            return ["drug", "guideline", "literature", "risk"]

        # 6. 如果只有 patient（纯姓名触发），追加默认工具
        if tools == {"patient"}:
            tools.update(["drug", "guideline", "literature", "risk"])

        return list(tools)

    def run(self, question: str, trace_callback=None) -> dict:
        print(f"[Planner] ===== run 被调用 =====")
        print(f"[Planner] 进入 run, trace_callback 是否为 None: {trace_callback is None}")
        # 🚀 最高优先级：检测“记住患者”等指令（直接在原始 question 中检测）
        if re.search(r'记住患者|记录患者|追加患者|补充患者', question):
            print(f"[Planner] 检测到患者操作指令，使用 patient 工具")
            return {"question": question, "tools": ["patient"]}

        # 提取当前问题（去除对话历史）
        current_question = question
        if "当前问题：" in question:
            current_question = question.split("当前问题：")[-1].strip()
        print(f"[Planner] 当前问题: {current_question}")

        # 检测身份声明
        if re.match(r'^用户[：:]', current_question.strip()):
            print(f"[Planner] 身份声明，不触发工具")
            return {"question": question, "tools": []}

        # 检测评估表生成
        if re.search(r'生成评估表|生成病历|生成档案|生成记录', current_question):
            print(f"[Planner] 检测到 report 关键词，强制使用 report 工具")
            return {"question": question, "tools": ["report"]}

        # 检测审批关键词
        approval_keywords = ["审批", "待审批", "驳回", "通过列表", "已通过", "已驳回", "全部列表", "审批通过", "驳回列表"]
        if any(kw in current_question for kw in approval_keywords):
            print(f"[Planner] 检测到审批指令，只使用 approval 工具")
            return {"question": question, "tools": ["approval"]}

        # 先运行规则
        tools = self.rule_based(question)
        has_patient = "patient" in tools
        has_report = "report" in tools

        # 如果规则匹配到了 report，直接返回（不经过 LLM 重新规划）
        if has_report:
            print(f"[Planner] 检测到 report 关键词，强制使用: {tools}")
            return {"question": question, "tools": tools}

        # LLM 重新规划
        if self.use_llm and (len(tools) == 0 or len(tools) > 3):
            print(f"[Planner] rule result {tools} -> using LLM")
            tools = self.llm_planner.select_tools(question)
            if not tools:
                tools = ["drug"]
                print("[Planner] LLM returned empty, fallback to drug")
            # 如果原规则中有 patient，但 LLM 去掉了，强制加回
            if has_patient and "patient" not in tools:
                tools.append("patient")
                print("[Planner] LLM removed patient, re-adding it")

        if trace_callback:
            print("[Planner] 准备调用 trace_callback")
            trace_callback("planner", {
                "question": question,
                "rule_result": tools,
                "used_llm": self.use_llm and (len(tools) == 0 or len(tools) > 3),
                "final_tools": tools
            })
            print("[Planner] trace_callback 调用完成")
        else:
            print("[Planner] trace_callback 为 None，跳过")

        return {"question": question, "tools": tools}