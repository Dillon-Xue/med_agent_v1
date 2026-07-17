import os
import re
from agents.llm_planner import LLMPlanner
from utils.response import mask_sensitive
from utils.config import get_llm_client
import logging
logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, use_llm: bool = True, trace_callback=None, specialty: str = "general"):
        self.specialty = specialty
        self.rules = {
            "drug": ["成分", "说明书", "副作用", "禁忌", "用法", "剂量", "不良反应", "功能"],
            "guideline": ["指南", "治疗", "推荐", "临床", "诊疗"],
            "literature": ["论文", "研究", "文献", "机制", "试验"],
            "risk": ["相互作用", "风险", "冲突", "合用", "联合用药", "配伍", "同服", "一起用", "联用"],
            "rag": ["特殊人群", "研发方案", "立项", "处方筛选", "药学研究"],
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

    def _extract_patient_name_from_history(self, full_question: str) -> str:
        """从完整问题的用户消息中提取患者姓名"""
        user_messages = re.findall(r'user:\s*(.*?)(?=\n(?:user:|assistant:|$))', full_question, re.DOTALL)
        combined_user_text = " ".join(user_messages)
        patterns = [
            r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})',
            r'([\u4e00-\u9fa5]{2,4})\s*的(?:信息|档案|病历|评估|报告)',
            r'(?:记住患者|记录患者|追加患者|补充患者)\s*([\u4e00-\u9fa5]{2,4})'
        ]
        for pattern in patterns:
            match = re.search(pattern, combined_user_text)
            if match:
                name = match.group(1)
                invalid_names = ["信息", "患者", "查询", "档案", "病历", "评估", "报告", "生成"]
                if name not in invalid_names and len(name) >= 2:
                    return name
        return ""

    def _extract_name_from_text(self, text: str) -> str:
        """从任意文本中提取患者姓名（用于当前问题）"""
        patterns = [
            r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})',
            r'([\u4e00-\u9fa5]{2,4})\s*的(?:信息|档案|病历|评估|报告)',
            r'(?:记住患者|记录患者|追加患者|补充患者)\s*([\u4e00-\u9fa5]{2,4})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1)
                invalid_names = ["信息", "患者", "查询", "档案", "病历", "评估", "报告", "生成"]
                if name not in invalid_names and len(name) >= 2:
                    return name
        return ""

    def rule_based(self, question: str) -> list:
        tools = set()
        stripped = question.strip()

        # 科室专属规则
        if self.specialty == "cardiology":
            tools.add("guideline")
            tools.add("drug")
            if any(kw in question for kw in ["支架", "搭桥", "心衰", "心肌梗死"]):
                tools.add("literature")
        elif self.specialty == "pharmacy":
            tools.add("drug")
            if any(kw in question for kw in ["剂量", "用法", "儿童", "老年人"]):
                tools.add("guideline")

        # 检测 patient 相关关键词
        if re.search(r'患者|记住|记录|档案|病历|信息|追加|补充|病号|增加', question):
            tools.add("patient")

        # 检测 report 相关关键词
        if re.search(r'评估表|生成报告|生成评估表|评估报告|生成病历|生成档案|生成记录', question):
            tools.add("report")

        # 匹配其他工具的关键词（跳过 patient 和 report）
        for tool, keywords in self.rules.items():
            if tool in ["patient", "report"]:
                continue
            for kw in keywords:
                if kw in question:
                    tools.add(tool)

        # 默认工具
        if not tools:
            return ["drug", "guideline", "literature", "risk"]

        # 如果只有 patient（纯关键词触发），追加默认工具
        if tools == {"patient"}:
            tools.update(["drug", "guideline", "literature", "risk"])

        return list(tools)

    def run(self, question: str, trace_callback=None) -> dict:
        logger.debug(f"[Planner] ===== run 被调用 =====")
        logger.debug(f"[Planner] 进入 run, trace_callback 是否为 None: {trace_callback is None}")

        # 提取当前问题（去除对话历史）
        current_question = question
        if "当前问题：" in question:
            current_question = question.split("当前问题：")[-1].strip()
        logger.info(f"[Planner] 当前问题: {mask_sensitive(current_question)}")

        # 最高优先级：检测“记住患者”等指令（基于当前问题）
        if re.search(r'记住患者|记录患者|追加患者|补充患者', current_question):
            logger.debug(f"[Planner] 检测到患者操作指令，使用 patient 工具")
            return {"question": question, "tools": ["patient"]}

        # 检测身份声明
        if re.match(r'^用户[：:]', current_question.strip()):
            logger.debug(f"[Planner] 身份声明，不触发工具")
            return {"question": question, "tools": []}

        # 检测评估表生成
        if re.search(r'生成评估表|生成病历|生成档案|生成记录', current_question):
            logger.debug(f"[Planner] 检测到 report 关键词，强制使用 report 工具")
            return {"question": question, "tools": ["report"]}

        # 检测审批关键词
        approval_keywords = ["审批", "待审批", "驳回", "通过列表", "已通过", "已驳回", "全部列表", "审批通过", "驳回列表"]
        if any(kw in current_question for kw in approval_keywords):
            logger.debug(f"[Planner] 检测到审批指令，只使用 approval 工具")
            return {"question": question, "tools": ["approval"]}

        # 提取历史中的患者姓名（基于完整问题，只提取 user 部分）
        historical_patient = self._extract_patient_name_from_history(question)
        # 提取当前问题中的患者姓名
        current_patient = self._extract_name_from_text(current_question)
        effective_patient = historical_patient or current_patient

        # 基于当前问题运行规则（避免误匹配）
        tools = self.rule_based(current_question)
        has_report = "report" in tools

        # 如果规则匹配到了 report，直接返回
        if has_report:
            logger.debug(f"[Planner] 检测到 report 关键词，强制使用: {tools}")
            return {"question": question, "tools": tools}

        # LLM 重新规划（基于当前问题）
        if self.use_llm and (len(tools) == 0 or len(tools) > 3):
            logger.debug(f"[Planner] rule result {tools} -> using LLM")
            tools = self.llm_planner.select_tools(current_question)
            if not tools:
                tools = ["drug"]
                logger.info("[Planner] LLM returned empty, fallback to drug")
            # 如果有效患者存在，但 LLM 去掉了 patient，强制加回
            if effective_patient and "patient" not in tools:
                tools.append("patient")
                logger.info(f"[Planner] LLM removed patient, re-adding it due to effective patient")

        # 最终根据 effective_patient 决定 patient 是否保留
        if effective_patient:
            if "patient" not in tools:
                tools.append("patient")
                logger.info(f"[Planner] 根据有效患者（{effective_patient}）添加 patient 工具")
        else:
            if "patient" in tools:
                tools.remove("patient")
                logger.info("[Planner] 强制移除 patient（无有效患者信息）")

        if trace_callback:
            logger.debug("[Planner] 准备调用 trace_callback")
            trace_callback("planner", {
                "question": question,
                "rule_result": tools,
                "used_llm": self.use_llm and (len(tools) == 0 or len(tools) > 3),
                "final_tools": tools
            })
            logger.debug("[Planner] trace_callback 调用完成")
        else:
            logger.debug("[Planner] trace_callback 为 None，跳过")

        return {"question": question, "tools": tools}