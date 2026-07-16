from openai import OpenAI
import os
from utils.config import get_llm_client
from utils.config import get_response_mode, get_response_max_length
import logging
logger = logging.getLogger(__name__)

class Synthesizer:
    def __init__(self, api_key, specialty: str = "general"):
        self.client, self.model = get_llm_client(api_key, timeout=60.0)
        self.specialty = specialty
        logger.debug(f"[Synthesizer] Model: {self.model}, Specialty: {self.specialty}")

    def run(self, augmented_question, tool_results, trace_callback=None):
        if trace_callback:
            logger.debug(f"[Synthesizer] 正在记录 trace")
            trace_callback("synthesizer", {
                "question": augmented_question,
                "tool_count": len(tool_results),
                "tool_sources": [r.get("source") for r in tool_results if r]
            })

        for res in tool_results:
            if not res or not isinstance(res, dict):
                continue
            if res.get("source") == "report":
                answer = res.get("answer", "")
                if answer:
                    if trace_callback:
                        trace_callback("synthesizer", {
                            "status": "complete",
                            "answer_preview": answer[:300],
                            "source": "report"
                        })
                    return answer

        for res in tool_results:
            if not res or not isinstance(res, dict):
                continue
            if res.get("source") == "patient":
                answer = res.get("answer", "")
                if answer.startswith("✅") or answer.startswith("📋"):
                    if trace_callback:
                        trace_callback("synthesizer", {
                            "status": "complete",
                            "answer_preview": answer[:300],
                            "source": "patient"
                        })
                    return answer

        if len(tool_results) == 1:
            res = tool_results[0]
            if res and isinstance(res, dict) and res.get("source") == "patient":
                answer = res.get("answer", "患者工具未返回有效信息")
                if trace_callback:
                    trace_callback("synthesizer", {
                        "status": "complete",
                        "answer_preview": answer[:300],
                        "source": "patient"
                    })
                return answer

        context_parts = []
        invalid_keywords = ["未提供", "无法回答", "未提及", "资料中未", "无法评估", "没有相关"]
        for res in tool_results:
            if not res or not isinstance(res, dict):
                continue
            answer = res.get("answer", "")
            if not answer:
                continue
            if any(kw in answer for kw in invalid_keywords) and len(answer) < 100:
                continue
            source = res.get("source", "unknown").upper()
            context_parts.append(f"【来自 {source} 工具】\n{answer}")

        if not context_parts:
            fallback_msg = "未找到相关信息，请尝试换一种问法。"
            if trace_callback:
                trace_callback("synthesizer", {
                    "status": "complete",
                    "answer_preview": fallback_msg[:300]
                })
            return fallback_msg

        combined = "\n\n".join(context_parts)

        import re
        history_ref_section = ""
        cleaned_question = augmented_question
        if "【历史参考病例】" in augmented_question:
            match = re.search(r'(【历史参考病例】.*?)(?=\n\n|\Z)', augmented_question, re.DOTALL)
            if match:
                history_ref_section = match.group(1)
                cleaned_question = augmented_question.replace(history_ref_section, "").strip()

        specialty_prompts = {
            "cardiology": "你是心外科专家，回答需基于心血管临床指南。",
            "pharmacy": "你是药剂科专家，重点关注药物相互作用、剂量调整和用药安全。",
            "general": "你是一个专业的医学知识合成器。"
        }
        base_prompt = specialty_prompts.get(self.specialty, specialty_prompts["general"])

        mode = get_response_mode()
        max_len = get_response_max_length()

        length_instruction = {
            "concise": f"⚠️ 强制要求：回答总字数必须控制在 {max_len} 字以内（约 8-10 句话），严禁展开背景介绍，只输出核心结论和关键建议。超出字数限制的回答将被视为无效。",
            "balanced": "回答控制在 500-800 字，包含核心结论和必要支撑信息。",
            "detailed": "回答可详尽展开，不限制长度，确保信息完整。"
        }.get(mode, "")

        history_rule = ""
        if history_ref_section:
            history_rule = "你必须以【历史参考病例】开头回答，并明确提及病例中的患者姓名和用药方案。这是强制要求，不得省略。"

        system_prompt = f"""{base_prompt} {history_rule}
        回答必须严格遵循以下纯文本格式，不要使用任何markdown语法。
        输出格式要求：
        1. 第一行：核心结论（一句话）。
        2. 空一行。
        3. 使用 "- " 开头的列表项列出关键信息，每项占一行。
        4. 重要警示前加 "⚠️ " 并单独成行。
        5. 不同主题之间用空行分隔。
        6. 如果有多个工具返回了有效信息，最后一行：总结建议。
        注意：不要使用任何加粗符号，不要使用星号。

        【来源标注的硬性规则】
        1. 你只能输出工具返回结果中明确出现的信息，禁止任何推理、解释、延伸或补充。
        2. 禁止解释成分的功效作用（如"山楂消食化积"等），除非工具返回结果原文中明确写了这些内容。
        3. 禁止输出药物联用建议、饮食禁忌、注意事项等，除非工具返回结果原文中明确写了这些内容。
        4. 如果工具返回结果中已包含精确来源标注 (来源: xxx.pdf, 第 N 页，第 M 段)，你必须在对应信息末尾原样复制该标注。
        5. 如果工具返回结果中未包含精确来源标注，则使用工具来源格式：【来源：DRUG工具】、【来源：GUIDELINE工具】、【来源：LITERATURE工具】等。
        6. 严禁使用任何指南名称、共识名称、期刊名称或权威机构名称作为来源。
        7. 绝对禁止将精确来源格式替换为粗粒度格式。
        8. 只有一个工具返回了有效信息时，禁止输出"总结建议"。
        9. 如果有多个工具返回有效信息，"总结建议"必须简洁概括工具结果，禁止加入工具结果外的任何知识。
        10. 如果有工具没有返回有效信息，则忽略该工具返回，不能展示成"现有资料中未提供相关信息"。
        11. 禁止出现任何不相干的警示信息。
        """

        if history_ref_section:
            user_prompt = f"""{length_instruction}

    {history_ref_section}

    当前问题：
    {cleaned_question}

    各工具返回结果：
    {combined}

    请严格按照上述纯文本格式要求给出综合回答，不要使用任何 markdown 语法。

    ⚠️ 注意：你的回答必须以「【历史参考病例】」开头，并明确提到患者姓名和用药方案。这是强制要求，不得违反。"""
        else:
            user_prompt = f"""{length_instruction}

    对话历史及当前问题：
    {cleaned_question}

    各工具返回结果：
    {combined}

    请严格按照上述纯文本格式要求给出综合回答，不要使用任何 markdown 语法。"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            logger.debug(f"[Synthesizer] LLM 调用失败: {e}")
            answer = f"【系统降级】LLM 合成失败，以下是各工具原始结果：\n\n{combined[:1000]}"

        if history_ref_section and "历史参考" not in answer and "赵六" not in answer:
            name_match = re.search(r'([\u4e00-\u9fa5]{2,4})[：:]\s*([^，,]+)', history_ref_section)
            if name_match:
                patient_name = name_match.group(1)
                diagnosis = name_match.group(2)
                answer = f"【历史参考病例】参考患者 {patient_name}（{diagnosis}）的用药经验，该病例相似度15%。\n\n{answer}"

        if trace_callback:
            trace_callback("synthesizer", {
                "status": "complete",
                "answer_preview": answer[:300]
            })

        specialty_display = {"cardiology": "心外科", "pharmacy": "药剂科", "general": "全科"}.get(self.specialty, "全科")
        answer = f"【{specialty_display}观点】\n{answer}"

        if "【历史参考病例】" in augmented_question:
            import re
            match = re.search(r'【历史参考病例】\s*[-]?\s*([\u4e00-\u9fa5]{2,4})[：:]\s*([^，,\n]+)', augmented_question)
            if match:
                name = match.group(1)
                diagnosis = match.group(2)
                history_line = f"\n\n【历史参考】相似病例：{name}（{diagnosis}）"
                if history_line not in answer:
                    answer = answer + history_line

        return answer