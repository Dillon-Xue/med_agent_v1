from openai import OpenAI
import os
from utils.config import get_llm_client
from utils.config import get_response_mode, get_response_max_length
import logging
import re
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
            # 只保留带精确来源标注的工具结果
            if not re.search(r"[(（]来源:.*?\.pdf.*?页", answer):
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
        # 历史参考病例不再强制输出

        system_prompt = f"""{base_prompt} {history_rule}
        回答必须严格遵循以下纯文本格式，不要使用任何markdown语法。
        输出格式要求：
        1. 第一行：核心结论（一句话，尽可能简洁）。
        2. 空一行。
        3. 关键信息用 "- " 开头的列表项列出，但同一来源的同类信息必须合并为一句话，禁止逐条拆分后重复标注来源。
        4. 仅当工具结果原文明确包含"警示"、"警告"、"注意"等字样时，才在对应内容前加 "⚠️ " 并单独成行。普通禁忌、副作用、用法用量等信息不需要加 ⚠️。
        5. 不同主题之间用空行分隔。
        6. 仅当有多个工具返回了有效信息时，最后一行才输出"总结建议"。
        注意：不要使用任何加粗符号，不要使用星号。

        【来源标注的硬性规则】
        1. 你只能输出工具返回结果中明确出现的信息，禁止任何推理、解释、延伸或补充。
        2. 禁止解释成分的功效作用（如"山楂消食化积"等），除非工具返回结果原文中明确写了这些内容。
        3. 禁止输出药物联用建议、饮食禁忌、注意事项等，除非工具返回结果原文中明确写了这些内容。
        4. 如果工具返回结果中已包含精确来源标注 (来源: xxx.pdf, 第 N 页，第 M 段)，你必须在对应信息末尾原样复制该标注。
        5. 同一文档同一页的多条信息必须合并为一句表述，仅在该句末尾标注一次来源。禁止把每条信息单独成行并重复标注来源。
        6. 如果工具返回结果中未包含精确来源标注，则使用工具来源格式：【来源：DRUG工具】、【来源：GUIDELINE工具】、【来源：LITERATURE工具】等。
        7. 严禁使用任何指南名称、共识名称、期刊名称或权威机构名称作为来源。
        8. 绝对禁止将精确来源格式替换为粗粒度格式。
        9. 只有一个工具返回了有效信息时，禁止输出"总结建议"。
        10. 如果有多个工具返回有效信息，"总结建议"必须简洁概括工具结果，禁止加入工具结果外的任何知识。
        11. 如果有工具没有返回有效信息，则忽略该工具返回，不能展示成"现有资料中未提供相关信息"。
        12. 禁止出现任何不相干的警示信息。
        13. 绝对禁止输出"资料中未提供xxx的其他信息"等无效提示。
        14. 绝对禁止将精确来源格式替换为粗粒度格式。
        15. 绝对禁止在回答中输出"【来源：XX工具】"标签。
        16. 只有一个工具返回有效信息时，绝对禁止输出"总结建议"。
        17. 绝对禁止输出模型推理内容，如"来源: 模型推理，请核实"。
        18. 绝对禁止对同一内容重复输出多行"⚠️"警示。
        """

        user_prompt = f"""{length_instruction}

    对话历史及当前问题：
    {cleaned_question}

    各工具返回结果：
    {combined}

    请严格按照上述纯文本格式要求给出综合回答，不要使用任何 markdown 语法。
    要求：同类信息必须合并为一句简洁表述，只在末尾标注一次来源。禁止逐条拆分重复标注。"""

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

        # 历史参考病例不再强制添加

        if trace_callback:
            trace_callback("synthesizer", {
                "status": "complete",
                "answer_preview": answer[:300]
            })

        specialty_display = {"cardiology": "心外科", "pharmacy": "药剂科", "general": "全科"}.get(self.specialty, "全科")
        answer = f"【{specialty_display}观点】\n{answer}"

        # 不在最终回答中附加历史参考病例


        # 后处理过滤
        skip_phrases = [
            "资料中未提供", "未提供相关", "未找到相关", "现有资料不足",
            "无法提供", "没有相关信息", "未提及相关", "无法回答",
            "模型推理", "请核实", "来源: 模型推理"
        ]
        lines_ans = answer.split('\n')
        filtered = []
        for line_a in lines_ans:
            s = line_a.strip()
            if not s:
                filtered.append(line_a)
                continue
            if s.startswith('【来源：') and s.endswith('工具】'):
                continue
            if '来源: 模型推理' in s or '来源：模型推理' in s:
                continue
            if s.startswith('【历史参考病例】') or s.startswith('【历史参考】'):
                continue
            if any(p in s for p in skip_phrases) and '⚠️' in s:
                continue
            if any(p in s for p in skip_phrases) and len(s) < 50:
                continue
            filtered.append(line_a)
        answer = '\n'.join(filtered)
        answer = re.sub(r'\n*总结建议[。：]?\s*$', '', answer).strip()
        answer = re.sub(r'总结建议[。：]?\s*\n*', '', answer).strip()

        return answer