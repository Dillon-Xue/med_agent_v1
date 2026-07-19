from utils.config import get_llm_client, get_response_mode, get_response_max_length
import os, logging, re
logger = logging.getLogger(__name__)

class Aggregator:
    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        self.client, self.model = get_llm_client(api_key)

    def run(self, question: str, agent_results: dict) -> str:
        if len(agent_results) == 1:
            return list(agent_results.values())[0]

        sections = []
        for specialty, answer in agent_results.items():
            sections.append(f"【{self._get_display_name(specialty)} 观点】\n{answer}")

        joined = "\n\n".join(sections)

        # 获取长度控制配置
        mode = get_response_mode()
        max_len = get_response_max_length()
        length_instruction = {
            "concise": "回答尽量简洁，只输出核心共识和关键建议，严禁展开各科室的详细论述。绝对禁止在回答中输出字数统计，如\"（字数：XXX）\"等。",
            "balanced": "回答控制在 500-800 字，包含核心结论和必要支撑信息。绝对禁止在回答中输出字数统计。",
            "detailed": "回答可详尽展开，不限制长度，确保信息完整。绝对禁止在回答中输出字数统计。"
        }.get(mode, "")

        prompt = f"""
        {length_instruction}

        问题：{question}

        以下是从各科室收集的专家意见：

        {joined}

        请综合上述意见，给出最终的综合建议。要求：
        1. 开头用一句话概括共识，但是不能出现共识两个字，需使用"各科室信息汇总："作为开头。
        2. 用 "- " 列表列出关键建议，每条建议必须同时标注科室来源和精确来源，格式示例：内容...【心外科】【药剂科】（来源: xxx.pdf, 第 N 页，第 M 段）。严禁遗漏科室来源，严禁仅使用科室名称代替精确来源。
        3. 如有分歧，明确说明各科室的分歧内容，如果没有分歧则可以不体现该项内容。
        4. 最后给出患者可以立即执行的具体行动建议。
        5. 每一项完整的内容输出后，都要加一个空行，让回答看起来有层次。
        6. 对特别重要的警示、绝对禁忌或紧急行动，必须在该条前加"⚠️ "前缀。绝对禁止使用 ** 加粗符号，不要使用任何 markdown 语法。
        7. 绝对禁止输出"（字数：XXX）"、"共XXX字"等任何字数统计信息。
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = resp.choices[0].message.content
        # 后处理：移除字数统计、markdown加粗、粗粒度工具来源标签，保留科室标签和精确来源
        lines = answer.split("\n")
        filtered = []
        for line in lines:
            s = line.strip()
            if "字数：" in s or ("共" in s and "字" in s and any(c.isdigit() for c in s)):
                if s.startswith("（") and s.endswith("）") and "字" in s:
                    continue
            line = re.sub(r'\*\*', '', line)
            # 只移除【来源：XX工具】这种粗粒度格式，保留【心外科】【药剂科】等科室标签
            line = re.sub(r'【来源：[^】]+】', '', line)
            # 格式优化：将（【科室】来源:）改为 【科室】（来源:），使层次更清晰
            line = re.sub(r'（(【[^】]+】+)来源[：:]', r'\1（来源:', line)
            filtered.append(line)
        answer = "\n".join(filtered)
        return answer

    def _get_display_name(self, specialty: str) -> str:
        mapping = {"cardiology": "心外科", "pharmacy": "药剂科", "general": "全科"}
        return mapping.get(specialty, specialty)