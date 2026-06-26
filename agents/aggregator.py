from utils.config import get_llm_client, get_response_mode, get_response_max_length
import os

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
            "concise": f"⚠️ 强制要求：综合回答总字数必须控制在 {max_len} 字以内（约 8-10 句话）。只输出核心共识和关键建议，严禁展开各科室的详细论述。",
            "balanced": "回答控制在 500-800 字，包含核心结论和必要支撑信息。",
            "detailed": "回答可详尽展开，不限制长度，确保信息完整。"
        }.get(mode, "")

        prompt = f"""
        {length_instruction}

        问题：{question}

        以下是从各科室收集的专家意见：

        {joined}

        请综合上述意见，给出最终的综合建议。要求：
        1. 开头用一句话概括共识，但是不能出现共识两个字，需使用"各科室信息汇总："作为开头。
        2. 用 "- " 列表列出各科室的关键建议，每条标注科室名称（如【心外科】）。
        3. 如有分歧，明确说明各科室的分歧内容，如果没有分歧则可以不体现该项内容。
        4. 最后给出患者可以立即执行的具体行动建议。
        5. 每一项完整的内容输出后，都要加一个空行，让回答看起来有层次。
        6. 对特别重要的警示、绝对禁忌或紧急行动，必须在该条前加“⚠️ ”前缀，并用 ** 将整条内容加粗，例如：**⚠️ 绝对禁忌：** 布洛芬等NSAIDs。格式必须为 **⚠️ 内容：**。
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content

    def _get_display_name(self, specialty: str) -> str:
        mapping = {"cardiology": "心外科", "pharmacy": "药剂科", "general": "全科"}
        return mapping.get(specialty, specialty)