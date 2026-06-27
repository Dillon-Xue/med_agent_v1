import json, logging
from openai import OpenAI
from utils.config import get_llm_client

logger = logging.getLogger(__name__)


class LLMPlanner:
    def __init__(self, api_key):
        # 🆕 使用统一客户端工厂
        self.client, self.model = get_llm_client(api_key)
        logger.debug(f"[LLMPlanner] Using model: {self.model}")

    def select_tools(self, question: str) -> list:
        tools_desc = {
            "drug": "药品说明书、成分、副作用、用法用量",
            "guideline": "临床指南、治疗推荐、诊疗规范",
            "literature": "医学论文、最新研究、作用机制",
            "risk": "药物相互作用、不良反应、禁忌症、风险"
        }
        prompt = f"""用户问题：{question}
可选工具及其功能：{tools_desc}
请返回需要调用的工具列表（JSON格式，例如["drug","risk"]），只输出JSON，不要解释。"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=10
            )
            content = resp.choices[0].message.content.strip()
            tools = json.loads(content)
            if isinstance(tools, list) and all(isinstance(t, str) for t in tools):
                return tools
        except Exception as e:
            logger.error(f"[LLMPlanner] error: {e}")
        return ["drug", "guideline", "literature", "risk"]  # fallback
