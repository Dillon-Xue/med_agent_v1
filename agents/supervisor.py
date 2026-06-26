from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils.config import get_llm_client  # 🆕 导入统一客户端工厂

load_dotenv()

app = FastAPI()


class Supervisor:
    def __init__(self):
        """初始化 Supervisor，使用统一客户端工厂"""
        # 🆕 使用 get_llm_client 创建客户端
        api_key = os.getenv("DASHSCOPE_API_KEY")
        self.client, self.model = get_llm_client(api_key)
        print(f"[Supervisor] Using model: {self.model}")

    def route(self, question: str) -> dict:
        prompt = f"""
    请判断以下医学问题应该由哪个科室处理，返回JSON格式。

    问题：{question}

    判断规则：
    - 心脏病、高血压、冠心病、心律失常 → {{"primary": "cardiology", "secondary": []}}
    - 用药方案、药物相互作用、药品说明书 → {{"primary": "pharmacy", "secondary": []}}
    - 既涉及心脏又涉及用药 → {{"primary": "cardiology", "secondary": ["pharmacy"]}}
    - 其他或不确定 → {{"primary": "general", "secondary": []}}

    只输出JSON，不要其他内容。
    """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            import json
            data = json.loads(content)
            primary = data.get("primary", "general")
            secondary = data.get("secondary", [])
            if not isinstance(secondary, list):
                secondary = []
            secondary = [s for s in secondary if s != primary]
            return {"primary": primary, "secondary": secondary}
        except Exception as e:
            print(f"[Supervisor] Route failed: {e}")
            return {"primary": "general", "secondary": []}


# 如果这个文件也作为独立的 FastAPI 服务运行
supervisor = Supervisor()

@app.get("/route")
def route_question(question: str):
    agent = supervisor.route(question)
    return {"agent": agent}