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

    def route(self, question: str) -> str:
        """根据问题路由到合适的 Agent"""
        prompt = f"""
请判断以下医学问题应该由哪个科室处理，只输出科室名称（cardiology/pharmacy/general）。

问题：{question}

判断规则：
- 心脏病、高血压、冠心病、心律失常 → cardiology
- 用药方案、药物相互作用、药品说明书 → pharmacy
- 其他或不确定 → general

只输出科室名称，不要其他内容。
"""
        try:
            # 🆕 使用 self.model 而不是硬编码
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            agent_name = resp.choices[0].message.content.strip().lower()
            print(f"[Supervisor] Routed to: {agent_name}")
            return agent_name if agent_name in ["cardiology", "pharmacy", "general"] else "general"
        except Exception as e:
            print(f"[Supervisor] Route failed: {e}")
            return "general"


# 如果这个文件也作为独立的 FastAPI 服务运行
supervisor = Supervisor()

@app.get("/route")
def route_question(question: str):
    agent = supervisor.route(question)
    return {"agent": agent}