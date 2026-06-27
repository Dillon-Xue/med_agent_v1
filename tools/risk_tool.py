from tools.base_tool import BaseTool
from utils.response import build_response
from utils.embeddings import DashscopeEmbeddings
import time, logging
logger = logging.getLogger(__name__)

class RiskTool(BaseTool):
    def __init__(self, base_dir, api_key):
        super().__init__(base_dir, "risk", api_key)
        self.embeddings = DashscopeEmbeddings()
        self.load_db(self.embeddings)
        logger.info(f"VECTOR READ PATH: {self.db_path}")
        self._trace = []

    def trace(self, stage: str, data: dict):
        self._trace.append({
            "tool": "risk",
            "stage": stage,
            "ts": time.time(),
            "data": data
        })

    def get_trace(self):
        return self._trace

    def clear_trace(self):
        self._trace = []

    def run(self, query: str):
        self.clear_trace()
        t0 = time.time()
        self.trace("run_start", {"query": query})

        try:
            docs = self.retrieve_with_optimization(query, k=5, use_llm_rerank=False)
        except Exception as e:
            self.trace("error", {"stage": "retrieve", "error": str(e)})
            return build_response(
                answer=f"检索失败: {str(e)}",
                source="risk",
                debug={"error": str(e)},
                trace=self.get_trace()
            )

        self.trace("retrieve", {"count": len(docs), "samples": [d.page_content[:200] for d in docs]})
        context = "\n\n".join([d.page_content for d in docs])
        self.trace("context", {"length": len(context), "preview": context[:300]})

        prompt = f"""
你是药物风险分析专家，专注于药物相互作用、不良反应、禁忌症。

只能基于给定资料回答，不允许编造。

资料：
{context}

问题：
{query}
"""
        messages = [
            {"role": "system", "content": "严格药物风险分析，仅依据资料"},
            {"role": "user", "content": prompt}
        ]
        #answer = self._safe_llm_call(messages)
        answer = self._safe_llm_call(messages, fallback_context=context)
        latency = round(time.time() - t0, 3)
        self.trace("final_answer", {"answer": answer, "latency": latency})

        return build_response(
            answer=answer,
            source="risk",
            debug={"retrieved": len(docs), "latency": latency},
            trace=self.get_trace()
        ) 
