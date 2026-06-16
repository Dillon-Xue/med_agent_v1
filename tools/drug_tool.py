from tools.base_tool import BaseTool
from utils.response import build_response
from utils.embeddings import DashscopeEmbeddings
import time

class DrugTool(BaseTool):
    def __init__(self, base_dir, api_key):
        super().__init__(base_dir, "drug", api_key)
        self.embeddings = DashscopeEmbeddings()
        self.load_db(self.embeddings)
        print("VECTOR READ PATH:", self.db_path)
        self._trace = []

    def trace(self, stage: str, data: dict):
        self._trace.append({
            "tool": "drug",
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

        # 检索（增加异常捕获）
        try:
            docs = self.vectordb.similarity_search(query, k=5)
        except Exception as e:
            self.trace("error", {"stage": "retrieve", "error": str(e)})
            return build_response(
                answer=f"检索失败: {str(e)}",
                source="drug",
                debug={"error": str(e)},
                trace=self.get_trace()
            )

        self.trace("retrieve", {"count": len(docs), "samples": [d.page_content[:200] for d in docs]})
        context = "\n\n".join([d.page_content for d in docs])
        self.trace("context", {"length": len(context), "preview": context[:300]})

        prompt = f"""
你是医学药物分析专家。

只能基于给定资料回答，不允许编造。

资料：
{context}

问题：
{query}
"""
        messages = [
            {"role": "system", "content": "严格医学药物分析"},
            {"role": "user", "content": prompt}
        ]
        #answer = self._safe_llm_call(messages)
        answer = self._safe_llm_call(messages, fallback_context=context)
        latency = round(time.time() - t0, 3)
        self.trace("final_answer", {"answer": answer, "latency": latency})

        return build_response(
            answer=answer,
            source="drug",
            debug={"retrieved": len(docs), "latency": latency},
            trace=self.get_trace()
        ) 
