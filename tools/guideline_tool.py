from tools.base_tool import BaseTool
from utils.response import build_response
from utils.embeddings import DashscopeEmbeddings
import time

class GuidelineTool(BaseTool):
    def __init__(self, base_dir, api_key):
        super().__init__(base_dir, "guideline", api_key)
        self.embeddings = DashscopeEmbeddings()
        self.load_db(self.embeddings)
        self._trace = []

    def trace(self, stage: str, data: dict):
        self._trace.append({
            "tool": "guideline",
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
                source="guideline",
                debug={"error": str(e)},
                trace=self.get_trace()
            )

        self.trace("retrieve", {"count": len(docs), "samples": [d.page_content[:200] for d in docs]})
        context, citations = self._build_cited_context(docs)
        self.trace("context", {"length": len(context), "preview": context[:300], "citations": citations})

        messages = [{
            "role": "user",
            "content": f"""
基于临床指南回答问题。

只能依据给定资料回答，
不允许编造。

资料：
{context}

问题：
{query}

重要要求：
1. 每条回答内容末尾必须标注精确来源，格式为：(来源: xxx.pdf, 第 N 页，第 M 段)。如果资料中涉及图表信息，格式为：(来源: xxx.pdf, 第 N 页, 图X, 第 M 段) 或 (来源: xxx.pdf, 第 N 页, 表X, 第 M 段)。
2. 来源信息必须直接从资料中复制，不得自行编造
3. 如果某条信息无法对应到资料中的来源，标注：(来源: 模型推理，请核实)
"""
        }]
        answer = self._safe_llm_call(messages, fallback_context=context)
        latency = round(time.time() - t0, 3)
        self.trace("final_answer", {"answer": answer, "latency": latency})

        return build_response(
            answer=answer,
            source="guideline",
            debug={"retrieved": len(docs), "latency": latency, "citations": citations},
            trace=self.get_trace()
        )