import os
import time
from openai import OpenAI
from langchain_chroma import Chroma
from utils.response import build_response
from utils.embeddings import DashscopeEmbeddings
from utils.config import get_llm_client

class RAGTool:
    def __init__(self, base_dir: str, api_key: str):
        self.base_dir = base_dir
        self.api_key = api_key
        # 🆕 使用统一客户端工厂
        self.client, self.model = get_llm_client(api_key, timeout=60.0)
        print(f"[RAGTool] Using model: {self.model}")
        self.db_path = os.path.join(base_dir, "vector_db", "rag")
        self.embeddings = DashscopeEmbeddings()
        self.vectordb = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        self._last_trace = []

    def trace(self, stage: str, data: dict):
        self._last_trace.append({
            "tool": "rag",
            "stage": stage,
            "data": data,
            "ts": time.time()
        })

    def get_trace(self):
        return self._last_trace

    def clear_trace(self):
        self._last_trace = []

    def rewrite(self, q: str) -> str:
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"改写为医学检索关键词（短句）：{q}"}],
                temperature=0,
                timeout=15.0
            )
            rq = resp.choices[0].message.content.strip()
        except Exception as e:
            rq = q
            self.trace("rewrite_error", {"error": str(e)})
        self.trace("rewrite", {"input": q, "output": rq, "latency": round(time.time() - t0, 3)})
        return rq

    def retrieve(self, query: str):
        t0 = time.time()
        try:
            docs = self.retrieve_with_optimization(query, k=5, use_llm_rerank=False)
        except Exception as e:
            docs = []
            self.trace("retrieve_error", {"error": str(e)})
        self.trace("retrieve", {"query": query, "count": len(docs), "latency": round(time.time() - t0, 3), "samples": [d.page_content[:200] for d in docs]})
        return docs

    def rerank(self, docs):
        t0 = time.time()
        top_docs = docs[:3]
        self.trace("rerank", {"input": len(docs), "output": len(top_docs), "latency": round(time.time() - t0, 3)})
        return top_docs

    def run(self, query: str):
        self.clear_trace()
        t0 = time.time()
        self.trace("run_start", {"query": query})
        rq = self.rewrite(query)
        docs = self.retrieve(rq)
        final_docs = self.rerank(docs)
        context = "\n\n".join([d.page_content for d in final_docs])
        self.trace("context", {"len": len(context), "preview": context[:300]})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "严格医学RAG，只能使用提供资料"},
                    {"role": "user", "content": f"资料：\n{context}\n\n问题：\n{query}"}
                ],
                temperature=0.2,
                timeout=30.0
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"LLM调用失败: {str(e)}"
            self.trace("llm_error", {"error": str(e)})

        latency = round(time.time() - t0, 3)
        self.trace("final_answer", {"answer": answer, "latency": latency})
        return build_response(
            answer=answer,
            source="rag",
            debug={"rewrite": rq, "retrieved": len(docs), "used": len(final_docs), "context_length": len(context), "latency": latency},
            trace=self.get_trace()
        ) 
