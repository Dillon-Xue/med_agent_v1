import os, time, logging
from openai import OpenAI
from langchain_chroma import Chroma
from utils.response import build_response
from utils.embeddings import DashscopeEmbeddings
from utils.config import get_llm_client
logger = logging.getLogger(__name__)

class RAGTool:
    def __init__(self, base_dir: str, api_key: str):
        self.base_dir = base_dir
        self.api_key = api_key
        self.client, self.model = get_llm_client(api_key, timeout=60.0)
        logger.info(f"[RAGTool] Using model: {self.model}")
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

    def _build_cited_context(self, docs):
        parts = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            source = meta.get("source", "未知来源")
            page = meta.get("page", "?")
            chunk_idx = meta.get("chunk_index", "?")
            parts.append(f"[{i}] {d.page_content}\n(来源: {source}, 第 {page} 页，第 {chunk_idx} 段)")
        return "\n\n".join(parts)

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
            docs = self.vectordb.similarity_search(query, k=10)
        except Exception as e:
            docs = []
            self.trace("retrieve_error", {"error": str(e)})
        self.trace("retrieve", {"query": query, "count": len(docs), "latency": round(time.time() - t0, 3), "samples": [d.page_content[:200] for d in docs]})
        return docs

    def rerank(self, docs):
        t0 = time.time()
        top_docs = docs[:5]
        self.trace("rerank", {"input": len(docs), "output": len(top_docs), "latency": round(time.time() - t0, 3)})
        return top_docs

    def run(self, query: str):
        self.clear_trace()
        t0 = time.time()
        self.trace("run_start", {"query": query})

        # 对话历史剥离
        tool_question = query
        if "当前问题：" in query:
            tool_question = query.rsplit("当前问题：", 1)[-1].strip()
        elif query.startswith("患者信息：") and "。问题：" in query:
            tool_question = query.split("。问题：", 1)[-1].strip()

        rq = self.rewrite(tool_question)
        docs = self.retrieve(rq)
        final_docs = self.rerank(docs)
        context = self._build_cited_context(final_docs)
        self.trace("context", {"len": len(context), "preview": context[:300]})

        system_prompt = """你是一位严格基于资料的医学助手。
【核心规则】
1. 必须只使用提供的资料回答问题，禁止输出资料外的任何推理、猜测或补充。
2. 每条事实后必须标注精确来源，格式：(来源: xxx.pdf, 第 N 页，第 M 段)。
3. 如果资料中没有直接答案，回答"根据现有资料，未找到相关信息。"，禁止编造。
4. 禁止输出"总结建议"、"【来源：XX工具】"等标签。
5. 只回答与问题直接相关的内容。"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"资料：\n{context}\n\n问题：\n{tool_question}"}
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