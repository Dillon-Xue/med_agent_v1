from abc import ABC, abstractmethod
from openai import OpenAI
from langchain_chroma import Chroma
import os, logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APITimeoutError, APIConnectionError
from tools.retriever import HybridRetriever
from typing import List, Optional
from utils.config import get_llm_provider, get_ollama_base_url, get_ollama_model, get_llm_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.getenv("MED_AGENT_ROOT", os.getcwd())

class BaseTool(ABC):
    def __init__(self, base_dir, db_name, api_key):
        self.base_dir = PROJECT_ROOT
        self.db_path = os.path.join(self.base_dir, "vector_db", db_name)
        logger.debug(f"\n[BASE_TOOL] db_path = {self.db_path}")

        self.client, self.model = get_llm_client(api_key, timeout=60.0)
        logger.debug(f"[BASE_TOOL] Using model: {self.model}")

        self.vectordb = None
        self.retriever = None

    def load_db(self, embedding_fn):
        self.vectordb = Chroma(
            persist_directory=self.db_path,
            embedding_function=embedding_fn,
            collection_name="langchain"
        )
        self.retriever = HybridRetriever(self.vectordb)
        logger.info(f"[BASE_TOOL] Chroma loaded from {self.db_path}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, APIError)),
        reraise=True
    )
    def _llm_call_with_retry(self, messages, temperature=0.2, model=None):
        if model is None:
            model = self.model
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=30.0
        )
        return resp.choices[0].message.content

    def _safe_llm_call(self, messages, temperature=0.2, model=None, fallback_context=None):
        try:
            return self._llm_call_with_retry(messages, temperature, model)
        except Exception as e:
            logger.error(f"[ERROR] LLM call failed after retries: {e}")
            if fallback_context:
                return f"【系统降级】LLM 服务暂时不可用，以下是检索到的相关信息：\n\n{fallback_context[:500]}"
            else:
                return f"LLM 调用失败: {str(e)}"

    def _build_cited_context(self, docs):
        context_parts = []
        citations = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", 1)
            chunk_index = doc.metadata.get("chunk_index", 1)
            content = doc.page_content.strip()
            if content:
                figure_ids = doc.metadata.get("figure_ids", "")
                fig_part = f", {figure_ids}" if figure_ids else ""
                cite_info = f"(来源: {source}, 第 {page} 页{fig_part}, 第 {chunk_index} 段)"
                context_parts.append(f"{content} {cite_info}")
                citations.append({
                    "index": i,
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index
                })
        return "\n\n".join(context_parts), citations

    def _build_rag_prompt(self, context: str, query: str, extra_rules: str = "") -> list:
        """构建标准 RAG prompt 消息列表"""
        base_rules = """
只能基于给定资料回答，不允许编造。

重要要求：
1. 每条回答内容末尾必须标注精确来源，格式为：(来源: xxx.pdf, 第 N 页，第 M 段)。如果资料中涉及图表信息，格式为：(来源: xxx.pdf, 第 N 页, 图X, 第 M 段) 或 (来源: xxx.pdf, 第 N 页, 表X, 第 M 段)。
2. 来源信息必须直接从资料中复制，不得自行编造
3. 如果某条信息无法对应到资料中的来源，标注：(来源: 模型推理，请核实)
"""
        if extra_rules:
            base_rules += "\n" + extra_rules
        return [
            {"role": "system", "content": "严格基于资料的医学助手"},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：\n{query}\n\n{base_rules}"}
        ]

    def run_rag_pipeline(self, query: str, source_label: str, extra_rules: str = "",
                          system_role: str = "严格基于资料的医学助手", k: int = 5,
                          use_llm_rerank: bool = False, fallback_context: str = None) -> dict:
        """通用 RAG 执行管道，子类可直接调用或复用"""
        import time
        t0 = time.time()
        self.clear_trace() if hasattr(self, 'clear_trace') else None
        self.trace("run_start", {"query": query}) if hasattr(self, 'trace') else None

        try:
            docs = self.retrieve_with_optimization(query, k=k, use_llm_rerank=use_llm_rerank)
        except Exception as e:
            if hasattr(self, 'trace'):
                self.trace("error", {"stage": "retrieve", "error": str(e)})
            return {
                "answer": f"检索失败: {str(e)}",
                "source": source_label,
                "success": False
            }

        if hasattr(self, 'trace'):
            self.trace("retrieve", {"count": len(docs), "samples": [d.page_content[:200] for d in docs]})

        context, citations = self._build_cited_context(docs)
        if hasattr(self, 'trace'):
            self.trace("context", {"length": len(context), "preview": context[:300], "citations": citations})

        messages = self._build_rag_prompt(context, query, extra_rules)
        if system_role:
            messages[0]["content"] = system_role

        answer = self._safe_llm_call(messages, fallback_context=context if fallback_context is None else fallback_context)
        latency = round(time.time() - t0, 3)
        if hasattr(self, 'trace'):
            self.trace("final_answer", {"answer": answer, "latency": latency})

        from utils.response import build_response
        return build_response(
            answer=answer,
            source=source_label,
            debug={"retrieved": len(docs), "latency": latency, "citations": citations},
            trace=self.get_trace() if hasattr(self, 'get_trace') else []
        )

    @abstractmethod
    def run(self, query: str):
        pass

    def retrieve_with_optimization(self, query: str, k: int = 5, use_llm_rerank: bool = False) -> List:
        results = self.retriever.retrieve(query, k=k*2)
        docs = [doc for doc, score in results]

        if use_llm_rerank and len(docs) > k:
            docs = self.retriever.rerank_with_llm(query, docs, top_k=k)
        else:
            docs = docs[:k]

        return docs
