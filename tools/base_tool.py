from abc import ABC, abstractmethod
from openai import OpenAI
from langchain_chroma import Chroma
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tools.retriever import HybridRetriever
from typing import List, Optional
from utils.config import get_llm_provider, get_ollama_base_url, get_ollama_model, get_llm_client

PROJECT_ROOT = os.getenv("MED_AGENT_ROOT", os.getcwd())

class BaseTool(ABC):
    def __init__(self, base_dir, db_name, api_key):
        self.base_dir = PROJECT_ROOT
        self.db_path = os.path.join(self.base_dir, "vector_db", db_name)
        print(f"\n[BASE_TOOL] db_path = {self.db_path}")

        self.client, self.model = get_llm_client(api_key, timeout=60.0)
        print(f"[BASE_TOOL] Using model: {self.model}")

        self.vectordb = None
        self.retriever = None

    def load_db(self, embedding_fn):
        self.vectordb = Chroma(
            persist_directory=self.db_path,
            embedding_function=embedding_fn,
            collection_name="langchain"
        )
        self.retriever = HybridRetriever(self.vectordb)
        print(f"[BASE_TOOL] Chroma loaded from {self.db_path}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _llm_call_with_retry(self, messages, temperature=0.2, model=None):
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
            print(f"[ERROR] LLM call failed after retries: {e}")
            if fallback_context:
                return f"【系统降级】LLM 服务暂时不可用，以下是检索到的相关信息：\n\n{fallback_context[:500]}"
            else:
                return f"LLM 调用失败: {str(e)}"

    @abstractmethod
    def run(self, query: str):
        pass 
    
    def retrieve_with_optimization(self, query: str, k: int = 5, use_llm_rerank: bool = False) -> List:
        """使用优化后的检索（混合检索 + 可选LLM重排序）"""
        # 1. 混合检索
        results = self.retriever.retrieve(query, k=k*2)
        docs = [doc for doc, score in results]

        # 2. 可选 LLM 重排序
        if use_llm_rerank and len(docs) > k:
            docs = self.retriever.rerank_with_llm(query, docs, top_k=k)
        else:
            docs = docs[:k]

        return docs
