import re
import os
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from openai import OpenAI
from utils.config import get_llm_client
import logging
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        # 🆕 使用统一客户端工厂
        self.client, self.model = get_llm_client()
        logger.debug(f"[HybridRetriever] Using model: {self.model}")

    def query_rewrite(self, query: str) -> List[str]:
        """用 LLM 生成 2-3 个不同角度的查询"""
        prompt = f"""
请将以下医学问题改写为 2-3 个不同角度的搜索查询，每个查询用换行分隔。
不同角度可以包括：症状描述、药物名称、治疗方式、疾病机制、风险因素等。

原始问题：{query}

只输出查询列表，每行一个，不要添加其他内容。
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            queries = [q.strip() for q in resp.choices[0].message.content.strip().split('\n') if q.strip()]
            # 去重，保留原始查询
            all_queries = [query] + [q for q in queries if q != query]
            return all_queries[:3]
        except Exception as e:
            logger.info(f"[Retriever] 查询改写失败: {e}")
            return [query]

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        return re.findall(r'\w+', text.lower())

    def _bm25_search(self, query: str, docs: List, top_k: int = 5) -> List[Tuple]:
        if len(docs) < 2:
            return docs

        corpus = [doc.page_content for doc, _ in docs]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self._tokenize(query)

        scores = bm25.get_scores(tokenized_query)
        # 强制转换为 Python float，避免 NumPy 数组问题
        scores = [float(s) for s in scores]
        
        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        normalized_scores = [float(s) / float(max_score) for s in scores]

        merged = []
        for idx, (doc, vec_score) in enumerate(docs):
            # 确保 vec_score 是标量
            if hasattr(vec_score, '__iter__') and not isinstance(vec_score, str):
                vec_score = vec_score[0] if len(vec_score) > 0 else 0.0
            vec_score = float(vec_score)
            vec_similarity = 1 - vec_score

            bm25_score = normalized_scores[idx] if idx < len(normalized_scores) else 0.0
            if hasattr(bm25_score, '__iter__') and not isinstance(bm25_score, str):
                bm25_score = bm25_score[0] if len(bm25_score) > 0 else 0.0
            bm25_score = float(bm25_score)

            merged_score = 0.6 * vec_similarity + 0.4 * bm25_score
            merged.append((doc, merged_score))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    def retrieve(self, query: str, k: int = 10, trace_callback=None) -> List[Tuple]:
        queries = self.query_rewrite(query)
        logger.debug(f"[Retriever] 改写后的查询: {queries}")

        if trace_callback:
            trace_callback("retriever", {"query": query, "rewritten_queries": queries})

        all_docs = {}
        for q in queries:
            docs = self.vectordb.similarity_search_with_score(q, k=k)
            for doc, score in docs:
                # 强制转换 score 为标量
                try:
                    score_val = float(score)
                except (TypeError, ValueError):
                    score_val = float(score[0]) if len(score) > 0 else 0.0
                # 使用Chroma返回的文档ID进行去重（如果可用），否则使用内容+元数据哈希
                doc_id = getattr(doc, "id", None)
                if not doc_id:
                    import hashlib
                    doc_id = hashlib.md5((doc.page_content[:200] + str(sorted(doc.metadata.items()))).encode()).hexdigest()
                if doc_id not in all_docs or score_val < all_docs[doc_id][1]:
                    all_docs[doc_id] = (doc, score_val)

        logger.debug(f"[Retriever] 向量检索去重后: {len(all_docs)} 条")

        if len(all_docs) < k:
            docs = self.vectordb.similarity_search_with_score(query, k=k*2)
            for doc, score in docs:
                try:
                    score_val = float(score)
                except (TypeError, ValueError):
                    score_val = float(score[0]) if len(score) > 0 else 0.0
                # 使用Chroma返回的文档ID进行去重（如果可用），否则使用内容+元数据哈希
                doc_id = getattr(doc, "id", None)
                if not doc_id:
                    import hashlib
                    doc_id = hashlib.md5((doc.page_content[:200] + str(sorted(doc.metadata.items()))).encode()).hexdigest()
                if doc_id not in all_docs:
                    all_docs[doc_id] = (doc, score_val)

        doc_list = list(all_docs.values())
        reranked = self._bm25_search(query, doc_list, top_k=k)

        if trace_callback:
            trace_callback("retriever", {
                "status": "complete",
                "query": query,
                "rewritten_queries": queries,
                "result_count": len(reranked),
                "top_result_preview": reranked[0][0].page_content[:200] if reranked else ""
            })
        return reranked[:k]

    def rerank_with_llm(self, query: str, docs: List, top_k: int = 5) -> List:
        """用 LLM 对检索结果进行重排序（更精确但更慢）"""
        if len(docs) <= top_k:
            return docs

        prompt = f"""
请根据以下问题，对检索到的文档进行相关性排序，只返回最相关的 {top_k} 个文档的编号（从 0 开始）。

问题：{query}

文档列表：
"""
        for i, doc in enumerate(docs):
            preview = doc.page_content[:250].replace('\n', ' ')
            prompt += f"{i}: {preview}...\n"

        prompt += f"\n只输出最相关的 {top_k} 个编号，用逗号分隔，如：2,4,0,1,3"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            # 解析返回的编号
            content = resp.choices[0].message.content.strip()
            # 提取所有数字
            indices = [int(x) for x in re.findall(r'\d+', content)]
            # 去重并取前 top_k 个
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen and idx < len(docs):
                    seen.add(idx)
                    unique_indices.append(idx)
                    if len(unique_indices) >= top_k:
                        break

            return [docs[i] for i in unique_indices]
        except Exception as e:
            logger.WARNING(f"[Retriever] LLM 重排序失败: {e}")
            return docs[:top_k]