"""
测试 tools/retriever.py 的混合检索逻辑
覆盖：BM25 归一化、混合得分计算、查询改写、去重
"""

import pytest
from unittest.mock import MagicMock, patch
from tools.retriever import HybridRetriever
from langchain_core.documents import Document


# ============================================================
# 测试用例 1：BM25 分数归一化（防止除零）
# ============================================================
class TestBM25Normalization:
    def test_max_score_zero(self, monkeypatch):
        """
        目标：当 max_score 为 0 时，归一化使用默认值 1.0，不抛出异常。
        输入：模拟 BM25Okapi.get_scores 返回全零数组。
        预期结果：_bm25_search 正常运行，normalized_scores 全为 0。
        """
        mock_vectordb = MagicMock()
        retriever = HybridRetriever(mock_vectordb)

        with patch('tools.retriever.BM25Okapi') as MockBM25:
            mock_bm25 = MagicMock()
            mock_bm25.get_scores.return_value = [0.0, 0.0, 0.0]
            MockBM25.return_value = mock_bm25

            docs = [
                (Document(page_content="doc1"), 0.5),
                (Document(page_content="doc2"), 0.3),
                (Document(page_content="doc3"), 0.8),
            ]
            result = retriever._bm25_search("query", docs, top_k=2)
            assert len(result) == 2
            # 排序结果应为 doc2, doc1（因为 vec_score 越小相似度越高）
            assert result[0][0].page_content == "doc2"
            assert result[1][0].page_content == "doc1"


# ============================================================
# 测试用例 2：混合得分计算（简化版）
# ============================================================
class TestMixedScore:
    def test_merged_score_calculation(self):
        """
        目标：验证加权平均计算正确。
        输入：两个文档，vec_score 分别为 0.2 和 0.5，
            BM25 分数为 0.8 和 0.6（归一化后为 1.0 和 0.75）。
        预期结果：
            文档1 merged_score = 0.6*(1-0.2) + 0.4*1.0 = 0.88
            文档2 merged_score = 0.6*(1-0.5) + 0.4*0.75 = 0.3+0.3 = 0.6
        """
        mock_vectordb = MagicMock()
        retriever = HybridRetriever(mock_vectordb)

        doc1 = Document(page_content="doc1")
        doc2 = Document(page_content="doc2")
        docs = [(doc1, 0.2), (doc2, 0.5)]

        with patch('tools.retriever.BM25Okapi') as MockBM25:
            mock_bm25 = MagicMock()
            # BM25 原始分数
            mock_bm25.get_scores.return_value = [0.8, 0.6]
            MockBM25.return_value = mock_bm25

            result = retriever._bm25_search("query", docs, top_k=2)
            # 结果按 merged_score 降序排列，文档1分高排前面
            assert len(result) == 2
            assert result[0][0].page_content == "doc1"
            assert abs(result[0][1] - 0.88) < 1e-6
            assert result[1][0].page_content == "doc2"
            assert abs(result[1][1] - 0.6) < 1e-6


# ============================================================
# 测试用例 3：查询改写
# ============================================================
class TestQueryRewrite:
    def test_query_rewrite_returns_multiple(self):
        """
        目标：query_rewrite 调用 LLM 并解析返回的查询列表。
        输入：query="胸痛"，Mock LLM 返回 "胸痛伴随症状\n胸痛原因"
        预期结果：返回 ["胸痛", "胸痛伴随症状", "胸痛原因"]。
        """
        mock_vectordb = MagicMock()
        retriever = HybridRetriever(mock_vectordb)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "胸痛伴随症状\n胸痛原因"
        mock_client.chat.completions.create.return_value = mock_response
        retriever.client = mock_client

        result = retriever.query_rewrite("胸痛")
        assert result == ["胸痛", "胸痛伴随症状", "胸痛原因"]

    def test_query_rewrite_fallback(self):
        """
        目标：LLM 调用失败时，返回仅包含原始查询的列表。
        输入：query="胸痛"，Mock LLM 抛出异常。
        预期结果：返回 ["胸痛"]。
        """
        mock_vectordb = MagicMock()
        retriever = HybridRetriever(mock_vectordb)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        retriever.client = mock_client

        result = retriever.query_rewrite("胸痛")
        assert result == ["胸痛"]


# ============================================================
# 测试用例 4：去重与合并
# ============================================================
class TestDeduplication:
    def test_retrieve_deduplicates(self):
        """
        目标：不同改写查询检索到相同文档时，retrieve 方法去重。
        输入：两个查询返回相同文档。
        预期结果：all_docs 中该文档只出现一次。
        """
        mock_vectordb = MagicMock()
        doc = Document(page_content="相同的文档", metadata={"source": "test"})
        mock_vectordb.similarity_search_with_score.return_value = [(doc, 0.5)]

        retriever = HybridRetriever(mock_vectordb)
        with patch.object(retriever, 'query_rewrite', return_value=["q1", "q2"]):
            with patch.object(retriever, '_bm25_search', return_value=[(doc, 0.5)]):
                result = retriever.retrieve("test", k=2)
                # 2个改写查询 + 1次补充检索（因为召回1<2），共3次
                assert mock_vectordb.similarity_search_with_score.call_count == 3
                assert len(result) == 1