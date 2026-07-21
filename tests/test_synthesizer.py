"""
测试 agents/synthesizer.py 的答案合成逻辑
覆盖：空结果、单工具直接返回、多工具合成、科室提示词
"""

import pytest
from unittest.mock import MagicMock, patch
from agents.synthesizer import Synthesizer


# ============================================================
# 测试用例 1：空结果处理
# ============================================================
class TestEmptyResults:
    def test_empty_tool_results(self):
        syn = Synthesizer(api_key="fake")
        syn.client = MagicMock()
        result = syn.run("问题", [])
        assert result == "未找到相关信息，请尝试换一种问法。"
        syn.client.chat.completions.create.assert_not_called()


# ============================================================
# 测试用例 2：单工具结果直接返回
# ============================================================
class TestSingleTool:
    def test_patient_tool_direct(self):
        syn = Synthesizer(api_key="fake")
        syn.client = MagicMock()
        result = syn.run("问题", [{"source": "patient", "answer": "✅ 已记住患者"}])
        assert result == "✅ 已记住患者"
        syn.client.chat.completions.create.assert_not_called()

    def test_report_tool_direct(self):
        syn = Synthesizer(api_key="fake")
        syn.client = MagicMock()
        result = syn.run("问题", [{"source": "report", "answer": "📎 下载评估表"}])
        assert result == "📎 下载评估表"
        syn.client.chat.completions.create.assert_not_called()


# ============================================================
# 测试用例 3：多工具结果合成
# ============================================================
class TestMultiTool:
    def test_synthesize_multiple(self):
        syn = Synthesizer(api_key="fake")
        syn.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "合成的答案"
        syn.client.chat.completions.create.return_value = mock_response

        tool_results = [
            {"source": "drug", "answer": "药品信息 (来源: 药品说明书.pdf, 第 2 页，第 1 段)"},
            {"source": "guideline", "answer": "指南信息 (来源: 临床指南.pdf, 第 5 页，第 2 段)"},
        ]
        result = syn.run("问题", tool_results)
        # 注意返回会包含科室标签，因此检查是否包含目标答案
        assert "合成的答案" in result
        call_args = syn.client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        user_msg = next(msg for msg in messages if msg["role"] == "user")
        assert "药品信息" in user_msg["content"]
        assert "指南信息" in user_msg["content"]


# ============================================================
# 测试用例 4：科室专属提示词
# ============================================================
class TestSpecialtyPrompt:
    def test_cardiology_prompt(self):
        syn = Synthesizer(api_key="fake", specialty="cardiology")
        syn.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "心外科回答"
        syn.client.chat.completions.create.return_value = mock_response

        result = syn.run("问题", [{"source": "drug", "answer": "信息 (来源: 药品说明书.pdf, 第 1 页，第 1 段)"}])
        call_args = syn.client.chat.completions.create.call_args[1]
        system_msg = next(msg for msg in call_args["messages"] if msg["role"] == "system")
        assert "心外科专家" in system_msg["content"]

    def test_pharmacy_prompt(self):
        syn = Synthesizer(api_key="fake", specialty="pharmacy")
        syn.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "药剂科回答"
        syn.client.chat.completions.create.return_value = mock_response

        result = syn.run("问题", [{"source": "drug", "answer": "信息 (来源: 药品说明书.pdf, 第 1 页，第 1 段)"}])
        call_args = syn.client.chat.completions.create.call_args[1]
        system_msg = next(msg for msg in call_args["messages"] if msg["role"] == "system")
        assert "药剂科专家" in system_msg["content"]