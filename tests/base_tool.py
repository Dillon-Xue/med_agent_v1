"""
测试 tools/base_tool.py 的重试和降级机制
覆盖：正常调用、重试成功、降级返回（有/无 fallback_context）
"""

import pytest
from unittest.mock import MagicMock, patch
from tools.base_tool import BaseTool
import os


# 定义一个具体的子类以便测试抽象方法
class ConcreteTool(BaseTool):
    def run(self, query: str):
        pass


# ============================================================
# 测试用例 1：正常 LLM 调用
# ============================================================
class TestNormalCall:
    def test_safe_llm_call_success(self):
        """
        目标：_safe_llm_call 在 LLM 正常时返回结果。
        """
        tool = ConcreteTool(base_dir="/tmp", db_name="test", api_key="fake")
        tool.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "正常回答"
        tool.client.chat.completions.create.return_value = mock_response

        result = tool._safe_llm_call([{"role": "user", "content": "test"}])
        assert result == "正常回答"


# ============================================================
# 测试用例 2：重试机制
# ============================================================
class TestRetry:
    def test_retry_success_after_failures(self):
        """
        目标：前两次失败，第三次成功，重试生效。
        """
        tool = ConcreteTool(base_dir="/tmp", db_name="test", api_key="fake")
        tool.client = MagicMock()
        # 模拟前两次抛异常，第三次成功
        calls = [Exception(), Exception(), MagicMock(choices=[MagicMock(message=MagicMock(content="第三次成功"))])]
        tool.client.chat.completions.create.side_effect = calls

        # _safe_llm_call 内部调用 _llm_call_with_retry，后者有 tenacity 重试
        result = tool._safe_llm_call([{"role": "user", "content": "test"}])
        assert result == "第三次成功"
        # 调用次数应为 3
        assert tool.client.chat.completions.create.call_count == 3


# ============================================================
# 测试用例 3：降级返回（有 fallback_context）
# ============================================================
class TestFallbackWithContext:
    def test_fallback_with_context(self):
        """
        目标：LLM 全部失败且提供 fallback_context 时，返回降级信息+片段。
        """
        tool = ConcreteTool(base_dir="/tmp", db_name="test", api_key="fake")
        tool.client = MagicMock()
        tool.client.chat.completions.create.side_effect = Exception("API error")

        result = tool._safe_llm_call(
            [{"role": "user", "content": "test"}],
            fallback_context="测试降级内容"
        )
        assert result.startswith("【系统降级】LLM 服务暂时不可用")
        assert "测试降级内容" in result


# ============================================================
# 测试用例 4：降级返回（无 fallback_context）
# ============================================================
class TestFallbackWithoutContext:
    def test_fallback_without_context(self):
        """
        目标：LLM 全部失败且无 fallback_context，返回纯错误信息。
        """
        tool = ConcreteTool(base_dir="/tmp", db_name="test", api_key="fake")
        tool.client = MagicMock()
        tool.client.chat.completions.create.side_effect = Exception("API error")

        result = tool._safe_llm_call([{"role": "user", "content": "test"}])
        assert result.startswith("LLM 调用失败:")
        assert "API error" in result