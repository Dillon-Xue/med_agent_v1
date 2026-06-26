"""
测试 agents/executor.py 的并行执行和异常处理
覆盖：正常并行、超时、工具不存在、异常捕获
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch
from agents.executor import Executor


# ============================================================
# 测试用例 1：正常并行执行
# ============================================================
@pytest.mark.asyncio
class TestNormalExecution:
    async def test_parallel_execution(self):
        """
        目标：多个工具并行执行，结果正确收集。
        输入：3 个 Mock 工具，分别返回 "result1", "result2", "result3"。
        预期结果：返回包含 3 个结果的列表，顺序与输入一致。
        """
        # 创建模拟工具对象，设置 run 方法的返回值
        tool1 = MagicMock()
        tool1.run.return_value = {"answer": "result1", "source": "tool1", "success": True}
        tool2 = MagicMock()
        tool2.run.return_value = {"answer": "result2", "source": "tool2", "success": True}
        tool3 = MagicMock()
        tool3.run.return_value = {"answer": "result3", "source": "tool3", "success": True}

        tools = {
            "tool1": tool1,
            "tool2": tool2,
            "tool3": tool3,
        }
        executor = Executor(tools)
        results = await executor.run(["tool1", "tool2", "tool3"], "question")
        assert len(results) == 3
        assert results[0]["answer"] == "result1"
        assert results[1]["answer"] == "result2"
        assert results[2]["answer"] == "result3"


# ============================================================
# 测试用例 2：超时处理
# ============================================================
@pytest.mark.asyncio
class TestTimeout:
    async def test_timeout(self):
        """
        目标：工具执行超过 60 秒时返回降级信息。
        输入：一个 Mock 工具，run 方法休眠 70 秒。
        预期结果：捕获 TimeoutError，返回 success=False 且 answer 包含“超时”。
        """
        def slow_run(question):
            time.sleep(70)
            return {"answer": "ok"}
        
        tool = MagicMock()
        tool.run.side_effect = slow_run
        tools = {"slow": tool}
        executor = Executor(tools)
        # 将超时临时改为 1 秒，以便测试快速通过
        with patch('agents.executor.asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            results = await executor.run(["slow"], "test")
            assert len(results) == 1
            assert results[0]["success"] is False
            assert "超时" in results[0]["answer"]


# ============================================================
# 测试用例 3：工具不存在
# ============================================================
@pytest.mark.asyncio
class TestMissingTool:
    async def test_missing_tool(self):
        """
        目标：请求未注册工具时返回错误信息。
        输入：tool_list 包含 "nonexistent"。
        预期结果：返回 success=False，answer 含“工具 nonexistent 未找到”。
        """
        tools = {}
        executor = Executor(tools)
        results = await executor.run(["nonexistent"], "question")
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "nonexistent" in results[0]["answer"]


# ============================================================
# 测试用例 4：工具执行异常
# ============================================================
@pytest.mark.asyncio
class TestException:
    async def test_tool_raises_exception(self):
        """
        目标：工具抛出异常时被捕获并返回错误信息。
        输入：Mock 工具 run 抛出 ValueError("test error")。
        预期结果：返回 success=False，answer 含“执行失败: test error”。
        """
        def failing_run(question):
            raise ValueError("test error")
        tool = MagicMock()
        tool.run.side_effect = failing_run
        tools = {"failing": tool}
        executor = Executor(tools)
        results = await executor.run(["failing"], "question")
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "test error" in results[0]["answer"]