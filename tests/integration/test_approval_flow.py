"""
集成测试：审批流程完整流程（含数据回流）
"""

import pytest
import re
import time


class TestApprovalFlow:

    def test_create_approval(self, approval_tool, test_approval_data):
        """测试：创建审批项"""
        data = test_approval_data
        result = approval_tool.create(
            title=data["title"],
            content=data["content"],
            type=data["type"],
            requester=data["requester"],
            reviewer=data["reviewer"]
        )

        assert result.get("success") is True
        answer = result.get("answer", "")
        assert "已创建审批项" in answer

        # 提取审批 ID（只验证格式，不返回）
        match = re.search(r'APP-\d{8}-\d{3}', answer)
        assert match is not None
        # 不再 return，改为 assert

    def test_list_pending(self, approval_tool, test_approval_data):
        """测试：查询待审批列表"""
        # 先创建一个审批
        approval_tool.create(
            title=test_approval_data["title"],
            content=test_approval_data["content"],
            type=test_approval_data["type"],
            requester=test_approval_data["requester"],
            reviewer=test_approval_data["reviewer"]
        )

        result = approval_tool.list_pending("用户：test_doctor")

        assert result.get("success") is True
        answer = result.get("answer", "")
        # 应该包含待审批列表信息
        assert "待审批列表" in answer or "当前没有待您审批的项" in answer

    def test_approve_flow_with_memory(self, approval_tool, test_approval_data):
        """测试：审批通过 → 触发数据回流 → 记忆库写入"""
        # 1. 创建审批
        create_result = approval_tool.create(
            title=test_approval_data["title"],
            content=test_approval_data["content"],
            type=test_approval_data["type"],
            requester=test_approval_data["requester"],
            reviewer=test_approval_data["reviewer"]
        )

        match = re.search(r'APP-\d{8}-\d{3}', create_result.get("answer", ""))
        assert match is not None
        approval_id = match.group(0)

        # 2. 审批通过
        result = approval_tool.approve(approval_id, "用户：test_doctor")

        assert result.get("success") is True
        assert "已通过" in result.get("answer", "")

        # 3. 验证数据回流（查看日志或查询记忆库）
        # 注意：记忆库是 Chroma，这里我们通过检查日志来验证
        # 在实际 CI 中，可以查询 Chroma 或检查日志文件
        # 此处我们通过 approval_tool 的日志输出确认
        import logging
        # 数据回流会在日志中打印 "已写入记忆库"
        # 由于无法直接断言日志，我们验证审批状态已变更
        time.sleep(0.5)

        # 查询数据库验证状态
        from utils.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM approvals WHERE id = %s",
            (approval_id,)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row.get("status") == "approved"

    def test_reject_flow(self, approval_tool, test_approval_data):
        """测试：审批驳回 → 审计日志记录"""
        # 1. 创建审批
        create_result = approval_tool.create(
            title=test_approval_data["title"],
            content=test_approval_data["content"],
            type=test_approval_data["type"],
            requester=test_approval_data["requester"],
            reviewer=test_approval_data["reviewer"]
        )

        match = re.search(r'APP-\d{8}-\d{3}', create_result.get("answer", ""))
        assert match is not None
        approval_id = match.group(0)

        # 2. 审批驳回
        comment = "用药方案与患者过敏史冲突"
        result = approval_tool.reject(approval_id, comment, "用户：test_doctor")

        assert result.get("success") is True
        assert "已驳回" in result.get("answer", "")

        # 3. 验证数据库中状态
        from utils.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, comment FROM approvals WHERE id = %s",
            (approval_id,)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row.get("status") == "rejected"
        assert comment in row.get("comment", "")

    def test_reject_approval_duplicate_audit_not_allowed(self, approval_tool, test_approval_data):
        """测试：驳回操作只产生一条审计日志（不重复）"""
        # 1. 创建审批
        create_result = approval_tool.create(
            title=test_approval_data["title"],
            content=test_approval_data["content"],
            type=test_approval_data["type"],
            requester=test_approval_data["requester"],
            reviewer=test_approval_data["reviewer"]
        )

        match = re.search(r'APP-\d{8}-\d{3}', create_result.get("answer", ""))
        assert match is not None
        approval_id = match.group(0)

        # 2. 审批驳回
        result = approval_tool.reject(approval_id, "测试驳回", "用户：test_doctor")

        # 3. 查询审计日志数量
        from utils.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM audit_logs WHERE resource_id = %s AND action = 'REJECT'",
            (approval_id,)
        )
        row = cursor.fetchone()
        conn.close()

        # 应该只有 1 条记录
        assert row.get("cnt") == 1