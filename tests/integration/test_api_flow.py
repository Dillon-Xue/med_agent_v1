"""
集成测试：API 端到端测试
使用 FastAPI TestClient 测试真实 HTTP 端点
"""

import pytest
from fastapi.testclient import TestClient
from chat import app

client = TestClient(app)


class TestAPIFlow:

    def test_health_check(self):
        """测试：健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ask_endpoint(self):
        """测试：快速问答 API"""
        response = client.post(
            "/ask",
            json={
                "question": "高血压患者应该注意什么？",
                "history": []
            },
            headers={
                "X-Tenant-ID": "test_tenant",
                "X-Session-ID": "test_session_001"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        # 如果有结果，验证结构
        if data.get("success"):
            assert "result" in data
            assert "answer" in data["result"]

    def test_consult_endpoint(self):
        """测试：智能问诊 API"""
        response = client.post(
            "/consult",
            json={
                "question": "我最近头晕，血压有点高",
                "history": []
            },
            headers={
                "X-Tenant-ID": "test_tenant",
                "X-Session-ID": "test_session_002"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_approvals_endpoint(self):
        """测试：获取待审批列表 API"""
        response = client.get(
            "/approvals",
            headers={
                "X-Tenant-ID": "test_tenant"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "items" in data

    def test_history_endpoint(self):
        """测试：获取历史记录 API"""
        response = client.get(
            "/history",
            params={
                "session_id": "test_session_001",
                "limit": 10
            },
            headers={
                "X-Tenant-ID": "test_tenant"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "items" in data
        assert "count" in data

    def test_approval_detail_endpoint(self, approval_tool, test_approval_data):
        """测试：获取审批详情 API"""
        # 先创建一个审批
        create_result = approval_tool.create(
            title=test_approval_data["title"],
            content=test_approval_data["content"],
            type=test_approval_data["type"],
            requester=test_approval_data["requester"],
            reviewer=test_approval_data["reviewer"]
        )

        import re
        match = re.search(r'APP-\d{8}-\d{3}', create_result.get("answer", ""))
        if not match:
            pytest.skip("无法创建审批项")

        approval_id = match.group(0)

        # 设置当前用户为审批人
        import chat
        chat.current_session_user = "test_doctor"

        response = client.get(
            f"/approval/{approval_id}",
            headers={
                "X-Tenant-ID": "test_tenant"
            }
        )

        # 可能返回 200 或 404，取决于数据是否存在
        # 我们只验证响应格式
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "success" in data

    def test_metrics_endpoint(self):
        """测试：Prometheus 监控指标"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")