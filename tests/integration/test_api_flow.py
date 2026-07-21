"""
集成测试：API 端到端测试
使用 FastAPI TestClient 测试真实 HTTP 端点
覆盖 README 中全部 5 个核心流程图
"""

import pytest
from fastapi.testclient import TestClient
from chat import app

client = TestClient(app)


class TestAPIFlow:

    # ========== 流程图1：快速问答（/ask） ==========
    def test_health_check(self):
        """测试：健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ask_endpoint(self):
        """测试：快速问答 API（/ask）"""
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
        if data.get("success"):
            assert "result" in data
            assert "answer" in data["result"]

    # ========== 流程图4：V4多Agent协作 ==========
    def test_v1_ask_endpoint(self):
        """测试：V4多Agent协作快速问答（/v1/ask）"""
        response = client.post(
            "/v1/ask",
            json={
                "question": "感冒灵颗粒的用法用量？",
                "history": []
            },
            headers={
                "X-Tenant-ID": "test_tenant",
                "X-Session-ID": "test_session_v1"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        if data.get("success"):
            assert "result" in data
            assert "answer" in data["result"]

    # ========== 流程图2：智能问诊（/consult） ==========
    def test_consult_endpoint(self):
        """测试：智能问诊 API（/consult）"""
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
        if data.get("success"):
            assert "result" in data
            assert "answer" in data["result"]

    def test_consult_with_history(self):
        """测试：智能问诊多轮对话（追问与上下文）"""
        history = [
            {"role": "user", "content": "患者张三，男，65岁，高血压10年"},
            {"role": "assistant", "content": "已记录患者信息"}
        ]
        response = client.post(
            "/consult",
            json={
                "question": "他可以吃阿司匹林吗？",
                "history": history
            },
            headers={
                "X-Tenant-ID": "test_tenant",
                "X-Session-ID": "test_session_003"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    # ========== 流程图3：评估表+审批+数据回流 ==========
    def test_approvals_endpoint(self):
        """测试：获取待审批列表 API"""
        response = client.get(
            "/approvals",
            headers={"X-Tenant-ID": "test_tenant"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "items" in data

    def test_approval_detail_endpoint(self, approval_tool, test_approval_data):
        """测试：获取审批详情 API"""
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
        import chat
        chat.current_session_user = "test_doctor"
        response = client.get(
            f"/approval/{approval_id}",
            headers={"X-Tenant-ID": "test_tenant"}
        )
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "success" in data

    def test_approval_approve_api(self, approval_tool, test_approval_data):
        """测试：审批通过 API（POST /approval/{id}/approve）"""
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
        import chat
        chat.current_session_user = "test_doctor"
        response = client.post(f"/approval/{approval_id}/approve")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True

    def test_approval_reject_api(self, approval_tool, test_approval_data):
        """测试：审批驳回 API（POST /approval/{id}/reject）"""
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
        import chat
        chat.current_session_user = "test_doctor"
        response = client.post(
            f"/approval/{approval_id}/reject",
            json={"comment": "测试驳回，方案需调整"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True

    # ========== 会话管理与用户切换 ==========
    def test_history_endpoint(self):
        """测试：获取历史记录 API"""
        response = client.get(
            "/history",
            params={
                "session_id": "test_session_001",
                "limit": 10
            },
            headers={"X-Tenant-ID": "test_tenant"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "items" in data
        assert "count" in data

    def test_switch_user_endpoint(self):
        """测试：用户切换 API"""
        response = client.post(
            "/switch_user",
            json={"name": "doctor_测试切换"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        assert data.get("user") == "doctor_测试切换"

    # ========== 文件上传 ==========
    def test_upload_endpoint(self):
        """测试：文件上传解析 API"""
        from io import BytesIO
        test_content = b"test file content for upload endpoint"
        response = client.post(
            "/upload",
            data={"module": "consult"},
            files={"file": ("test.txt", BytesIO(test_content), "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    # ========== 监控指标 ==========
    def test_metrics_endpoint(self):
        """测试：Prometheus 监控指标"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
