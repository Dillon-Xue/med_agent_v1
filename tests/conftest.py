"""
测试公共夹具（Fixtures）
提供 Mock 对象：LLM 客户端、向量库、数据库、工具注册表
"""

import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List
from langchain_core.documents import Document


# ============================================================
# Mock 配置：避免在测试中加载真实环境变量
# ============================================================
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """自动加载测试环境变量，避免依赖 .env"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-api-key")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_pass")
    monkeypatch.setenv("DB_NAME", "test_db")
    monkeypatch.setenv("MED_AGENT_ROOT", "/tmp")
    monkeypatch.setenv("LLM_MODEL_NAME", "qwen-test")
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")  # 避免 ollama 分支


# ============================================================
# Mock LLM 客户端
# ============================================================
@pytest.fixture
def mock_llm_client():
    """
    模拟 OpenAI 客户端，返回固定响应，避免真实 API 调用。
    使用方式：
        client, model = get_llm_client()
        # 在测试中 mock 这个方法
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "测试合成的回答"

    mock_create = MagicMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    # 同时提供 model 名称
    mock_model = "qwen-test"
    return mock_client, mock_model


# ============================================================
# Mock 向量库（Chroma）
# ============================================================
@pytest.fixture
def mock_vectordb():
    """
    模拟 Chroma 向量库，返回预设文档和分数。
    返回的文档对象模拟 langchain.schema.Document。
    """
    from langchain.schema import Document

    docs = [
        Document(page_content="文档1：关于感冒的治疗", metadata={"source": "test1"}),
        Document(page_content="文档2：关于心脏搭桥术后用药", metadata={"source": "test2"}),
        Document(page_content="文档3：药物相互作用说明", metadata={"source": "test3"}),
    ]
    # 模拟 similarity_search_with_score 返回 (doc, score) 列表
    scores = [0.8, 0.5, 0.3]
    mock_retriever = MagicMock()
    mock_retriever.similarity_search_with_score = MagicMock(
        return_value=list(zip(docs, scores))
    )
    return mock_retriever


# ============================================================
# Mock 数据库连接（使用内存 SQLite）
# ============================================================
@pytest.fixture
def mock_db_connection(monkeypatch):
    """
    模拟 MySQL 连接，实际使用 SQLite 内存数据库，并替换 PatientTool 的数据库方法。
    """
    import sqlite3
    import pymysql

    # 创建内存数据库并初始化表
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            gender TEXT,
            age TEXT,
            id_card TEXT,
            phone TEXT,
            address TEXT,
            allergy TEXT,
            medication TEXT,
            symptoms TEXT,
            diagnosis TEXT,
            info TEXT,
            doctor_id TEXT,
            tenant_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE approvals (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            type TEXT,
            requester TEXT,
            reviewer TEXT,
            doctor_id TEXT,
            tenant_id TEXT,
            status TEXT DEFAULT 'pending',
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP
        )
    ''')
    conn.commit()

    # Mock pymysql.connect 返回这个 SQLite 连接
    def mock_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(pymysql, "connect", mock_connect)
    return conn


# ============================================================
# Mock 工具注册表
# ============================================================
@pytest.fixture
def mock_tools(mock_vectordb):
    """
    模拟 get_tools() 返回的工具字典，所有工具返回固定值。
    """
    from tools.drug_tool import DrugTool
    from tools.guideline_tool import GuidelineTool
    from tools.literature_tool import LiteratureTool
    from tools.risk_tool import RiskTool
    from tools.patient_tool import PatientTool
    

    # 创建 Mock 工具对象
    def make_mock_tool(name, return_value):
        tool = MagicMock()
        tool.run = MagicMock(return_value=return_value)
        tool.get_trace = MagicMock(return_value=[])
        tool.clear_trace = MagicMock()
        tool.trace = MagicMock()
        return tool

    return {
        "drug": make_mock_tool("drug", {"answer": "药品信息", "source": "drug", "success": True}),
        "guideline": make_mock_tool("guideline", {"answer": "指南信息", "source": "guideline", "success": True}),
        "literature": make_mock_tool("literature", {"answer": "文献信息", "source": "literature", "success": True}),
        "risk": make_mock_tool("risk", {"answer": "风险信息", "source": "risk", "success": True}),
        "patient": make_mock_tool("patient", {"answer": "患者档案", "source": "patient", "success": True}),
        "report": make_mock_tool("report", {"answer": "报告生成", "source": "report", "success": True}),
        "approval": make_mock_tool("approval", {"answer": "审批信息", "source": "approval", "success": True}),
        "file": make_mock_tool("file", {"answer": "文件解析", "source": "file", "success": True}),
    }


# ============================================================
# 辅助函数：创建测试用的文档对象
# ============================================================
@pytest.fixture
def sample_documents():
    from langchain.schema import Document
    return [
        Document(page_content="测试文档1：感冒灵颗粒用于风热感冒", metadata={}),
        Document(page_content="测试文档2：心脏搭桥术后需慎用NSAIDs", metadata={}),
        Document(page_content="测试文档3：阿莫西林与酒精无显著相互作用", metadata={}),
    ]