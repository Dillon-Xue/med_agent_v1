"""
本次 Bugfix 验证测试
覆盖问题 1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 的修改验证
"""
import os
import sys
import re
import pytest
import uuid
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== 问题1: ContextVar 并发安全 ==========
def test_context_var_used_instead_of_global():
    from utils.thread_context import doctor_id_var, tenant_id_var
    from contextvars import ContextVar
    assert isinstance(doctor_id_var, ContextVar)
    assert isinstance(tenant_id_var, ContextVar)
    doctor_id_var.set("test_doc_1")
    assert doctor_id_var.get() == "test_doc_1"

# ========== 问题4: ingest.py False Positive 过滤 ==========
def test_extract_figure_ids_filters_false_positives():
    from ingest import extract_figure_ids
    # "图2周" 应该被过滤
    ids = extract_figure_ids("患者用药图2周后好转")
    assert "图2" not in ids, f"False positive 未被过滤: {ids}"
    # "图1" 应该被保留
    ids2 = extract_figure_ids("见图1和表2")
    assert "图1" in ids2
    assert "表2" in ids2

# ========== 问题5: 文件上传类型白名单 ==========
def test_upload_allowed_types_defined():
    from utils.config import get_allowed_upload_types
    types = get_allowed_upload_types()
    assert "image/png" in types
    assert "image/jpeg" in types
    assert "application/pdf" in types

def test_chat_upload_has_type_check():
    import chat
    src = open(chat.__file__, "r", encoding="utf-8").read()
    assert "allowed_types" in src or "file.content_type not in allowed_types" in src

# ========== 问题6: embeddings 降级响应 ==========
def test_embed_documents_fallback():
    from utils.embeddings import DashscopeEmbeddings
    emb = DashscopeEmbeddings()
    # 由于无 API key，调用会失败，但应返回零向量而非抛出异常
    try:
        result = emb.embed_documents(["test"])
        assert len(result) == 1
        assert len(result[0]) == 1536
        assert result[0][0] == 0.0
    except Exception as e:
        pytest.fail(f"embed_documents 未做降级: {e}")

def test_embed_query_fallback():
    from utils.embeddings import DashscopeEmbeddings
    emb = DashscopeEmbeddings()
    try:
        result = emb.embed_query("test")
        assert len(result) == 1536
        assert result[0] == 0.0
    except Exception as e:
        pytest.fail(f"embed_query 未做降级: {e}")

# ========== 问题7: 审批审计原子性 ==========
def test_approval_audit_before_commit():
    import tools.approval_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    # approve 中 log_audit 应在 conn.commit() 之前（第一个commit前）
    approve_start = src.find("def approve(")
    approve_end = src.find("def _write_to_memory(", approve_start)
    approve_block = src[approve_start:approve_end]
    first_commit_pos = approve_block.find("conn.commit()")
    audit_pos = approve_block.find("log_audit(")
    assert audit_pos != -1 and audit_pos < first_commit_pos, "approve 中 log_audit 应在第一个 commit 之前"

    # reject 中 log_audit 也应在 conn.commit() 之前
    reject_start = src.find("def reject(")
    reject_end = src.find("def run(", reject_start)
    reject_block = src[reject_start:reject_end]
    first_commit_pos_r = reject_block.find("conn.commit()")
    audit_pos_r = reject_block.find("log_audit(")
    assert audit_pos_r != -1 and audit_pos_r < first_commit_pos_r, "reject 中 log_audit 应在第一个 commit 之前"

# ========== 问题8: 患者数据解密返回 ==========
def test_search_patients_decrypts_sensitive_fields():
    import tools.patient_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    # search_patients 中应对 id_card 和 phone 调用 decrypt_if_needed
    sp_start = src.find("def search_patients(")
    sp_end = src.find("def get_patient_by_id_card(", sp_start)
    sp_block = src[sp_start:sp_end]
    assert "decrypt_if_needed" in sp_block, "search_patients 应解密敏感字段"

def test_get_patient_by_id_card_decrypts():
    import tools.patient_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    gp_start = src.find("def get_patient_by_id_card(")
    gp_end = src.find("def _llm_extract(", gp_start)
    gp_block = src[gp_start:gp_end]
    assert "decrypt_if_needed" in gp_block, "get_patient_by_id_card 应解密敏感字段"

# ========== 问题9: 数据库 UNIQUE 约束防重复 ==========
def test_patients_table_has_unique_constraint():
    import tools.patient_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert "UNIQUE KEY" in src or "UNIQUE" in src, "patients 表应含唯一约束"

# ========== 问题10: 信息追加合并 ==========
def test_merge_field_logic():
    from tools.patient_tool import PatientTool
    tool = PatientTool()
    # 利用 remember 中的内部逻辑来验证合并
    # 由于 _merge_field 是 remember 内部函数，我们通过源码验证
    import tools.patient_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert "def _merge_field(" in src, "应存在 _merge_field 函数"
    assert "_merge_field(allergy" in src, "allergy 应使用合并"
    assert "_merge_field(medication" in src, "medication 应使用合并"
    assert "_merge_field(symptoms" in src, "symptoms 应使用合并"

# ========== 问题11: 位置信息保留（rag_tool跳过改写） ==========
def test_rag_tool_location_keywords_skip_rewrite():
    from tools.rag_tool import RAGTool
    import inspect
    src = inspect.getsource(RAGTool.run)
    assert "location_keywords" in src, "RAGTool.run 应检测位置关键词"
    assert "location_keywords_detected" in src, "应标记跳过改写原因"

# ========== 问题12: 去重键改进（md5） ==========
def test_retriever_uses_md5_for_dedup():
    import tools.retriever as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert "hashlib.md5" in src, "retriever 应使用 md5 作为去重键"

# ========== 问题13: 重试策略细化 ==========
def test_base_tool_retry_only_api_errors():
    import tools.base_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert "retry_if_exception_type((APITimeoutError, APIConnectionError, APIError))" in src, "应只对 API 错误重试"
    # 不应再出现通用 Exception 重试
    bad_pattern = "retry_if_exception_type(Exception)"
    assert bad_pattern not in src, "不应使用通用 Exception 重试"

# ========== 问题14: 全局异常处理 ==========
def test_chat_has_global_exception_handler():
    import chat
    src = open(chat.__file__, "r", encoding="utf-8").read()
    assert "@app.exception_handler(Exception)" in src, "应注册全局异常处理器"
    assert 'content={"success": False, "error":' in src, "全局异常应返回统一 JSON"

# ========== 问题15: 配置集中化 ==========
def test_config_has_new_getters():
    from utils.config import (
        get_chunk_size, get_chunk_overlap, get_max_iterations,
        get_min_text_length, get_max_upload_size, get_allowed_upload_types
    )
    assert isinstance(get_chunk_size(), int)
    assert isinstance(get_chunk_overlap(), int)
    assert isinstance(get_max_iterations(), int)
    assert isinstance(get_min_text_length(), int)
    assert isinstance(get_max_upload_size(), int)
    assert isinstance(get_allowed_upload_types(), set)

# ========== 问题16: 模板方法模式 ==========
def test_base_tool_has_rag_pipeline_methods():
    import tools.base_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert "def _build_rag_prompt(" in src, "应存在 _build_rag_prompt"
    assert "def run_rag_pipeline(" in src, "应存在 run_rag_pipeline"

# ========== 问题17: 临时文件清理增强 ==========
def test_file_tool_has_safe_unlink():
    import tools.file_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert "os.unlink(tmp_path)" in src, "应删除临时文件"
    # 验证有 try-except 包裹 unlink
    assert "except OSError:" in src or "except Exception" in src, "unlink 应有异常处理"

# ========== 问题18: patient_tool 大改 ==========
def test_patient_tool_id_card_validation():
    import tools.patient_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert r"re.match(r'^(\d{15}|\d{17}[\dXx])$'" in src, "应强制校验身份证号格式"

def test_patient_tool_no_info_column_in_db():
    import tools.patient_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    # 建表语句中不应有 info 字段
    create_start = src.find("CREATE TABLE patients")
    create_end = src.find(")", create_start) + 1
    create_block = src[create_start:create_end]
    assert "info" not in create_block, "建表语句不应包含 info 字段"
    # SELECT 语句中不应再查询 info（但允许 info 局部变量存在）
    # 简单检查：不再出现 ", info, doctor_id"
    assert ", info, doctor_id" not in src, "SELECT 不应包含 info 字段"

def test_patient_tool_merge_on_append_and_update():
    import tools.patient_tool as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    remember_start = src.find("def remember(")
    remember_end = src.find("def recall(", remember_start)
    remember_block = src[remember_start:remember_end]
    # append=True 和 append=False 时都应合并关键字段
    assert "_merge_field(allergy" in remember_block
    assert "_merge_field(medication" in remember_block
    assert "_merge_field(symptoms" in remember_block

# ========== 问题19: consult_graph max_iterations 环境变量 ==========
def test_consult_graph_reads_max_iterations_from_env():
    import agents.consult_graph as mod
    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert 'os.getenv("MAX_ITERATIONS"' in src, "max_iterations 应从环境变量读取"
