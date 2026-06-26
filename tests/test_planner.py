"""
测试 agents/planner.py 的工具选择逻辑
覆盖：关键词规则、科室规则、姓名检测（run 方法）、历史继承、LLM 兜底
"""

import pytest
from unittest.mock import MagicMock, patch
from agents.planner import Planner
from agents.llm_planner import LLMPlanner


# ============================================================
# 测试用例 1：关键词规则匹配（rule_based）
# ============================================================
class TestRuleBasedKeywords:
    def test_drug_keyword(self):
        """副作用→drug"""
        planner = Planner(use_llm=False)
        tools = planner.rule_based("这个药有什么副作用")
        assert "drug" in tools
        # risk 不包含“副作用”，所以不期望 risk

    def test_guideline_keyword(self):
        """指南→guideline"""
        planner = Planner(use_llm=False)
        tools = planner.rule_based("临床指南怎么说")
        assert "guideline" in tools

    def test_risk_keyword(self):
        """相互作用→risk"""
        planner = Planner(use_llm=False)
        tools = planner.rule_based("药物相互作用")
        assert "risk" in tools

    def test_risk_alternative_keyword(self):
        """风险→risk"""
        planner = Planner(use_llm=False)
        tools = planner.rule_based("有什么风险")
        assert "risk" in tools


# ============================================================
# 测试用例 2：科室专属规则（rule_based）
# ============================================================
class TestSpecialtyRules:
    def test_cardiology_specialty(self):
        planner = Planner(use_llm=False, specialty="cardiology")
        tools = planner.rule_based("胸痛")
        assert "guideline" in tools
        assert "drug" in tools

    def test_pharmacy_specialty(self):
        planner = Planner(use_llm=False, specialty="pharmacy")
        tools = planner.rule_based("用药安全")
        assert "drug" in tools
        assert "risk" in tools

    def test_cardiology_extra_literature(self):
        planner = Planner(use_llm=False, specialty="cardiology")
        tools = planner.rule_based("支架术后")
        assert "literature" in tools


# ============================================================
# 测试用例 3：姓名误匹配修复（rule_based 不误加）
# ============================================================
class TestPatientMisdetection:
    def test_no_patient_for_symptoms(self):
        """
        症状词不应触发 patient（rule_based 已删除纯姓名检测）
        """
        planner = Planner(use_llm=False)
        for q in ["胸痛", "心脏搭桥", "感冒"]:
            tools = planner.rule_based(q)
            assert "patient" not in tools


# ============================================================
# 测试用例 4：run 方法中的 patient 逻辑
# ============================================================
class TestRunPatientLogic:
    def test_run_adds_patient_for_explicit_name_in_current(self):
        """
        当前问题含明确姓名模式（我是XXX），run 应添加 patient
        """
        planner = Planner(use_llm=False)
        # 模拟 run 方法内部会调用 _extract_patient_name_from_history 和检测当前问题
        # 我们直接测试 run 方法
        # 由于 run 会调用 rule_based 和 LLM，我们 mock rule_based 返回空，便于观察 patient 是否被添加
        with patch.object(planner, 'rule_based', return_value=[]):
            result = planner.run("我是张三，肚子痛")
            # 应该添加 patient（因为当前问题有“我是张三”）
            assert "patient" in result["tools"]

    def test_run_adds_patient_for_name_in_query_format(self):
        """“张三的信息”应添加 patient"""
        planner = Planner(use_llm=False)
        with patch.object(planner, 'rule_based', return_value=[]):
            result = planner.run("张三的信息")
            assert "patient" in result["tools"]

    def test_run_adds_patient_for_remember_command(self):
        """“记住患者 李四”应添加 patient"""
        planner = Planner(use_llm=False)
        # 这条会触发 run 中的“记住患者”优先检测，直接返回 ["patient"]
        result = planner.run("记住患者 李四：青霉素过敏")
        assert result["tools"] == ["patient"]

    def test_run_does_not_add_patient_for_symptoms(self):
        """症状词不应添加 patient"""
        planner = Planner(use_llm=False)
        with patch.object(planner, 'rule_based', return_value=[]):
            result = planner.run("肚子痛")
            assert "patient" not in result["tools"]

    def test_run_keeps_patient_from_history(self):
        """历史中有姓名时保留 patient"""
        planner = Planner(use_llm=False)
        # 构造完整 question，包含历史 user: 我是张三
        question = "对话历史：\nuser: 我是张三\nassistant: 好的\n当前问题：肚子痛"
        # 模拟 rule_based 返回空，让 run 决定添加 patient
        with patch.object(planner, 'rule_based', return_value=[]):
            result = planner.run(question)
            # 历史有姓名，应添加 patient
            assert "patient" in result["tools"]

    def test_run_removes_patient_if_no_history_and_no_name(self):
        """无历史且当前问题无姓名，移除 patient（即使 rule_based 误加）"""
        planner = Planner(use_llm=False)
        # 模拟 rule_based 返回含 patient（但正常情况下不会，这里测试移除逻辑）
        with patch.object(planner, 'rule_based', return_value=["patient"]):
            result = planner.run("肚子痛")
            # 应被移除
            assert "patient" not in result["tools"]


# ============================================================
# 测试用例 5：历史姓名提取（_extract_patient_name_from_history）
# ============================================================
class TestHistoryPatientExtraction:
    def test_extract_name_from_user(self):
        planner = Planner(use_llm=False)
        history = "对话历史：\nuser: 我是张三\nassistant: 好的\n当前问题：肚子痛"
        name = planner._extract_patient_name_from_history(history)
        assert name == "张三"

    def test_ignore_assistant_mentions(self):
        """assistant 回复中的“患者”不应被提取"""
        planner = Planner(use_llm=False)
        history = "对话历史：\nuser: 感冒了\nassistant: 术后患者属临床研究排除...\n当前问题：发烧"
        name = planner._extract_patient_name_from_history(history)
        assert name == ""


# ============================================================
# 测试用例 6：LLM 兜底分支（run 方法）
# ============================================================
class TestLLMFallback:
    def test_llm_called_when_zero_tools(self):
        """
        规则返回空时，LLM 被调用。
        """
        mock_llm_planner = MagicMock()
        mock_llm_planner.select_tools.return_value = ["drug"]
        planner = Planner(use_llm=True)
        planner.llm_planner = mock_llm_planner
        # 模拟 rule_based 返回空
        with patch.object(planner, 'rule_based', return_value=[]):
            result = planner.run("你好")
            mock_llm_planner.select_tools.assert_called_once()
            assert "drug" in result["tools"]

    def test_fallback_to_drug_when_llm_empty(self):
        """
        LLM 返回空列表，回退到 ["drug"]
        """
        mock_llm_planner = MagicMock()
        mock_llm_planner.select_tools.return_value = []
        planner = Planner(use_llm=True)
        planner.llm_planner = mock_llm_planner
        with patch.object(planner, 'rule_based', return_value=[]):
            result = planner.run("任意问题")
            assert result["tools"] == ["drug"]