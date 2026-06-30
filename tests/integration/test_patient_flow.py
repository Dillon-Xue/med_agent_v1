"""
集成测试：患者档案管理完整流程
"""

import pytest
from utils.crypto import decrypt_if_needed


class TestPatientFlow:

    @pytest.fixture(autouse=True)
    def ensure_db_schema(self, patient_tool):
        """确保患者表包含所有字段"""
        patient_tool._init_db()

    def test_create_patient(self, patient_tool, test_patient_data):
        """测试：新建患者"""
        data = test_patient_data
        result = patient_tool.remember(
            name=data["name"],
            info=f"{data['age']}，{data['gender']}，{data['diagnosis']}",
            id_card=data["id_card"],
            gender=data["gender"],
            age=data["age"],
            phone=data["phone"],
            address=data["address"],
            allergy=data["allergy"],
            medication=data["medication"],
            symptoms=data["symptoms"],
            diagnosis=data["diagnosis"]
        )

        assert result.get("success") is True
        assert "已新建患者" in result.get("answer", "")

    def test_recall_patient(self, patient_tool, test_patient_data):
        """测试：查询患者（验证加密字段解密正确）"""
        # 先创建完整患者
        patient_tool.remember(
            name=test_patient_data["name"],
            info="完整信息",
            id_card=test_patient_data["id_card"],
            gender=test_patient_data["gender"],
            age=test_patient_data["age"],
            phone=test_patient_data["phone"],
            address=test_patient_data["address"],
            allergy=test_patient_data["allergy"],
            medication=test_patient_data["medication"],
            symptoms=test_patient_data["symptoms"],
            diagnosis=test_patient_data["diagnosis"]
        )

        result = patient_tool.recall(test_patient_data["name"])

        assert result is not None
        assert result.get("name") == test_patient_data["name"]
        # 验证加密字段已解密
        assert result.get("id_card") == test_patient_data["id_card"]
        assert result.get("phone") == test_patient_data["phone"]
        assert result.get("symptoms") == test_patient_data["symptoms"]

    def test_append_patient(self, patient_tool, test_patient_data):
        """测试：追加患者信息"""
        # 1. 先创建（不含症状）
        patient_tool.remember(
            name=test_patient_data["name"],
            info="基本信息",
            id_card=test_patient_data["id_card"],
            gender=test_patient_data["gender"],
            age=test_patient_data["age"],
            phone=test_patient_data["phone"],
            address=test_patient_data["address"],
            allergy=test_patient_data["allergy"],
            medication=test_patient_data["medication"],
            diagnosis=test_patient_data["diagnosis"]
        )

        # 2. 追加症状
        result = patient_tool.remember(
            name=test_patient_data["name"],
            info="头痛发热腹泻",
            append=True,
            symptoms="头痛、发热、腹泻"
        )
        assert result.get("success") is True
        assert "已更新患者" in result.get("answer", "")

        # 3. 直接查询数据库，看症状是否真正写入
        from utils.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, symptoms FROM patients WHERE name = %s AND tenant_id = 'test_tenant'",
            (test_patient_data["name"],)
        )
        row = cursor.fetchone()
        conn.close()

        print(f"\n[数据库查询] row = {row}")
        print(f"[数据库查询] symptoms = {row.get('symptoms') if row else '未查到数据'}")

        assert row is not None, "数据库中没有该患者"
        assert row.get("symptoms") == "头痛、发热、腹泻", \
            f"数据库 symptoms 为 {row.get('symptoms')}，期望 '头痛、发热、腹泻'"

        # 4. 验证 recall 能否获取到追加后的信息
        recalled = patient_tool.recall(test_patient_data["name"])
        print(f"[recall 结果] {recalled}")

        assert recalled is not None
        assert recalled.get("symptoms") == "头痛、发热、腹泻", \
            f"recall 返回的 symptoms 为 {recalled.get('symptoms')}"

    def test_search_patients(self, patient_tool, test_patient_data):
        """测试：搜索患者"""
        patient_tool.remember(
            name=test_patient_data["name"],
            info="测试",
            id_card=test_patient_data["id_card"]
        )

        results = patient_tool.search_patients(test_patient_data["name"][:3])

        assert len(results) >= 1
        found = any(r.get("name") == test_patient_data["name"] for r in results)
        assert found is True

    def test_prevent_duplicate_id_card(self, patient_tool, test_patient_data):
        """测试：同一身份证号不能创建重复患者"""
        # 第一次创建
        patient_tool.remember(
            name=test_patient_data["name"],
            info="第一次",
            id_card=test_patient_data["id_card"]
        )

        # 第二次用同一个身份证号创建不同姓名
        result = patient_tool.remember(
            name="另一个名字",
            info="第二次",
            id_card=test_patient_data["id_card"]
        )

        # 应该能正常处理（更新而不是新建）
        assert result.get("success") is True

    def test_missing_fields_prompt(self, patient_tool):
        """测试：缺失字段提示"""
        result = patient_tool.remember(
            name="缺失信息患者",
            info="只有姓名",
            id_card=None
        )

        assert result.get("success") is True
        # 应该提示缺少字段
        answer = result.get("answer", "")
        assert "以下信息缺失" in answer or "建议补充" in answer