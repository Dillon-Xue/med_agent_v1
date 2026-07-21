"""
集成测试专用 fixtures
连接真实数据库（patient_db），使用 test_tenant 隔离数据
"""

import os
import sys
import pytest
import uuid
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

# 使用用户提供的数据库配置
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "your_strong_password"
DB_NAME = "patient_db"
DB_PORT = 3306


@pytest.fixture(scope="session")
def test_db_connection():
    """
    直接连接到 patient_db，不创建新库
    """
    import pymysql

    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    yield conn
    conn.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_database(test_db_connection):
    """
    确保表结构完整（如果缺失则创建）
    使用生产环境的 init.sql
    """
    conn = test_db_connection
    cursor = conn.cursor()

    # 可选：执行 init.sql 确保表存在（如果表已存在则跳过）
    init_sql_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "init.sql"
    )

    if os.path.exists(init_sql_path):
        with open(init_sql_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
            statements = [s.strip() for s in sql_content.split(';') if s.strip()]
            for stmt in statements:
                try:
                    cursor.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"Warning: {e}")
        conn.commit()

    # 确保 audit_logs 表存在
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `audit_logs` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `user_id` VARCHAR(100) NOT NULL,
            `tenant_id` VARCHAR(50) DEFAULT 'default',
            `action` VARCHAR(50) NOT NULL,
            `resource_type` VARCHAR(50) NOT NULL,
            `resource_id` VARCHAR(100) DEFAULT NULL,
            `detail` TEXT DEFAULT NULL,
            `ip_address` VARCHAR(50) DEFAULT NULL,
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_tenant_action (tenant_id, action),
            INDEX idx_user_id (user_id),
            INDEX idx_resource (resource_type, resource_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.commit()

    yield

    # 测试结束后清理测试数据（只删除 test_tenant 的数据）
    cursor.execute("DELETE FROM approvals WHERE tenant_id = 'test_tenant'")
    cursor.execute("DELETE FROM patients WHERE tenant_id = 'test_tenant'")
    cursor.execute("DELETE FROM audit_logs WHERE tenant_id = 'test_tenant'")
    conn.commit()


@pytest.fixture
def test_tenant():
    return "test_tenant"


@pytest.fixture
def test_doctor():
    return "test_doctor"


@pytest.fixture
def test_patient_data():
    unique_suffix = str(uuid.uuid4().int)[:6]
    return {
        "name": f"测试患者_{unique_suffix}",
        "gender": "男",
        "age": "45岁",
        "id_card": f"11010119900101{unique_suffix[:4]}",
        "phone": f"1380000{unique_suffix}",
        "address": "北京市朝阳区测试路1号",
        "allergy": "无",
        "medication": "硝苯地平控释片 30mg qd",
        "symptoms": "头痛、发热",
        "diagnosis": "高血压"
    }


@pytest.fixture
def test_approval_data():
    return {
        "title": "用药方案评估审批：测试患者",
        "content": """
患者姓名：测试患者
年龄：45岁
性别：男
临床诊断：高血压
目前用药：硝苯地平控释片 30mg qd

评估结果：血压控制良好，建议继续当前用药方案。
用药目标：维持血压稳定，预防并发症。
用药注意事项：定期监测血压，避免低血压。
""",
        "type": "medication_evaluation",
        "requester": "doctor_李",
        "reviewer": "test_doctor"
    }


@pytest.fixture(autouse=True)
def setup_test_context(test_tenant, test_doctor):
    # 设置数据库环境变量（确保 database.py 使用正确凭据）
    os.environ["DB_USER"] = "root"
    os.environ["DB_PASSWORD"] = "your_strong_password"
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_PORT"] = "3306"
    os.environ["DB_NAME"] = "patient_db"

    from utils.thread_context import doctor_id_var, tenant_id_var
    original_user = doctor_id_var.get()
    doctor_id_var.set(test_doctor)
    original_tenant = tenant_id_var.get()
    tenant_id_var.set(test_tenant)

    yield

    if original_user:
        doctor_id_var.set(original_user)
    else:
        doctor_id_var.set("")
    if original_tenant:
        tenant_id_var.set(original_tenant)
    else:
        tenant_id_var.set("")


@pytest.fixture
def patient_tool():
    from tools.patient_tool import PatientTool
    return PatientTool()


@pytest.fixture
def approval_tool():
    from tools.approval_tool import ApprovalTool
    return ApprovalTool()


@pytest.fixture
def memory_tool():
    from tools.memory_tool import MemoryTool
    return MemoryTool()