import os
import re
import json
import pymysql
import time
from utils.response import build_response
from openai import OpenAI
from utils.audit import log_audit
from utils.crypto import encrypt_if_needed, decrypt_if_needed


class PatientTool:
    def __init__(self):
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_user = os.getenv("DB_USER", "root")
        self.db_password = os.getenv("DB_PASSWORD", "yourpassword")
        self.db_name = os.getenv("DB_NAME", "patient_db")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._init_db()

    def _get_connection(self):
        return pymysql.connect(
            host=self.db_host,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            charset='utf8mb4'
        )

    def _init_db(self):
        """初始化表结构（支持结构化字段）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SHOW TABLES LIKE 'patients'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # 检查并添加缺失的字段
            for field, col_type in [
                ('gender', 'VARCHAR(10)'),
                ('age', 'VARCHAR(10)'),
                ('phone', 'VARCHAR(20)'),
                ('address', 'VARCHAR(200)'),
                ('allergy', 'VARCHAR(200)'),
                ('medication', 'VARCHAR(200)'),
                ('symptoms', 'VARCHAR(500)'),
                ('diagnosis', 'VARCHAR(500)')
            ]:
                cursor.execute(f"SHOW COLUMNS FROM patients LIKE '{field}'")
                if not cursor.fetchone():
                    cursor.execute(f"ALTER TABLE patients ADD COLUMN {field} {col_type}")
        else:
            cursor.execute('''
                CREATE TABLE patients (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    gender VARCHAR(10),
                    age VARCHAR(10),
                    id_card VARCHAR(20),
                    phone VARCHAR(20),
                    address VARCHAR(200),
                    allergy VARCHAR(200),
                    medication VARCHAR(200),
                    symptoms VARCHAR(500),
                    diagnosis VARCHAR(500),
                    info TEXT,
                    tenant_id VARCHAR(50) DEFAULT 'default',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY idx_name_idcard (name, id_card,  doctor_id),
                    INDEX idx_tenant (tenant_id),
                    INDEX idx_doctor (doctor_id)
                )
            ''')
        
        conn.commit()
        conn.close()

    def _get_tenant(self):
        try:
            import chat
            if hasattr(chat, 'get_current_tenant'):
                return chat.get_current_tenant()
        except ImportError:
            pass
        return "default"

     # 🆕 获取当前医生 ID
    def _get_doctor_id(self) -> str:
        try:
            import chat
            if hasattr(chat, 'current_session_user') and chat.current_session_user:
                return chat.current_session_user
        except ImportError:
            pass
        return "default"

    def _get_existing_patient(self, name: str, id_card: str = None) -> dict:
        """
        根据姓名和身份证号查询已有记录
        如果传入 id_card 但查不到，尝试仅用 name 查询（兼容旧数据）
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        tenant_id = self._get_tenant()
        doctor_id = self._get_doctor_id()
        
        if id_card:
            # 先用 name + id_card 查询
            cursor.execute(
                "SELECT name, gender, age, id_card, phone, address, allergy, medication, symptoms, diagnosis, info, doctor_id FROM patients WHERE name = %s AND id_card = %s AND tenant_id = %s AND doctor_id = %s",
                (name, id_card, tenant_id, doctor_id)
            )
            row = cursor.fetchone()
            if row:
                conn.close()
                return {
                    "name": row[0],
                    "gender": row[1] or "",
                    "age": row[2] or "",
                    "id_card":decrypt_if_needed(row[3] or ""),
                    "phone": decrypt_if_needed(row[4] or ""),
                    "address": row[5] or "",
                    "allergy": row[6] or "",
                    "medication": row[7] or "",
                    "symptoms": row[8] or "",
                    "diagnosis": row[9] or "",
                    "info": row[10] or "",
                    "doctor_id": row[11] or ""
                }
            # 🆕 如果 name + id_card 查不到，尝试仅用 name 查询
            cursor.execute(
                "SELECT name, gender, age, id_card, phone, address, allergy, medication, symptoms, diagnosis, info, doctor_id FROM patients WHERE name = %s  AND tenant_id = %s AND doctor_id = %s",
                (name, tenant_id, doctor_id)
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                # 如果原有 id_card 为空，可以更新；否则可能是不同患者
                if not row[3]:  # id_card 为空
                    print(f"[PatientTool] 找到同名患者 {name}，原有 id_card 为空，将更新此记录")
                    return {
                        "name": row[0],
                        "gender": row[1] or "",
                        "age": row[2] or "",
                        "id_card":decrypt_if_needed(row[3] or ""),
                        "phone": decrypt_if_needed(row[4] or ""),
                        "address": row[5] or "",
                        "allergy": row[6] or "",
                        "medication": row[7] or "",
                        "symptoms": row[8] or "",
                        "diagnosis": row[9] or "",
                        "info": row[10] or "",
                        "doctor_id": row[11] or ""
                    }
                else:
                    print(f"[PatientTool] 警告：已存在同名患者 {name} 但 id_card 不同 ({row[3]})，请确认是否为同一人")
                    # 仍然返回，让调用方决定
                    return {
                        "name": row[0],
                        "gender": row[1] or "",
                        "age": row[2] or "",
                        "id_card":decrypt_if_needed(row[3] or ""),
                        "phone": decrypt_if_needed(row[4] or ""),
                        "address": row[5] or "",
                        "allergy": row[6] or "",
                        "medication": row[7] or "",
                        "symptoms": row[8] or "",
                        "diagnosis": row[9] or "",
                        "info": row[10] or "",
                        "doctor_id": row[11] or ""
                    }
        else:
            # 没有 id_card，仅用 name 查询
            cursor.execute(
                "SELECT name, gender, age, id_card, phone, address, allergy, medication, symptoms, diagnosis, info, doctor_id FROM patients WHERE name = %s AND tenant_id = %s AND doctor_id = %s",
                (name, tenant_id, doctor_id)
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "name": row[0],
                    "gender": row[1] or "",
                    "age": row[2] or "",
                    "id_card":decrypt_if_needed(row[3] or ""),
                    "phone": decrypt_if_needed(row[4] or ""),
                    "address": row[5] or "",
                    "allergy": row[6] or "",
                    "medication": row[7] or "",
                    "symptoms": row[8] or "",
                    "diagnosis": row[9] or "",
                    "info": row[10] or "",
                    "doctor_id": row[11] or ""
                }
        
        return None

    def check_missing_fields(self, data: dict) -> list:
        """
        检查缺失字段
        返回缺失字段列表，每个元素包含字段名和提示
        """
        required_fields = [
            {"key": "gender", "label": "性别", "example": "男/女"},
            {"key": "age", "label": "年龄", "example": "30岁"},
            {"key": "phone", "label": "联系方式", "example": "17688909987"},
            {"key": "address", "label": "家庭住址", "example": "杭州市滨江区"},
            {"key": "allergy", "label": "过敏史", "example": "无过敏 / 对青霉素过敏"},
            {"key": "medication", "label": "用药史", "example": "未服药 / 服用感冒灵颗粒"},
            {"key": "symptoms", "label": "症状描述", "example": "头痛、发热、腹泻"},
            {"key": "diagnosis", "label": "临床诊断", "example": "高血压 / 感冒"},
            {"key": "id_card", "label": "身份证号", "example": "410123199001011234"},  # 🆕 新增
        ]
        
        missing = []
        for field in required_fields:
            value = data.get(field["key"])
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing.append(field)
        return missing

    def remember(self, name: str, info: str = None, append: bool = False, 
             id_card: str = None, gender: str = None, age: str = None,
             phone: str = None, address: str = None, allergy: str = None,
             medication: str = None, symptoms: str = None, diagnosis: str = None) -> dict:
        """
        保存患者信息（结构化字段 + 唯一确认）
        """
        print(f"[PatientTool.remember] 开始执行")
        print(f"[PatientTool.remember] name: {name}, id_card: {id_card}")
        print(f"[PatientTool.remember] 传入的 info: {info[:100] if info else 'None'}...")
        print(f"[PatientTool.remember] 结构化参数: gender={gender}, age={age}, phone={phone}, address={address}")
        print(f"[PatientTool.remember] 结构化参数: allergy={allergy}, medication={medication}, symptoms={symptoms}, diagnosis={diagnosis}")

        name = name.strip()
        tenant_id = self._get_tenant()
        doctor_id = self._get_doctor_id()

        # 🆕 对敏感字段加密
        id_card_encrypted = encrypt_if_needed(id_card) if id_card else None
        phone_encrypted = encrypt_if_needed(phone) if phone else None
        print(f"[PatientTool.remember] tenant_id: {tenant_id}, doctor_id: {doctor_id}")

        # ===== 🆕 如果结构化字段为空，用 LLM 从 info 中解析 =====
        if info and not any([gender, age, phone, address, allergy, medication, symptoms, diagnosis, id_card]):
            print(f"[PatientTool.remember] 结构化字段为空，调用 LLM 解析 info")
            parsed = self._parse_info_with_llm(info)
            print(f"[PatientTool.remember] LLM 解析结果: {parsed}")
            
            if parsed.get("gender"): gender = parsed["gender"]
            if parsed.get("age"): age = parsed["age"]
            if parsed.get("phone"): phone = parsed["phone"]
            if parsed.get("address"): address = parsed["address"]
            if parsed.get("allergy"): allergy = parsed["allergy"]
            if parsed.get("medication"): medication = parsed["medication"]
            if parsed.get("symptoms"): symptoms = parsed["symptoms"]
            if parsed.get("diagnosis"): diagnosis = parsed["diagnosis"]
            if parsed.get("id_card"): id_card = parsed["id_card"]
            # 重新加密
            id_card_encrypted = encrypt_if_needed(id_card) if id_card else None
            phone_encrypted = encrypt_if_needed(phone) if phone else None
        
        # 如果没有身份证号，提示
        if not id_card:
            print("[PatientTool.remember] 警告：未提供身份证号，仅使用姓名匹配")
        
        # 组装 info（便于显示）
        info_parts = []
        if gender: info_parts.append(f"性别：{gender}")
        if age: info_parts.append(f"年龄：{age}")
        if phone: info_parts.append(f"联系方式：{phone}")
        if address: info_parts.append(f"家庭住址：{address}")
        if allergy: info_parts.append(f"过敏史：{allergy}")
        if medication: info_parts.append(f"用药史：{medication}")
        if symptoms: info_parts.append(f"症状：{symptoms}")
        if diagnosis: info_parts.append(f"诊断：{diagnosis}")
        if id_card: info_parts.append(f"身份证号：{id_card}")
        new_info = "，".join(info_parts) if info_parts else (info or "无详细信息")

        # 查询已有记录（唯一确认）
        existing = self._get_existing_patient(name, id_card)
        
        conn = self._get_connection()
        cursor = conn.cursor()

        if existing:
            # 更新已有记录（合并字段）
            if append:
                # 追加模式：保留原有值，补充新值
                final_gender = gender if gender and gender.strip() != "" else existing.get("gender", "")
                final_age = age if age and age.strip() != "" else existing.get("age", "")
                final_id_card = id_card_encrypted if id_card_encrypted else existing.get("id_card", "")
                final_phone = phone_encrypted if phone_encrypted else existing.get("phone", "")
                final_address = address if address and address.strip() != "" else existing.get("address", "")
                final_allergy = allergy if allergy and allergy.strip() != "" else existing.get("allergy", "")
                final_medication = medication if medication and medication.strip() != "" else existing.get("medication", "")
                final_symptoms = symptoms if symptoms and symptoms.strip() != "" else existing.get("symptoms", "")
                final_diagnosis = diagnosis if diagnosis and diagnosis.strip() != "" else existing.get("diagnosis", "")
                final_info = existing.get("info", "") + f"\n\n[追加于 {time.strftime('%Y-%m-%d %H:%M:%S')}] {new_info}"
            else:
                # 覆盖模式：用新值覆盖
                final_gender = gender if gender and gender.strip() != "" else existing.get("gender", "")
                final_age = age if age and age.strip() != "" else existing.get("age", "")
                final_id_card = id_card_encrypted if id_card_encrypted else existing.get("id_card", "")
                final_phone = phone_encrypted if phone_encrypted else existing.get("phone", "")
                final_address = address if address and address.strip() != "" else existing.get("address", "")
                final_allergy = allergy if allergy and allergy.strip() != "" else existing.get("allergy", "")
                final_medication = medication if medication and medication.strip() != "" else existing.get("medication", "")
                final_symptoms = symptoms if symptoms and symptoms.strip() != "" else existing.get("symptoms", "")
                final_diagnosis = diagnosis if diagnosis and diagnosis.strip() != "" else existing.get("diagnosis", "")
                final_info = existing.get("info", "") + f"\n\n[追加于 {time.strftime('%Y-%m-%d %H:%M:%S')}] {new_info}"

            existing_id_card = existing.get("id_card")

            if existing_id_card:
                sql = """
                    UPDATE patients SET 
                        gender = %s, age = %s, id_card = %s, phone = %s, 
                        address = %s, allergy = %s, medication = %s, 
                        symptoms = %s, diagnosis = %s, info = %s, 
                        updated_at = CURRENT_TIMESTAMP 
                    WHERE name = %s AND id_card = %s AND tenant_id = %s AND doctor_id = %s
                """
                params = (final_gender, final_age, final_id_card, final_phone, 
                        final_address, final_allergy, final_medication, 
                        final_symptoms, final_diagnosis, final_info, 
                        name, existing_id_card, tenant_id, doctor_id)
            else:
                sql = """
                    UPDATE patients SET 
                        gender = %s, age = %s, id_card = %s, phone = %s, 
                        address = %s, allergy = %s, medication = %s, 
                        symptoms = %s, diagnosis = %s, info = %s, 
                        updated_at = CURRENT_TIMESTAMP 
                    WHERE name = %s AND tenant_id = %s AND doctor_id = %s
                """
                params = (final_gender, final_age, final_id_card, final_phone, 
                        final_address, final_allergy, final_medication, 
                        final_symptoms, final_diagnosis, final_info, 
                        name, tenant_id, doctor_id)
            cursor.execute(sql, params)
            print(f"[PatientTool.remember] 更新影响行数: {cursor.rowcount}")
            action = "更新"
        else:
            # 🆕 插入新记录（使用原始字段，不是 final_*）
            sql = """
                INSERT INTO patients 
                (name, gender, age, id_card, phone, address, allergy, medication, symptoms, diagnosis, info, tenant_id, doctor_id) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (name, gender, age, id_card_encrypted, phone_encrypted, 
                    address, allergy, medication, symptoms, diagnosis, 
                    new_info, tenant_id, doctor_id)
            cursor.execute(sql, params)
            print(f"[PatientTool.remember] 插入影响行数: {cursor.rowcount}")
            action = "新建"

        conn.commit()
        conn.close()
        print(f"[PatientTool.remember] 操作完成，action={action}")

        # 🆕 构建返回数据（解密后返回给用户）
        if existing:
            # 更新时使用 final_* 变量
            saved_data = {
                "name": name,
                "gender": final_gender,
                "age": final_age,
                "id_card": decrypt_if_needed(final_id_card) if final_id_card else "",
                "phone": decrypt_if_needed(final_phone) if final_phone else "",
                "address": final_address,
                "allergy": final_allergy,
                "medication": final_medication,
                "symptoms": final_symptoms,
                "diagnosis": final_diagnosis
            }
        else:
            # 新建时使用原始字段
            saved_data = {
                "name": name,
                "gender": gender or "",
                "age": age or "",
                "id_card": decrypt_if_needed(id_card_encrypted) if id_card_encrypted else "",
                "phone": decrypt_if_needed(phone_encrypted) if phone_encrypted else "",
                "address": address or "",
                "allergy": allergy or "",
                "medication": medication or "",
                "symptoms": symptoms or "",
                "diagnosis": diagnosis or ""
            }
        
        missing = self.check_missing_fields(saved_data)
        
        answer = f"✅ 已{action}患者 {name} 的信息"
        
        recorded = []
        if saved_data.get("gender"): recorded.append(f"性别：{saved_data['gender']}")
        if saved_data.get("age"): recorded.append(f"年龄：{saved_data['age']}")
        if saved_data.get("phone"): recorded.append(f"联系方式：{saved_data['phone']}")
        if saved_data.get("address"): recorded.append(f"家庭住址：{saved_data['address']}")
        if saved_data.get("allergy"): recorded.append(f"过敏史：{saved_data['allergy']}")
        if saved_data.get("medication"): recorded.append(f"用药史：{saved_data['medication']}")
        if saved_data.get("symptoms"): recorded.append(f"症状：{saved_data['symptoms']}")
        if saved_data.get("diagnosis"): recorded.append(f"诊断：{saved_data['diagnosis']}")
        if saved_data.get("id_card"): recorded.append(f"身份证号：{saved_data['id_card']}")
        
        if recorded:
            answer += "\n\n📋 当前已记录：\n- " + "\n- ".join(recorded)
        
        if missing:
            answer += "\n\n⚠️ 以下信息缺失，建议补充："
            for field in missing:
                answer += f"\n- {field['label']}（如：{field['example']}）"
            answer += "\n\n💡 可以使用以下命令补充："
            answer += f"\n记住患者 {name}：<补充信息>"
            answer += f"\n或：追加患者 {name}：联系方式 17688909987，过敏史 无"

        return build_response(
            answer=answer,
            source="patient",
            debug={
                "name": name,
                "id_card": decrypt_if_needed(id_card_encrypted) if id_card_encrypted else "",
                "doctor_id": doctor_id,
                "tenant_id": tenant_id,
                "action": action,
                "missing_fields": [f["key"] for f in missing],
                "saved_data": saved_data
            }
        )

    def recall(self, name: str, id_card: str = None) -> dict:
        """查询患者信息"""
        name = name.strip()
        return self._get_existing_patient(name, id_card)

    def search_patients(self, name: str) -> list:
        """搜索患者"""
        name = name.strip()
        conn = self._get_connection()
        cursor = conn.cursor()
        tenant_id = self._get_tenant()
        doctor_id = self._get_doctor_id()
        cursor.execute(
            "SELECT name, gender, age, id_card, phone, address, allergy, medication, symptoms, diagnosis, info, doctor_id FROM patients WHERE name LIKE %s AND tenant_id = %s AND doctor_id = %s",
            (f"%{name}%", tenant_id, doctor_id)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "name": row[0],
                "gender": row[1] or "",
                "age": row[2] or "",
                "id_card": row[3] or "",
                "phone": row[4] or "",
                "address": row[5] or "",
                "allergy": row[6] or "",
                "medication": row[7] or "",
                "symptoms": row[8] or "",
                "diagnosis": row[9] or "",
                "info": row[10] or "",
                "doctor_id": row[11] or ""
            }
            for row in rows
        ]

    def get_patient_by_id_card(self, id_card: str) -> dict:
        """根据身份证号查询"""
        conn = self._get_connection()
        cursor = conn.cursor()
        tenant_id = self._get_tenant()
        doctor_id = self._get_doctor_id()
        cursor.execute(
            "SELECT name, gender, age, id_card, phone, address, allergy, medication, symptoms, diagnosis, info, doctor_id FROM patients WHERE id_card = %s AND tenant_id = %s AND doctor_id = %s",
            (id_card, tenant_id, doctor_id)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "name": row[0],
                "gender": row[1] or "",
                "age": row[2] or "",
                "id_card": row[3] or "",
                "phone": row[4] or "",
                "address": row[5] or "",
                "allergy": row[6] or "",
                "medication": row[7] or "",
                "symptoms": row[8] or "",
                "diagnosis": row[9] or "",
                "info": row[10] or "",
                "doctor_id": row[11] or ""
            }
        return None

    def _llm_extract(self, query: str) -> dict:
        """
        使用 LLM 提取患者操作信息和结构化字段
        """
        print(f"[PatientTool] ==== _llm_extract 开始 ====")
        print(f"[PatientTool._llm_extract] 原始输入: {query}")
        try:
            prompt = f"""
    请从以下用户输入中提取患者操作信息，返回 JSON 格式。
    输入：{query}

    判断规则：
    1. 如果用户想保存/记住患者信息 → action = "remember"
    2. 如果用户想追加/补充患者信息 → action = "append"
    3. 如果用户想查询患者信息 → action = "query"
    4. 如果与患者无关 → action = "unknown"

    提取以下字段（如果存在）：
    - name: 患者姓名（中文名，如张三、李四、王小波）
    - gender: 性别（男/女）
    - age: 年龄（如：30岁、30、32）
    - phone: 联系方式（手机号，如17688987678）
    - address: 家庭住址（有明确的地名，比如省份、市县等信息）
    - allergy: 过敏史
    - medication: 当前用药史
    - symptoms: 症状描述
    - diagnosis: 临床诊断
    - id_card: 身份证号（18位数字）
    - info: 除上述字段外的其他描述信息

    输出必须为 JSON 格式，如：
    {{"action": "remember", "name": "张三", "gender": "女", "age": "20岁", "phone": "", "address": "", "allergy": "", "medication": "", "symptoms": "", "diagnosis": "", "id_card": "", "info": ""}}
    """
            print(f"[PatientTool._llm_extract] 发送 prompt:\n{prompt}")
            resp = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            print(f"[PatientTool._llm_extract] LLM 原始返回: {content}")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            print(f"[PatientTool._llm_extract] 解析后的 JSON: {data}")
            print(f"[PatientTool] ==== _llm_extract 结束 ====")
            return data
        except Exception as e:
            print(f"[PatientTool._llm_extract] LLM 提取失败: {e}")
            import traceback
            traceback.print_exc()
            return {"action": "unknown", "name": "", "info": ""}

    def run(self, query: str) -> dict:
        """统一入口"""
        print(f"[PatientTool] ==== run 开始 ====")
        print(f"[PatientTool] 原始 query: {query}")
        # 身份证号检测
        id_match = re.fullmatch(r'\s*(\d{17}[\dXx])\s*', query)
        if id_match:
            id_card = id_match.group(1)
            return build_response(
                answer=f"✅ 检测到身份证号：{id_card}\n请先指定患者姓名，例如：记住患者 张三：身份证号 {id_card}",
                source="patient",
                debug={"id_card": id_card}
            )

        # 🆕 使用 LLM 提取结构化数据
        extracted = self._llm_extract(query)
        print(f"[PatientTool] LLM 提取结果: {extracted}")
        
        action = extracted.get("action", "unknown")
        name = extracted.get("name", "").strip()
        
        # 如果 LLM 没提取到 name，尝试从 query 中正则提取
        if not name:
            name_match = re.search(r'([\u4e00-\u9fa5]{2,4})', query)
            if name_match:
                name = name_match.group(1)

            # 🆕 从 LLM 提取结果中获取结构化字段
            gender = extracted.get("gender")
            age = extracted.get("age")
            phone = extracted.get("phone")
            address = extracted.get("address")
            allergy = extracted.get("allergy")
            medication = extracted.get("medication")
            symptoms = extracted.get("symptoms")
            diagnosis = extracted.get("diagnosis")
            id_card = extracted.get("id_card")
            info = extracted.get("info", "")
            
            # 如果 info 为空，组装一个
            if not info:
                info_parts = []
                if gender: info_parts.append(f"性别：{gender}")
                if age: info_parts.append(f"年龄：{age}")
                if phone: info_parts.append(f"联系方式：{phone}")
                if address: info_parts.append(f"家庭住址：{address}")
                if allergy: info_parts.append(f"过敏史：{allergy}")
                if medication: info_parts.append(f"用药史：{medication}")
                if symptoms: info_parts.append(f"症状：{symptoms}")
                if diagnosis: info_parts.append(f"诊断：{diagnosis}")
                if id_card: info_parts.append(f"身份证号：{id_card}")
                info = "，".join(info_parts)
            
            # 如果 info 还是空，用原始 query
            if not info:
                info = query

            if action == "remember":
                if not name:
                    return build_response(
                        answer="❌ 无法提取患者姓名，请使用格式：记住患者 张三：60岁，男，肚子痛",
                        source="patient",
                        success=False
                    )
            
                return self.remember(
                    name=name,
                    info=info,
                    append=False,
                    id_card=id_card,
                    gender=gender,
                    age=age,
                    phone=phone,
                    address=address,
                    allergy=allergy,
                    medication=medication,
                    symptoms=symptoms,
                    diagnosis=diagnosis
                )
            
            elif action == "append":
                if not name:
                    return build_response(
                        answer="❌ 无法提取患者姓名，请使用格式：追加患者 张三：今天吃了健胃消食片",
                        source="patient",
                        success=False
                    )
                
                return self.remember(
                    name=name,
                    info=info,
                    append=True,
                    id_card=id_card,
                    gender=gender,
                    age=age,
                    phone=phone,
                    address=address,
                    allergy=allergy,
                    medication=medication,
                    symptoms=symptoms,
                    diagnosis=diagnosis
                )
        
            elif action == "query":
                if not name:
                    return build_response(
                        answer="❌ 请指定要查询的患者姓名，例如：张三的信息",
                        source="patient",
                        success=False
                )
            stored = self.recall(name)
            if stored:
                lines = [f"📋 患者 {name} 的档案："]
                if stored.get("gender"): lines.append(f"性别：{stored['gender']}")
                if stored.get("age"): lines.append(f"年龄：{stored['age']}")
                if stored.get("phone"): lines.append(f"联系方式：{stored['phone']}")
                if stored.get("address"): lines.append(f"家庭住址：{stored['address']}")
                if stored.get("allergy"): lines.append(f"过敏史：{stored['allergy']}")
                if stored.get("medication"): lines.append(f"用药史：{stored['medication']}")
                if stored.get("symptoms"): lines.append(f"症状：{stored['symptoms']}")
                if stored.get("diagnosis"): lines.append(f"诊断：{stored['diagnosis']}")
                if stored.get("id_card"): lines.append(f"身份证号：{stored['id_card']}")
                return build_response(
                    answer="\n".join(lines),
                    source="patient",
                    debug={"name": name}
                )
            else:
                return build_response(
                    answer=f"❌ 未找到患者 {name} 的信息，请先使用「记住患者 {name}：信息」录入。",
                    source="patient",
                    success=False
                )
        else:
            return None

    def _parse_info_with_llm(self, info: str) -> dict:
        """
        使用 LLM 从 info 文本中提取结构化字段
        """
        print(f"[PatientTool] ==== _parse_info_with_llm 开始 ====")
        print(f"[PatientTool] 待解析的 info: {info}")
        
        try:
            prompt = f"""
    请从以下患者信息中提取结构化字段，返回 JSON 格式。
    输入：{info}

    提取字段：
    - gender: 性别（男/女），未提到则留空
    - age: 年龄（如：35岁），未提到则留空
    - phone: 联系方式（手机号，11位数字，以1开头），未提到则留空
    - address: 家庭住址（含省/市/区），未提到则留空
    - allergy: 过敏史，**只有用户明确说"无过敏"或"无过敏史"才填"无"，否则留空**
    - medication: 当前用药史（药物名称），未提到则留空
    - symptoms: 症状描述（如：肚子痛、头痛、发热），未提到则留空
    - diagnosis: 临床诊断（如：急性肠胃炎、感冒），未提到则留空
    - id_card: 身份证号（18位），未提到则留空

    **注意**：
    1. 手机号是11位数字，以1开头，如 17688909987
    2. 如果用户只说了"无过敏"，填"无"
    3. 如果用户完全没提过敏史，留空
    4. 诊断和症状的区别：诊断是医生的结论（如"急性肠胃炎"），症状是患者的主观感受（如"肚子痛"）

    只输出 JSON，不要其他内容。
    示例输出：
    {{"gender": "男", "age": "35岁", "phone": "17688909987", "address": "杭州市", "allergy": "无", "medication": "布洛芬", "symptoms": "肚子痛", "diagnosis": "", "id_card": ""}}
    """
            print(f"[PatientTool] 发送给 LLM 的 prompt:\n{prompt}")
            
            resp = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            print(f"[PatientTool] LLM 原始返回: {content}")
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            print(f"[PatientTool] 解析后的 JSON: {data}")
            return data
            
        except Exception as e:
            print(f"[PatientTool] LLM 解析失败: {e}")
            import traceback
            traceback.print_exc()
            return {}