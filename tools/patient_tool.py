import os
import json
import re
import pymysql
from utils.response import build_response
from openai import OpenAI
import time


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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES LIKE 'patients'")
        table_exists = cursor.fetchone()
        if table_exists:
            cursor.execute("SHOW COLUMNS FROM patients LIKE 'id_card'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE patients ADD COLUMN id_card VARCHAR(20)")
            cursor.execute("SHOW COLUMNS FROM patients LIKE 'diagnosis'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE patients ADD COLUMN diagnosis TEXT")
        else:
            cursor.execute('''
                CREATE TABLE patients (
                    name VARCHAR(100) NOT NULL,
                    id_card VARCHAR(20),
                    info TEXT NOT NULL,
                    diagnosis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (name, id_card)
                )
            ''')
        conn.commit()
        conn.close()

    def _llm_extract(self, query: str) -> dict:
        system_prompt = """你是一个信息提取助手。从用户输入中提取患者相关操作。

判断规则：
1. 如果用户想保存/记住患者信息 → action = "remember"
2. 如果用户想追加/补充患者信息 → action = "append"
3. 如果用户想查询患者信息 → action = "query"
4. 如果与患者无关 → action = "unknown"

重要：请准确区分药物名称和中文姓名。
- "布洛芬"、"头孢"、"阿莫西林"等是药物，不是患者姓名 → action = "unknown"
- "张三"、"李四"等是常见中文姓名，若上下文涉及查询或保存，则 action = "query" 或 "remember"

提取姓名（中文姓名，2-4个字）和描述信息（除姓名外的所有相关描述）。

输出必须为 JSON 格式，如：
{"action": "remember", "name": "张三", "info": "60岁，男，肚子痛，未吃药"}
{"action": "query", "name": "张三", "info": ""}
{"action": "unknown", "name": "", "info": ""}
"""
        try:
            resp = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            return data
        except Exception as e:
            print(f"[PatientTool] LLM extraction failed: {e}")
            import re
            if re.search(r'记住患者|记录患者|追加患者|补充患者', query):
                name_match = re.search(r'([\u4e00-\u9fa5]{2,4})', query)
                if name_match:
                    return {"action": "remember", "name": name_match.group(1), "info": query}
            if re.search(r'患者.*信息|信息.*患者', query):
                name_match = re.search(r'([\u4e00-\u9fa5]{2,4})', query)
                if name_match:
                    return {"action": "query", "name": name_match.group(1), "info": ""}
            return {"action": "unknown", "name": "", "info": ""}

    def remember(self, name: str, info: str, append: bool = True) -> dict:
        name = name.strip()
        info = info.strip()

        # 提取身份证号
        id_match = re.search(r'身份证号?\s*[:：]?\s*(\d{17}[\dXx])', info)
        id_card = id_match.group(1) if id_match else None

        # 提取诊断（可选）
        diagnosis_match = re.search(r'(?:诊断|病情|疾病)[:：]?\s*(.+)', info)
        new_diagnosis = diagnosis_match.group(1) if diagnosis_match else None

        conn = self._get_connection()
        cursor = conn.cursor()

        # 情况1：提供了身份证号
        if id_card:
            # 用身份证号查找
            cursor.execute("SELECT info, diagnosis FROM patients WHERE id_card = %s", (id_card,))
            existing = cursor.fetchone()
            if existing:
                # 找到了，更新该记录
                existing_info, existing_diagnosis = existing
                new_info = existing_info + f"\n\n[追加于 {time.strftime('%Y-%m-%d %H:%M:%S')}] {info}" if append else info
                # 合并诊断
                if new_diagnosis:
                    merged_diagnosis = existing_diagnosis + "、" + new_diagnosis if existing_diagnosis else new_diagnosis
                else:
                    merged_diagnosis = existing_diagnosis
                cursor.execute(
                    "UPDATE patients SET info = %s, diagnosis = %s, updated_at = CURRENT_TIMESTAMP WHERE id_card = %s",
                    (new_info, merged_diagnosis, id_card)
                )
                action = "更新"
            else:
                # 身份证号不存在，检查是否有同名记录
                cursor.execute("SELECT info, diagnosis FROM patients WHERE name = %s", (name,))
                existing = cursor.fetchone()
                if existing:
                    # 同名记录存在，更新该记录并补充身份证号
                    existing_info, existing_diagnosis = existing
                    new_info = existing_info + f"\n\n[追加于 {time.strftime('%Y-%m-%d %H:%M:%S')}] {info}" if append else info
                    if new_diagnosis:
                        merged_diagnosis = existing_diagnosis + "、" + new_diagnosis if existing_diagnosis else new_diagnosis
                    else:
                        merged_diagnosis = existing_diagnosis
                    cursor.execute(
                        "UPDATE patients SET info = %s, diagnosis = %s, id_card = %s, updated_at = CURRENT_TIMESTAMP WHERE name = %s",
                        (new_info, merged_diagnosis, id_card, name)
                    )
                    action = "更新（补充身份证号）"
                else:
                    # 完全新患者，插入
                    cursor.execute(
                        "INSERT INTO patients (name, id_card, info, diagnosis) VALUES (%s, %s, %s, %s)",
                        (name, id_card, info, new_diagnosis)
                    )
                    action = "新建"
        else:
            # 情况2：没有身份证号，只用姓名查找
            cursor.execute("SELECT info, diagnosis FROM patients WHERE name = %s", (name,))
            existing = cursor.fetchone()
            if existing:
                existing_info, existing_diagnosis = existing
                new_info = existing_info + f"\n\n[追加于 {time.strftime('%Y-%m-%d %H:%M:%S')}] {info}" if append else info
                if new_diagnosis:
                    merged_diagnosis = existing_diagnosis + "、" + new_diagnosis if existing_diagnosis else new_diagnosis
                else:
                    merged_diagnosis = existing_diagnosis
                cursor.execute(
                    "UPDATE patients SET info = %s, diagnosis = %s, updated_at = CURRENT_TIMESTAMP WHERE name = %s",
                    (new_info, merged_diagnosis, name)
                )
                action = "更新"
            else:
                cursor.execute(
                    "INSERT INTO patients (name, id_card, info, diagnosis) VALUES (%s, %s, %s, %s)",
                    (name, None, info, new_diagnosis)
                )
                action = "新建"

        conn.commit()
        conn.close()

        return build_response(
            answer=f"✅ 已{action}患者 {name} 的信息：{info}",
            source="patient",
            debug={"name": name, "id_card": id_card, "info": info, "diagnosis": new_diagnosis, "action": action}
        )

    def recall(self, name: str, id_card: str = None):
        name = name.strip()
        conn = self._get_connection()
        cursor = conn.cursor()
        if id_card:
            cursor.execute("SELECT info, diagnosis, id_card FROM patients WHERE name = %s AND id_card = %s", (name, id_card))
        else:
            cursor.execute("SELECT info, diagnosis, id_card FROM patients WHERE name = %s", (name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"info": row[0], "diagnosis": row[1], "id_card": row[2]}
        return None

    def search_patients(self, name: str) -> list:
        name = name.strip()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, id_card, info, diagnosis FROM patients WHERE name = %s", (name,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "name": row[0],
                "id_card": row[1] or "",
                "info": row[2],
                "diagnosis": row[3] or ""
            }
            for row in rows
        ]

    def get_patient_by_id_card(self, id_card: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, id_card, info, diagnosis FROM patients WHERE id_card = %s", (id_card,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"name": row[0], "id_card": row[1], "info": row[2], "diagnosis": row[3]}
        return None

    def run(self, query: str) -> dict:
        id_match = re.fullmatch(r'\s*(\d{17}[\dXx])\s*', query)
        if id_match:
            id_card = id_match.group(1)
            return build_response(
                answer=f"✅ 检测到身份证号：{id_card}\n请先指定患者姓名，例如：记住患者 张三：身份证号 {id_card}",
                source="patient",
                debug={"id_card": id_card}
            )

        extracted = self._llm_extract(query)
        action = extracted.get("action", "unknown")
        name = extracted.get("name", "").strip()
        info = extracted.get("info", "").strip()

        if action == "remember":
            if not name or not info:
                return build_response(
                    answer="❌ 无法提取患者姓名或信息，请使用格式：记住患者 张三：60岁，男，肚子痛",
                    source="patient",
                    success=False
                )
            return self.remember(name, info, append=True)
        elif action == "append":
            if not name or not info:
                return build_response(
                    answer="❌ 无法提取患者姓名或追加信息，请使用格式：追加患者 张三：今天吃了健胃消食片",
                    source="patient",
                    success=False
                )
            return self.remember(name, info, append=True)
        elif action == "query":
            if not name:
                return build_response(
                    answer="❌ 请指定要查询的患者姓名，例如：张三的信息",
                    source="patient",
                    success=False
                )
            stored = self.recall(name)
            if stored:
                return build_response(
                    answer=f"📋 患者 {name} 的档案：\n身份证号：{stored.get('id_card', '无')}\n诊断：{stored.get('diagnosis', '无')}\n详情：{stored['info']}",
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