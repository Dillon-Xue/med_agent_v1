import os
import re
from datetime import datetime
from docx import Document
from utils.response import build_response


class ReportTool:
    def __init__(self):
        self.template_path = "templates/登记表模板.docx"
        self.output_dir = "reports/"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, query: str) -> dict:
        """统一入口：生成评估表"""
        clean_query = re.sub(r'^对话历史：.*?当前问题：', '', query)
        match = re.search(r'(?:生成评估表|生成病历|生成档案|生成记录)\s*([\u4e00-\u9fa5]{2,4})', clean_query)
        if not match:
            match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*的(?:评估表|病历|档案)', clean_query)
        if not match:
            match = re.search(r'([\u4e00-\u9fa5]{2,4})', clean_query)
        if not match:
            return build_response(
                answer="❌ 请指定患者姓名，例如：生成评估表 张三",
                source="report",
                success=False
            )
        patient_name = match.group(1)

        from tools.tool_registry import get_tools
        tools = get_tools()
        patient_tool = tools.get("patient")
        candidates = patient_tool.search_patients(patient_name)

        if not candidates:
            return build_response(
                answer=f"❌ 未找到患者 {patient_name} 的档案，请先录入患者信息。",
                source="report",
                success=False
            )

        if len(candidates) == 1:
            return self._generate_from_candidate(candidates[0])

        return self._show_candidates(candidates, patient_name)

    def _show_candidates(self, candidates: list, name: str) -> dict:
        lines = [f"找到 {len(candidates)} 位名为 {name} 的患者，请选择："]
        for i, c in enumerate(candidates, 1):
            info_preview = c.get("info", "")[:40]
            lines.append(f"{i}. {c['name']}，{info_preview}...")
        lines.append("请回复数字（如 1）选择，或输入更详细的信息确认。")
        return build_response(
            answer="\n".join(lines),
            source="report",
            success=False,
            debug={"candidates": candidates}
        )

    def _generate_from_candidate(self, candidate: dict) -> dict:
        id_card = candidate.get("id_card", "")
        if not id_card:
            return build_response(
                answer=f"❌ 患者 {candidate['name']} 缺少身份证号，请先补充：\n记住患者 {candidate['name']}：身份证号 410123199001011234\n然后重新生成评估表。",
                source="report",
                success=False
            )

        info_dict = self._parse_patient_info(
            candidate.get("info", ""),
            candidate.get("diagnosis", "")
        )

        if info_dict.get("家庭住址"):
            info_dict["家庭住址"] = self._complete_address(info_dict["家庭住址"])

        drug_suggestion = self._get_drug_suggestion(candidate["name"], info_dict)
        assessment = self._generate_assessment(candidate["name"], info_dict, drug_suggestion)

        output_path = self._fill_template(
            candidate["name"],
            info_dict,
            assessment,
            id_card
        )

        download_url = f"/reports/{os.path.basename(output_path)}"
        return build_response(
            answer=f"✅ 评估表已生成：{candidate['name']}\n📎 [下载评估表]({download_url})",
            source="report",
            debug={
                "patient": candidate["name"],
                "output": output_path,
                "download_url": download_url
            }
        )

    def _complete_address(self, address: str) -> str:
        if not address:
            return address
        if re.search(r'省.*市', address):
            return address
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            prompt = f"""
请将以下中国城市地址补全为完整的“省份+城市”格式，只输出结果，不要添加任何解释。
输入地址：{address}
输出格式：省份+城市
示例：长沙市 → 湖南省长沙市
"""
            resp = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            full_address = resp.choices[0].message.content.strip()
            if full_address:
                return full_address
        except Exception as e:
            print(f"[ReportTool] 地址补全失败: {e}")
        return address

    def _get_drug_suggestion(self, name: str, info: dict) -> str:
        try:
            from tools.tool_registry import get_tools
            tools = get_tools()
            drug_tool = tools.get("drug")
            if not drug_tool:
                return ""
            query = f"{name}，{info.get('临床诊断', '')}，{info.get('主要问题', '')}，请推荐适合的药品"
            result = drug_tool.run(query)
            if result and isinstance(result, dict):
                return result.get("answer", "")
        except Exception as e:
            print(f"[ReportTool] 调用 drug tool 失败: {e}")
        return ""

    def _parse_patient_info(self, text: str, diagnosis: str = "") -> dict:
        info = {
            "姓名": "",
            "性别": "",
            "年龄": "",
            "联系方式": "",
            "家庭住址": "",
            "目前用药": "",
            "临床诊断": diagnosis or "",
            "主要问题": "",
        }

        # 1. 提取性别
        gender_match = re.search(r'(男|女|男性|女性)', text)
        if gender_match:
            info["性别"] = "男" if gender_match.group(1) in ["男", "男性"] else "女"

        # 2. 提取年龄
        age_match = re.search(r'(\d{1,3})\s*(?:岁|周岁|年)', text)
        if age_match:
            info["年龄"] = age_match.group(1) + "岁"

        # 3. 提取手机号
        phone_match = re.search(r'(1[3-9]\d{9})', text)
        if phone_match:
            info["联系方式"] = phone_match.group(1)

        # 4. 提取住址
        city_match = re.search(r'([\u4e00-\u9fa5]{2,6}市)', text)
        if city_match:
            info["家庭住址"] = city_match.group(1)

        # 5. 提取用药（包括“喝了咳嗽糖浆”等）
        med_patterns = [
            r'(?:喝了|吃了|服用|用了|打过|输过)\s*([^，,。；;.、]+)',
            r'用药\s*[:：]?\s*([^，,。；;.]+)',
        ]
        for pattern in med_patterns:
            med_match = re.search(pattern, text)
            if med_match:
                info["目前用药"] = med_match.group(1).strip()
                break

        if not info["目前用药"]:
            no_med_patterns = [
                r'没有吃\s*药', r'未\s*用药', r'不\s*吃药', r'无\s*用药', r'没\s*吃药',
                r'暂未\s*用药', r'暂未\s*吃药', r'尚未\s*用药', r'尚未\s*吃药',
                r'未服\s*药', r'未\s*服药', r'没\s*服\s*药'
            ]
            for pattern in no_med_patterns:
                if re.search(pattern, text):
                    info["目前用药"] = "无"
                    break

        # 6. 提取诊断（如果未提供）
        if not info["临床诊断"]:
            diag_match = re.search(r'(?:诊断|病情|疾病|症状)\s*[:：]?\s*([^，,。；;.]+)', text)
            if diag_match:
                info["临床诊断"] = diag_match.group(1).strip()
            else:
                symptom_list = ["失眠", "头痛", "发烧", "发热", "咳嗽", "感冒", "高血压", "糖尿病",
                                "心脏病", "胃痛", "腹痛", "腹泻", "恶心", "呕吐", "头晕", "乏力",
                                "胸闷", "气短", "心悸", "焦虑", "抑郁", "多梦", "耳鸣"]
                for symptom in symptom_list:
                    if symptom in text:
                        if info["临床诊断"]:
                            info["临床诊断"] += "、" + symptom
                        else:
                            info["临床诊断"] = symptom
                        if not info["主要问题"]:
                            info["主要问题"] = symptom
                        break

        if not info["主要问题"] and info["临床诊断"]:
            info["主要问题"] = info["临床诊断"].split("、")[0] if "、" in info["临床诊断"] else info["临床诊断"]

        if not info["临床诊断"]:
            info["临床诊断"] = "待进一步检查明确"

        return info

    def _generate_assessment(self, name: str, info: dict, drug_suggestion: str) -> dict:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        drug_reference = ""
        if drug_suggestion:
            drug_reference = f"【药物参考信息】\n{drug_suggestion}\n"

        prompt = f"""
请根据以下患者信息生成评估报告内容。

患者姓名：{name}
年龄：{info.get('年龄', '未知')}
性别：{info.get('性别', '未知')}
临床诊断：{info.get('临床诊断', '无')}
目前用药：{info.get('目前用药', '无')}
{drug_reference}
请生成以下四项内容，每项内容要简洁、专业，不要包含任何"助手"、"工具"、"来源"等标记。

1. 评估结果：简要概括患者当前的整体健康状况（40-60字）
2. 主要问题：从诊断中提取最核心的一个问题（20-30字）
3. 用药目标：根据患者诊断和目前用药，给出具体的用药目标和推荐方案。如果已有用药，写明继续或调整建议；如果无用药，结合药物参考信息给出具体药物推荐（包括药名和预期效果）。
4. 用药注意事项：根据患者情况，提醒用药过程中的关键注意事项（40-60字）

输出格式（每项占一行）：
评估结果：...
主要问题：...
用药目标：...
用药注意事项：...
"""
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = resp.choices[0].message.content
        result = {"评估结果": "", "主要问题": "", "用药目标": "", "用药注意事项": ""}
        for line in content.split("\n"):
            if "评估结果：" in line:
                result["评估结果"] = line.replace("评估结果：", "").strip()
            elif "主要问题：" in line:
                result["主要问题"] = line.replace("主要问题：", "").strip()
            elif "用药目标：" in line:
                result["用药目标"] = line.replace("用药目标：", "").strip()
            elif "用药注意事项：" in line:
                result["用药注意事项"] = line.replace("用药注意事项：", "").strip()
        return result

    def _fill_template(self, name: str, info: dict, assessment: dict, id_card: str) -> str:
        doc = Document(self.template_path)
        now = datetime.now()
        档案号 = f"DA{now.strftime('%Y%m%d%H%M%S')}"

        # 如果目前用药为空，设为“无”
        if not info.get("目前用药"):
            info["目前用药"] = "无"

        raw_map = {
            "档案号": 档案号,
            "姓名": name,
            "身份证号": id_card,
            "家庭住址": info.get("家庭住址", ""),
            "评估时间": now.strftime("%Y年%m月%d日"),
            "性别": info.get("性别", ""),
            "年龄": info.get("年龄", ""),
            "联系方式": info.get("联系方式", ""),
            "临床诊断": info.get("临床诊断", ""),
            "主要问题": assessment.get("主要问题", ""),
            "目前用药": info.get("目前用药", ""),
            "评估结果": assessment.get("评估结果", ""),
            "用药目标": assessment.get("用药目标", ""),
            "用药注意事项": assessment.get("用药注意事项", ""),
        }

        replace_map = {}
        for key, value in raw_map.items():
            replace_map[f"{{{{{key}}}}}"] = value
            replace_map[f"《{key}》"] = value

        for para in doc.paragraphs:
            for placeholder, value in replace_map.items():
                if placeholder in para.text:
                    para.text = para.text.replace(placeholder, value)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        for placeholder, value in replace_map.items():
                            if placeholder in para.text:
                                para.text = para.text.replace(placeholder, value)

        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"{name}_评估表_{timestamp}.docx")
        doc.save(output_path)
        return output_path