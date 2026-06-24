import os
import re
import json
import asyncio
import concurrent.futures
import traceback
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from tools.tool_registry import get_tools
from openai import OpenAI

# 全局调试开关
DEBUG = True

class ConsultGraph:
    def __init__(self):
        self.tools = get_tools()
        self.planner = Planner()
        self.executor = Executor(self.tools)
        self.synthesizer = Synthesizer(api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("analyze_gap", self._analyze_gap)
        graph.add_node("ask_missing", self._ask_missing)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("synthesize", self._synthesize)

        graph.set_entry_point("analyze_gap")

        graph.add_conditional_edges(
            "analyze_gap",
            self._should_ask_or_execute,
            {
                "ask": "ask_missing",
                "execute": "execute_tools"
            }
        )
        graph.add_edge("ask_missing", END)
        graph.add_edge("execute_tools", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _log(self, *args, **kwargs):
        if DEBUG:
            print("[DEBUG]", *args, **kwargs)

    def _extract_info_from_text(self, text: str) -> dict:
        """使用 LLM 从文本中提取结构化患者信息，并用正则做补充提取"""
        # 初始化结果（包含新增字段）
        result = {
            "姓名": None, "年龄": None, "过敏史": None, "用药史": None, "id_card": None,
            "性别": None, "家庭住址": None, "联系方式": None
        }
        
        try:
            prompt = f"""
请从以下用户输入中提取患者的健康信息，返回 JSON 格式。
输入：{text}

需要提取的字段：
- name: 患者姓名（如果有“我是XX”或“我叫XX”则提取，否则为 null）
- age: 年龄（如“30岁”，如果没有则为 null）
- allergy: 过敏史（如果有过敏信息，提取具体过敏原，如“西红柿”；如果明确说“无过敏”则为“无”；如果没有提及则为 null）
- medication: 当前正在服用的药物（如果用户明确说“在服用XX”，提取药物名称；如果明确说“未服药”则为“无”；如果没有提及则为 null）
- id_card: 身份证号（如果提到身份证号/身份证/ID，提取18位数字，否则为 null）
- gender: 性别（男/女，如果没有则为 null）
- address: 家庭住址（如果有具体地址，如“杭州市滨江区”，否则为 null）
- phone: 联系方式（手机号，如“17688909987”，否则为 null）

重要规则：
1. 输入可能包含表格格式（如“姓名 | 年龄 | 过敏史”），请正确识别对应字段的值。
2. 如果字段显示为“无”、“未服药”、“未提及”、“null”等，请提取为“无”或 null。

只输出 JSON，不要其他内容。
"""
            resp = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            
            # 提取 LLM 结果，并统一转换为字符串
            for key, val in {
                "姓名": "name", "年龄": "age", "过敏史": "allergy",
                "用药史": "medication", "id_card": "id_card",
                "性别": "gender", "家庭住址": "address", "联系方式": "phone"
            }.items():
                raw = data.get(val)
                result[key] = str(raw) if raw is not None else None
            
            print(f"[ConsultGraph] LLM 提取结果: {result}")
            
        except Exception as e:
            print(f"[ConsultGraph] LLM 提取失败，降级到正则: {e}")
        
        # 正则补充提取（合并结果）
        regex_result = self._extract_info_with_regex(text)
        print(f"[ConsultGraph] 正则补充提取结果: {regex_result}")
        
        # 合并：如果 LLM 提取的字段为空，用正则的结果补充
        for key in ["姓名", "年龄", "过敏史", "用药史", "id_card", "性别", "家庭住址", "联系方式"]:
            if not result.get(key) and regex_result.get(key):
                result[key] = regex_result[key]
        
        print(f"[ConsultGraph] 最终提取结果: {result}")
        return result

    def _extract_info_with_regex(self, text: str) -> dict:
        """降级方案：使用正则提取（支持表格格式）"""
        print(f"[DEBUG] _extract_info_with_regex 被调用，text 长度: {len(text)}")
        print(f"[DEBUG] text 内容:\n{text[:500]}")
        info = {
            "姓名": None, "年龄": None, "过敏史": None, "用药史": None, "id_card": None,
            "性别": None, "家庭住址": None, "联系方式": None
        }

        # ===== 首先尝试从表格格式中提取 =====
        lines = text.strip().split('\n')
        header_row = None
        data_row = None

        for i, line in enumerate(lines):
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if any(kw in line for kw in ['姓名', '年龄', '过敏史', '身份证', '用药', '服药', '症状', '性别', '住址', '地址', '联系方式', '电话']):
                    header_row = parts
                    if i + 1 < len(lines):
                        next_parts = [p.strip() for p in lines[i + 1].split('|') if p.strip()]
                        if len(next_parts) >= len(header_row) - 1:
                            data_row = next_parts
                    break

        if header_row and data_row:
            for idx, field in enumerate(header_row):
                if idx < len(data_row):
                    value = data_row[idx]
                    if '姓名' in field and not info["姓名"]:
                        info["姓名"] = value
                    elif '年龄' in field and not info["年龄"]:
                        info["年龄"] = value
                    elif '过敏' in field and not info["过敏史"]:
                        info["过敏史"] = value
                    elif '身份证' in field and not info["id_card"]:
                        info["id_card"] = value
                    elif '用药' in field or '服药' in field:
                        if not info["用药史"]:
                            info["用药史"] = value
                    elif '性别' in field and not info["性别"]:
                        info["性别"] = value
                    elif '家庭住址' in field or '地址' in field:
                        if not info["家庭住址"]:
                            info["家庭住址"] = value
                    elif '联系方式' in field or '电话' in field:
                        if not info["联系方式"]:
                            info["联系方式"] = value

        # ===== 如果表格提取失败，回退到原有正则逻辑 =====
        if not info["姓名"]:
            name_match = re.search(r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})', text)
            if name_match:
                info["姓名"] = name_match.group(1)

        if not info["年龄"]:
            age_match = re.search(r'(\d{1,3})\s*岁', text)
            if age_match:
                info["年龄"] = age_match.group(1) + "岁"

        if not info["过敏史"]:
            if "过敏" in text:
                if re.search(r'(无|没有|否认)\s*过敏', text):
                    info["过敏史"] = "无"
                else:
                    allergy_match = re.search(r'对\s*([\u4e00-\u9fa5]{2,6})\s*过敏', text)
                    if allergy_match:
                        info["过敏史"] = "对" + allergy_match.group(1) + "过敏"
                    else:
                        info["过敏史"] = "有过敏史"

        if not info["用药史"] and ("服用" in text or "吃" in text or "服药" in text):
            drug_match = re.search(r'(?:服用|吃|服药)\s*([\u4e00-\u9fa5]{2,6})', text)
            if drug_match:
                info["用药史"] = drug_match.group(1)

        if not info["id_card"]:
            id_match = re.search(r'(\d{17}[\dXx])', text)
            if id_match:
                info["id_card"] = id_match.group(1)

        if not info["性别"]:
            gender_match = re.search(r'(?:性别|姓别)[:：]\s*([男女])', text)
            if gender_match:
                info["性别"] = gender_match.group(1)
            else:
                # 直接查找男/女
                if re.search(r'\b男\b', text):
                    info["性别"] = "男"
                elif re.search(r'\b女\b', text):
                    info["性别"] = "女"

        if not info["家庭住址"]:
            address_match = re.search(r'(?:家庭住址|住址|地址)[:：]\s*([^\n,，]+)', text)
            if address_match:
                info["家庭住址"] = address_match.group(1).strip()
            else:
                address_match = re.search(r'家住\s*([\u4e00-\u9fa5]+(?:省|市|区))', text)
                if address_match:
                    info["家庭住址"] = address_match.group(1).strip()

        if not info["联系方式"]:
            phone_match = re.search(r'(?:联系方式|电话|手机)[:：]\s*(1[3-9]\d{9})', text)
            if phone_match:
                info["联系方式"] = phone_match.group(1)
            else:
                phone_match = re.search(r'(1[3-9]\d{9})', text)
                if phone_match:
                    info["联系方式"] = phone_match.group(1)

        return info

    def _get_patient_info_from_db(self, name: str) -> dict:
        from tools.tool_registry import get_tools
        tools = get_tools()
        patient_tool = tools.get("patient")
        if patient_tool:
            stored = patient_tool.recall(name)
            if stored:
                info_text = stored.get("info", "")
                self._log("从数据库读取到:", info_text)
                extracted = self._extract_info_from_text(info_text)
                extracted["全文档案"] = info_text
                return extracted
        return {}

    def _analyze_gap(self, state: AgentState) -> AgentState:
        self._log("========== _analyze_gap 开始 ==========")
        history = state.get("history", [])
        patient_info = state.get("patient_info", {})
        current_patient = state.get("current_patient", "")

        all_user_text = ""
        for msg in history:
            if msg.get("role") == "user":
                all_user_text += " " + msg.get("content", "")
            if msg.get("role") == "system" and "【文件内容】" in msg.get("content", ""):
                all_user_text += " " + msg.get("content", "")
        name_match = re.search(r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
        if name_match:
            current_patient = name_match.group(1)

        if current_patient and not patient_info:
            db_info = self._get_patient_info_from_db(current_patient)
            if db_info:
                for key in ["年龄", "过敏史", "用药史", "性别", "家庭住址", "联系方式"]:
                    if db_info.get(key):
                        patient_info[key] = db_info[key]
                if db_info.get("全文档案"):
                    patient_info["全文档案"] = db_info["全文档案"]

        extracted = self._extract_info_from_text(all_user_text)
        self._log("从历史提取的新信息:", extracted)
        for key in ["年龄", "过敏史", "用药史", "性别", "家庭住址", "联系方式"]:
            if extracted.get(key):
                patient_info[key] = extracted[key]

        missing = []
        if "年龄" not in patient_info:
            missing.append("年龄")
        if "过敏史" not in patient_info:
            missing.append("过敏史")
        if "用药史" not in patient_info:
            missing.append("用药史")

        state["current_patient"] = current_patient
        state["patient_info"] = patient_info
        state["missing_info"] = missing
        state["iteration"] = state.get("iteration", 0) + 1
        state["max_iterations"] = state.get("max_iterations", 5)

        self._log("最终 patient_info:", patient_info)
        self._log("最终 missing_info:", missing)
        self._log("========== _analyze_gap 结束 ==========")
        return state

    def _should_ask_or_execute(self, state: AgentState) -> str:
        if state["missing_info"] and state["iteration"] <= state.get("max_iterations", 5):
            return "ask"
        return "execute"

    def _ask_missing(self, state: AgentState) -> AgentState:
        missing = state["missing_info"]
        if not missing:
            state["final_answer"] = "请描述您的具体症状，我来帮您分析。"
            return state

        questions = {
            "年龄": "请问您的年龄是？（例如：30岁）",
            "过敏史": "请问您是否有药物过敏史？（例如：无过敏或对青霉素过敏）",
            "用药史": "请问您目前是否在服用其他药物？（例如：未服用或服用感冒灵）"
        }
        ask_text = "为了给您更精准的建议，请补充以下信息：\n"
        for item in missing:
            ask_text += "- " + questions.get(item, item) + "\n"
        state["final_answer"] = ask_text
        return state

    def _execute_tools(self, state: AgentState) -> AgentState:
        original_question = state["question"]
        patient_info = state.get("patient_info", {})
        current_patient = state.get("current_patient", "")

        parts = []
        if patient_info.get("年龄"):
            parts.append("年龄：" + patient_info["年龄"])
        if patient_info.get("过敏史"):
            parts.append("过敏史：" + patient_info["过敏史"])
        if patient_info.get("用药史"):
            parts.append("用药史：" + patient_info["用药史"])
        if patient_info.get("性别"):
            parts.append("性别：" + patient_info["性别"])
        if patient_info.get("家庭住址"):
            parts.append("家庭住址：" + patient_info["家庭住址"])
        if patient_info.get("联系方式"):
            parts.append("联系方式：" + patient_info["联系方式"])
        if patient_info.get("全文档案"):
            parts.append("既往档案：" + patient_info["全文档案"])
        
        info_str = "，".join(parts) if parts else ""
        enhanced = "患者信息：" + info_str + "。问题：" + original_question if parts else original_question

        plan = self.planner.run(enhanced)
        tool_list = plan.get("tools", ["drug", "guideline", "literature", "risk"])
        tool_list = [t for t in tool_list if t != "patient"]

        if "report" in tool_list:
            import re
            # 优先从原始问题中提取姓名（格式：生成评估表：张三 或 生成评估表 张三）
            name_match = re.search(r'生成评估表[：:]\s*([\u4e00-\u9fa5]{2,4})', original_question)
            if not name_match:
                name_match = re.search(r'生成评估表\s*([\u4e00-\u9fa5]{2,4})', original_question)
            if not name_match:
                name_match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*的(?:评估表|报告|档案|病历)', original_question)
            
            if name_match:
                extracted_name = name_match.group(1)
                kw_match = re.search(r'(生成评估表|生成报告|生成档案|生成病历)', original_question)
                if kw_match:
                    keyword = kw_match.group(1)
                    tool_question = f"{keyword} {extracted_name}"
                else:
                    tool_question = f"{original_question} {extracted_name}"
                print(f"[ConsultGraph] 为 report 工具构建专用问题: {tool_question}")
            else:
                # 如果没有提取到姓名，使用原始问题（让 report_tool 自己处理）
                tool_question = original_question
                print(f"[ConsultGraph] 未提取到姓名，使用原始问题: {tool_question}")
        else:
            tool_question = enhanced

        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.executor.run(tool_list, tool_question))
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self.executor.run(tool_list, tool_question))
        except Exception as e:
            print("[ConsultGraph] 执行工具出错:", e)
            results = []

        if not isinstance(results, list):
            results = []

        state["tool_results"] = results
        return state

    def _synthesize(self, state: AgentState) -> AgentState:
        current_patient = state.get("current_patient", "")
        question = state["question"]
        
        if current_patient and current_patient not in question:
            has_name = re.search(r'[\4e00-\u9fa5]{2,4}',question)
            if not has_name:
                if "生成评估表" in question:
                    question = f"生成评估表 {current_patient}"
                elif "生成报告" in question:
                    question = f"生成报告 {current_patient}"
                elif "生成档案" in question:
                    question = f"生成档案 {current_patient}"
                elif "生成病历" in question:
                    question = f"生成病历 {current_patient}"
                else:
                    question = f"{current_patient} {question}"
                print(f"[ConsultGraph] 已将患者姓名拼接到问题: {question}")
            else:
                print(f"[ConsultGraph] 问题中已包含姓名，跳过拼接")

        if current_patient:
            try:
                import chat
                chat.current_session_user = current_patient
                print(f"[ConsultGraph] 已同步患者姓名到会话: {current_patient}")
            except Exception as e:
                print(f"[ConsultGraph] 同步患者姓名失败: {e}")

        print(f"[ConsultGraph._synthesize] 开始执行")
        results = state.get("tool_results", [])
        if not isinstance(results, list):
            results = []

        answer = self.synthesizer.run(question, results)
        print(f"[ConsultGraph._synthesize] 合成答案完成")

        state["final_answer"] = answer
        return state

    def run(self, question: str, history: list = None) -> str:
        self._log("========== run 开始 ==========")
        patient_info = {}
        file_extracted_name = None
        file_content = ""
        result = None  # 修复 result 变量未定义的问题

        if history:
            for msg in history:
                if msg.get("role") == "system":
                    content_part = msg.get("content", "")
                    if "【文件内容】" in content_part or "文件内容" in content_part:
                        if "【文件内容】" in content_part:
                            file_content = content_part.split("【文件内容】")[1].strip()
                        else:
                            file_content = content_part.strip()
                        self._log("检测到文件内容，准备提取患者信息...")
                        break

        if file_content:
            self._log("检测到文件内容，开始自动提取患者信息...")
            extracted = self._extract_info_from_text(file_content)
            
            if extracted.get("姓名"):
                name = extracted["姓名"]
                file_extracted_name = name
                
                from tools.tool_registry import get_tools
                tools = get_tools()
                patient_tool = tools.get("patient")
                if patient_tool:
                    result = patient_tool.remember(
                        name=name,
                        info="，".join([
                            f"性别：{extracted.get('性别')}" if extracted.get('性别') else "",
                            f"年龄：{extracted.get('年龄')}" if extracted.get('年龄') else "",
                            f"联系方式：{extracted.get('联系方式')}" if extracted.get('联系方式') else "",
                            f"家庭住址：{extracted.get('家庭住址')}" if extracted.get('家庭住址') else "",
                            f"过敏史：{extracted.get('过敏史')}" if extracted.get('过敏史') else "",
                            f"用药史：{extracted.get('用药史')}" if extracted.get('用药史') else "",
                            f"症状：{extracted.get('症状')}" if extracted.get('症状') else "",
                            f"身份证号：{extracted.get('id_card')}" if extracted.get('id_card') else "",
                        ]).strip("，"),
                        id_card=extracted.get('id_card'),
                        gender=extracted.get('性别'),
                        age=extracted.get('年龄'),
                        phone=extracted.get('联系方式'),
                        address=extracted.get('家庭住址'),
                        allergy=extracted.get('过敏史'),
                        medication=extracted.get('用药史'),
                        symptoms=extracted.get('症状')
                    )
                    if result:
                        self._log(f"已保存患者 {name}，缺失字段提示: {result.get('debug', {}).get('missing_fields', [])}")

        self._log("问题:", question)
        self._log("历史:", history)
        all_user_text = ""
        for msg in (history or []):
            if msg.get("role") == "user":
                all_user_text += " " + msg.get("content", "")
        all_user_text += " " + question
        self._log("所有用户文本:", all_user_text)

        # ===== 🆕 提取患者姓名（支持多种格式） =====
        name = None

        # 1. 尝试从“我是XXX”格式提取
        name_match = re.search(r'(?:我是|我叫|我是患者|患者)\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
        if name_match:
            name = name_match.group(1)
            self._log(f"从身份声明提取姓名: {name}")

        # 2. 尝试从“生成病历 XXX”格式提取
        if not name:
            name_match = re.search(r'(?:生成病历|生成评估表|生成报告|生成档案)\s*[:：]?\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
            if name_match:
                name = name_match.group(1)
                self._log(f"从生成命令提取姓名: {name}")

        # 3. 尝试从“XXX的XXX”格式提取
        if not name:
            name_match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*的(?:信息|档案|病历|评估)', all_user_text)
            if name_match:
                name = name_match.group(1)
                self._log(f"从查询命令提取姓名: {name}")

        # 4. 从文件内容提取的姓名作为兜底
        if not name and file_extracted_name:
            name = file_extracted_name
            self._log("使用从文件内容提取的姓名:", name)

        self._log("提取姓名:", name)
        
        if name:
            db_info = self._get_patient_info_from_db(name)
            if db_info:
                for key in ["年龄", "过敏史", "用药史", "性别", "家庭住址", "联系方式"]:
                    if db_info.get(key):
                        patient_info[key] = db_info[key]
                if db_info.get("全文档案"):
                    patient_info["全文档案"] = db_info["全文档案"]
                self._log("从数据库加载档案:", patient_info)

        extracted = self._extract_info_from_text(all_user_text)
        for key in ["年龄", "过敏史", "用药史", "性别", "家庭住址", "联系方式"]:
            if extracted.get(key):
                patient_info[key] = extracted[key]

        missing = []
        if "年龄" not in patient_info:
            missing.append("年龄")
        if "过敏史" not in patient_info:
            missing.append("过敏史")
        if "用药史" not in patient_info:
            missing.append("用药史")

        initial_state = {
            "question": question,
            "history": history or [],
            "current_patient": name or "",
            "patient_info": patient_info,
            "missing_info": missing,
            "tool_results": [],
            "final_answer": "",
            "iteration": 0,
            "max_iterations": 5
        }
        self._log("初始状态:", initial_state)
        self._log("========== run 结束 ==========")

        try:
            result = self.graph.invoke(initial_state)
            return result.get("final_answer", "未能生成回答")
        except Exception as e:
            print("ConsultGraph 运行异常:")
            traceback.print_exc()
            return "问诊过程出错：" + str(e)