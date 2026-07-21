import os
import os, re, json, asyncio, concurrent.futures, traceback, logging
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from tools.tool_registry import get_tools
from openai import OpenAI
from utils.response import mask_sensitive, mask_dict_sensitive
from utils.config import get_llm_client
import chat as chat_module  # 放在文件顶部导入区域
from tools.memory_tool import MemoryTool

logger = logging.getLogger(__name__)

# 全局调试开关
DEBUG = True

class ConsultGraph:
    def __init__(self):
        self.tools = get_tools()
        self.planner = Planner()
        self.executor = Executor(self.tools)
        # Synthesizer 使用 get_llm_client
        self.synthesizer = Synthesizer(api_key=os.getenv("DASHSCOPE_API_KEY"))
        # 使用统一客户端工厂
        self.client, self.model = get_llm_client(os.getenv("DASHSCOPE_API_KEY"))
        logger.debug(f"[ConsultGraph] Using model: {self.model}")
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("analyze_gap", self._analyze_gap)
        graph.add_node("ask_missing", self._ask_missing)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("reflect", self._reflect_on_answer)
        graph.add_node("recover_with_rerank", self._recover_with_rerank)
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
        graph.add_edge("synthesize", "reflect") 

        graph.add_conditional_edges(
            "reflect",
            self._should_continue_or_output,
            {
                "output": END,
                "retry_synthesize": "synthesize",
                "trigger_rerank": "recover_with_rerank",
            }
        )
        graph.add_edge("recover_with_rerank", "synthesize")

        return graph.compile()
    
    def _log(self, *args, **kwargs):
        if DEBUG:
            # 对打印内容脱敏
            masked_args = []
            for arg in args:
                if isinstance(arg, dict):
                    masked_args.append(mask_dict_sensitive(arg))
                elif isinstance(arg, str):
                    masked_args.append(mask_sensitive(arg))
                else:
                    masked_args.append(arg)
            logger.debug("[DEBUG]", *masked_args, **kwargs)

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
                model=self.model,
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
            
            logger.info(f"[ConsultGraph] LLM 提取结果:{mask_sensitive(result)}")
            
        except Exception as e:
            logger.error(f"[ConsultGraph] LLM 提取失败，降级到正则: {e}")
        
        # 正则补充提取（合并结果）
        regex_result = self._extract_info_with_regex(text)
        logger.debug(f"[ConsultGraph] 正则补充提取结果: {mask_sensitive(regex_result)}")
        
        # 合并：如果 LLM 提取的字段为空，用正则的结果补充
        for key in ["姓名", "年龄", "过敏史", "用药史", "id_card", "性别", "家庭住址", "联系方式"]:
            if not result.get(key) and regex_result.get(key):
                result[key] = regex_result[key]
        
        logger.info(f"[ConsultGraph] 最终提取结果: {mask_sensitive(result)}")
        return result

    def _extract_info_with_regex(self, text: str) -> dict:
        """降级方案：使用正则提取（支持表格格式）"""
        logger.debug(f"[DEBUG] _extract_info_with_regex 被调用，text 长度: {len(text)}")
        logger.debug(f"[DEBUG] text 内容:\n{text[:500]}")
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
            name_match = re.search(r'(?:我是|我叫|我是患者|患者|记住患者)\s*([\u4e00-\u9fa5]{2,4})', text)
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
        name_match = re.search(r'(?:我是|我叫|我是患者|患者|记住患者)\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
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
        #state["iteration"] = state.get("iteration", 0) + 1
        state["max_iterations"] = state.get("max_iterations", int(os.getenv("MAX_ITERATIONS", "5")))

        self._log("最终 patient_info:", patient_info)
        self._log("最终 missing_info:", missing)
        self._log("========== _analyze_gap 结束 ==========")
        return state

    def _should_ask_or_execute(self, state: AgentState) -> str:
        if state["missing_info"] and state["iteration"] <= state.get("max_iterations", int(os.getenv("MAX_ITERATIONS", "5"))):
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

    def _extract_current_question(self, enhanced: str) -> str:
        """从 enhanced 中提取当前问题，剥离对话历史前缀，避免污染工具的 RAG 检索"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DEBUG] _extract_current_question 输入: {enhanced[:80]}...")
        if "当前问题：" in enhanced:
            # 取最后一个"当前问题："之后的内容（防止历史中也有此标记）
            current_q = enhanced.rsplit("当前问题：", 1)[-1].strip()
            logger.info(f"[DEBUG] _extract_current_question 输出: {current_q}")
            return current_q
        # 没有"当前问题："标记，剥离可能的"患者信息：...。问题："前缀
        if enhanced.startswith("患者信息：") and "。问题：" in enhanced:
            return enhanced.split("。问题：", 1)[-1].strip()
        return enhanced

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

        #  L4 语义记忆检索（注入历史参考）
        memory_context = ""
        try:
            from utils.thread_context import doctor_id_var
            doctor_id = doctor_id_var.get() or 'default' 
            memory_tool = MemoryTool()
            results = memory_tool.recall(original_question, k=2, doctor_id=doctor_id, min_similarity=0.3)
            if results:
                memory_lines = ["【历史参考病例】"]
                for r in results:
                    meta = r["metadata"]
                    memory_lines.append(
                        f"- {meta.get('patient_name', '未知')}：{meta.get('diagnosis', '')}，"
                        f"用药：{meta.get('medications', '无')}"
                    )
                memory_context = "\n".join(memory_lines) + "\n"
                self._log(f"检索到 {len(results)} 条历史病例")
        except Exception as e:
            print(f"[Memory] 检索失败: {e}")

        if memory_context:
            enhanced = memory_context + enhanced

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
                logger.debug(f"[ConsultGraph] 为 report 工具构建专用问题: {tool_question}")
            else:
                # 如果没有提取到姓名，使用原始问题（让 report_tool 自己处理）
                tool_question = original_question
                logger.debug(f"[ConsultGraph] 未提取到姓名，使用原始问题: {tool_question}")
        else:
            # 工具只接收当前问题，不接收对话历史（避免污染 RAG 检索和工具内部 LLM）
            if "当前问题：" in enhanced:
                tool_question = enhanced.rsplit("当前问题：", 1)[-1].strip()
            elif enhanced.startswith("患者信息：") and "。问题：" in enhanced:
                tool_question = enhanced.split("。问题：", 1)[-1].strip()
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
            logger.error("[ConsultGraph] 执行工具出错:", e)
            results = []

        if not isinstance(results, list):
            results = []

        state["tool_results"] = results
        return state

    def _synthesize(self, state: AgentState) -> AgentState:
        """
        合成回答（支持接收反思反馈意见）
        带详细日志，显示答案前后对比
        """
        current_patient = state.get("current_patient", "")
        question = state["question"]
        results = state.get("tool_results", [])
        if not isinstance(results, list):
            results = []

        # ===== 获取反思反馈（如果有） =====
        feedback = state.get("critique_feedback", "")
        old_answer = state.get("final_answer", "")  # 用于对比

        # ==================== 合成前日志 ====================
        if feedback:
            logger.info(f"\n{'─'*40}")
            logger.info(f"[Synthesize] 携带反馈意见重新生成（第 {state.get('iteration', 0)} 轮）")
            logger.info(f"  反馈内容: {feedback[:100]}...")
            if old_answer:
                logger.info(f" 旧答案预览: {old_answer[:150]}...")
        else:
            logger.info(f"[Synthesize] 正常生成初始答案")
        # ========================================================

        # =====  构建增强问题（注入反馈） =====
        if feedback:
            augmented_question = (
                f"{question}\n\n"
                f"【上一轮反思反馈】{feedback}\n"
                f"请根据反馈意见修正上述回答，只输出修正后的完整答案，不要输出其他内容。"
            )
        else:
            augmented_question = question

        # ===== 调用 Synthesizer 生成答案 =====
        answer = self.synthesizer.run(augmented_question, results)

        # ===== 保存到状态 =====
        state["final_answer"] = answer

        # ==================== 合成后对比日志 ====================
        if feedback and old_answer:
            if old_answer != answer:
                logger.info(f"修正完成")
                logger.info(f"新答案预览: {answer[:150]}...")
                if len(old_answer) > 50 and len(answer) > 50:
                    logger.info(f"答案长度变化: {len(old_answer)} → {len(answer)} 字符")
            else:
                logger.warning(f"携带反馈重新生成，但答案未发生变化")
        # ========================================================

        logger.debug(f"[ConsultGraph._synthesize] 合成答案完成")
        return state

    def run(self, question: str, history: list = None) -> str:
        self._log("========== run 开始 ==========")
        patient_info = {}
        file_extracted_name = None
        file_content = ""
        result = None  # 修复 result 变量未定义的问题

         # 检测是否是查询患者信息的意图
        if re.search(r'[\u4e00-\u9fa5]{2,4}\s*的(?:信息|档案|病历|评估|报告)', question):
            # 提取姓名
            name_match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*的(?:信息|档案|病历|评估|报告)', question)
            if name_match:
                name = name_match.group(1)
                # 从数据库查询患者
                from tools.tool_registry import get_tools
                tools = get_tools()
                patient_tool = tools.get("patient")
                if patient_tool:
                    info = patient_tool.recall(name)
                    if info:
                        lines = [f"患者 {name} 的档案："]
                        if info.get("gender"): lines.append(f"性别：{info['gender']}")
                        if info.get("age"): lines.append(f"年龄：{info['age']}")
                        if info.get("phone"): lines.append(f"联系方式：{info['phone']}")
                        if info.get("address"): lines.append(f"家庭住址：{info['address']}")
                        if info.get("allergy"): lines.append(f"过敏史：{info['allergy']}")
                        if info.get("medication"): lines.append(f"用药史：{info['medication']}")
                        if info.get("symptoms"): lines.append(f"症状：{info['symptoms']}")
                        if info.get("diagnosis"): lines.append(f"诊断：{info['diagnosis']}")
                        if info.get("id_card"): lines.append(f"身份证号：{info['id_card']}")
                        return "\n".join(lines)
                    else:
                        return f"未找到患者 {name} 的档案"


        if history:
            for msg in history:
                if not isinstance(msg, dict):
                    continue
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

        # ===== 提取患者姓名（支持多种格式） =====
        # ===== 提取患者姓名（按优先级，过滤无效词） =====
        name = None
        invalid_names = ["信息", "患者", "查询", "档案", "病历", "评估", "报告", "生成"]

        # 1. 优先匹配 "XXX的信息" 格式
        name_match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*的(?:信息|档案|病历|评估|报告)', all_user_text)
        if name_match:
            candidate = name_match.group(1)
            if candidate not in invalid_names:
                name = candidate
                self._log(f"从查询命令提取姓名: {name}")

        # 2. 匹配 "生成病历 XXX" / "生成评估表 XXX" 等格式
        if not name:
            name_match = re.search(r'(?:生成病历|生成评估表|生成报告|生成档案)\s*[:：]?\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
            if name_match:
                candidate = name_match.group(1)
                if candidate not in invalid_names:
                    name = candidate
                    self._log(f"从生成命令提取姓名: {name}")
        # 2b. 匹配 "给XXX生成/查看/评估..." 格式
        if not name:
            name_match = re.search(r'(?:给|帮|为|替)\s*([\u4e00-\u9fa5]{2,4})\s*(?:生成|查看|评估|开|做)', all_user_text)
            if name_match:
                candidate = name_match.group(1)
                if candidate not in invalid_names:
                    name = candidate
                    self._log(f"从给命令提取姓名: {name}")

        # 3. 匹配 "我是XXX" / "我叫XXX" / "患者XXX" 格式
        if not name:
            name_match = re.search(r'(?:我是|我叫|我是患者|患者|记住患者)\s*([\u4e00-\u9fa5]{2,4})', all_user_text)
            if name_match:
                candidate = name_match.group(1)
                if candidate not in invalid_names:
                    name = candidate
                    self._log(f"从身份声明提取姓名: {name}")

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
            "max_iterations": int(os.getenv("MAX_ITERATIONS", "3"))
        }
        self._log("初始状态:", initial_state)
        self._log("========== run 结束 ==========")

        try:
            result = self.graph.invoke(initial_state)
            return result.get("final_answer", "未能生成回答")
        except Exception as e:
            logger.error("ConsultGraph 运行异常:")
            traceback.print_exc()
            return "问诊过程出错：" + str(e)

    def _reflect_on_answer(self, state: AgentState) -> AgentState:
        """
        反思节点：检查答案质量，一次性列出所有问题
        """
        answer = state.get("final_answer", "")
        question = state.get("question", "")
        tool_results = state.get("tool_results", [])
        iteration = state.get("iteration", 0) + 1
        max_iterations = state.get("max_iterations", int(os.getenv("MAX_ITERATIONS", "3")))

        # 打印进入时的迭代值，帮助排查
        logger.info(f"[Reflect] 进入时 iteration = {state.get('iteration', '未设置')}，本轮自增后为 {iteration}")

        # 日志：循环开始
        logger.info(f"\n{'='*60}")
        logger.info(f"[Reflect] 开始第 {iteration} 轮反思循环 (最大 {max_iterations} 轮)")
        logger.info(f"{'='*60}")
        logger.info(f"当前问题: {question[:100]}...")
        logger.info(f"当前答案预览: {answer[:150]}...")

        # 统计检索到的文档数量
        doc_count = 0
        for res in tool_results:
            if res and isinstance(res, dict):
                debug = res.get("debug", {})
                doc_count += debug.get("retrieved", 0) or debug.get("count", 0)
        logger.info(f"检索到的文档总数: {doc_count}")

        # 默认值
        pass_check = True
        reason = "自查服务异常，默认放行"
        is_data_insufficient = False
        critique = ""

        try:
            critique_prompt = f"""
    你是一位严格的医学质控专家。请审核以下 AI 给出的医疗建议，**一次性找出所有存在的问题**。

    【患者问题】
    {question}

    【AI 生成的回答】
    {answer}

    【检索到的文档数量】
    {doc_count} 条

    请从以下维度逐一审核：
    1. 资料充分性：检索到的资料是否足以支撑这个回答？
    2. 绝对禁忌：是否推荐了患者明确禁用的药物或疗法？
    3. 准确性：剂量、用法、诊断逻辑是否准确？有无明显事实错误？
    4. 完整性：是否遗漏了重要的警示信息、禁忌症或个体化评估？
    5. 幻觉风险：是否编造了不存在的来源、指南或医学事实？

    请输出 JSON 格式，**只输出 JSON，不要其他内容**：
    {{
        "pass": true/false,
        "reason": "通过/不通过的原因，一句话概括",
        "issues": [
            {{
                "severity": "critical|major|minor",
                "description": "问题描述，具体指出哪里错了",
                "fix_instruction": "给出具体的修改指令，告诉 AI 应该怎么改"
            }}
        ],
        "is_data_insufficient": true/false
    }}
    """
            # 关键修改：超时改为 5 秒，不传递 max_retries
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0,
                timeout=60.0   # 从 8 秒缩短到 5 秒
            )
            content = resp.choices[0].message.content.strip()

            # 提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            feedback = json.loads(content)
            pass_check = feedback.get("pass", True)
            reason = feedback.get("reason", "无明确原因")
            is_data_insufficient = feedback.get("is_data_insufficient", False)

            # 打印评估详情
            logger.info("=" * 60)
            logger.info("【反思评估详情】")
            logger.info(f"  自查结果: {'通过' if pass_check else '不通过'}")
            logger.info(f"  原因: {reason}")
            issues = feedback.get("issues", [])
            if issues:
                severity_order = {"critical": 0, "major": 1, "minor": 2}
                issues.sort(key=lambda x: severity_order.get(x.get("severity", "minor"), 3))
                logger.info(f"发现 {len(issues)} 个问题:")
                for i, issue in enumerate(issues, 1):
                    severity = issue.get("severity", "unknown")
                    desc = issue.get("description", "")
                    logger.info(f"  {i}. [{severity}] {desc}")
                critique_parts = []
                for i, issue in enumerate(issues, 1):
                    desc = issue.get("description", "")
                    fix = issue.get("fix_instruction", "")
                    critique_parts.append(f"{i}. 问题：{desc} → 修改：{fix}")
                critique = "\n".join(critique_parts)
            else:
                critique = ""

            logger.info(f"自查结果: {'通过' if pass_check else '不通过'}")
            logger.info(f"原因: {reason}")
            if critique:
                logger.info(f"综合修改意见:\n{critique}")

        except Exception as e:
            import traceback
            logger.error(f"[Reflect] LLM 自查失败: {type(e).__name__}: {e}")
            logger.error(f"[Reflect] 详细堆栈:\n{traceback.format_exc()}")
            logger.warning("[Reflect] 由于自查服务异常，本次反思无法进行评估，默认放行，不进入修正循环。")
            # pass_check 保持默认值 True
            logger.info(f"[Reflect] 当前答案（无评估）:\n{answer}")

        # 判断是否需要 Rerank
        need_rerank = False
        if not pass_check and is_data_insufficient and doc_count < 5:
            need_rerank = True
            logger.warning(f"[Reflect] 触发 Rerank：资料严重不足（文档数={doc_count}）")

        # 保存反思历史
        reflection_history = state.get("reflection_history", [])
        reflection_history.append({
            "iteration": iteration,
            "pass": pass_check,
            "reason": reason,
            "is_data_insufficient": is_data_insufficient,
            "need_rerank": need_rerank,
            "feedback": critique
        })
        logger.info(f"[Reflect] 第 {iteration} 轮结束")
        logger.info(f"自查结果: {'通过' if pass_check else '不通过'}")
        logger.info(f"原因: {reason}")
        if not pass_check and critique:
            logger.info(f"修改意见预览: {critique[:150]}...")

        state["iteration"] = iteration
        state["max_iterations"] = max_iterations
        state["critique_feedback"] = critique
        state["need_rerank"] = need_rerank
        state["reflection_history"] = reflection_history

        if pass_check:
            state["critique_feedback"] = ""

        logger.info(f"[Reflect] 第 {iteration} 轮结束，状态: {'准备输出' if pass_check else '等待路由判断'}")
        return state

    def _should_continue_or_output(self, state: AgentState) -> str:
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", int(os.getenv("MAX_ITERATIONS", "3")))
        
        # 安全获取最新一条反思记录
        history = state.get("reflection_history", [])
        last_reflect = history[-1] if history else {}
        pass_check = last_reflect.get("pass", False)
        need_rerank = state.get("need_rerank", False)
        reason = last_reflect.get("reason", "未知原因")
        feedback = last_reflect.get("feedback", "")
        is_data_insufficient = last_reflect.get("is_data_insufficient", False)

        # ==================== 路由决策日志 ====================
        logger.info(f"\n{'─'*40}")
        logger.info(f"[Router] 第 {iteration} 轮路由决策")
        logger.info(f"自查结果: {'通过' if pass_check else '不通过'}")
        logger.info(f"原因: {reason}")
        logger.info(f"资料不足: {is_data_insufficient}")
        logger.info(f"是否需要 Rerank: {need_rerank}")
        # ========================================================
        if not pass_check and feedback:
            logger.info(f"修改意见预览：{feedback[:100]}...")

        if pass_check:
            logger.info(f"[Router] 决策: 输出结果")
            return "output"

        if iteration >= max_iterations:
            logger.warning(f"[Router] 已达最大迭代次数 {max_iterations}，强制输出（附带人工复核警告）")
            final_answer = state.get("final_answer", "")
            state["final_answer"] = (
                f"经过 {max_iterations} 轮自查仍无法完全确认以下建议的可靠性，请医生复核。\n\n"
                f"{final_answer}"
            )
            return "output"

        if need_rerank:
            logger.info(f"[Router] 决策: 触发 Rerank 补救（资料不足）")
            return "trigger_rerank"

        logger.info(f"[Router] 决策: 返回 Synthesizer 重新生成（携带反馈意见）")
        return "retry_synthesize"

    def _recover_with_rerank(self, state: AgentState) -> AgentState:
        """
        触发 LLM Rerank，并打印详细的补救日志
        """
        question = state.get("question", "")
        old_results_count = len(state.get("tool_results", []))

        # ==================== Rerank 触发日志 ====================
        logger.info(f"\n{'='*60}")
        logger.info(f"[Rerank] 触发 LLM Rerank 补救流程")
        logger.info(f"{'='*60}")
        logger.info(f"  问题: {question[:80]}...")
        logger.info(f"  原有工具结果数: {old_results_count}")
        # ==========================================================

        # 获取当前工具列表（从旧结果中提取来源）
        tool_results = state.get("tool_results", [])
        used_tools = []
        for res in tool_results:
            if res and isinstance(res, dict):
                source = res.get("source")
                if source and source not in used_tools:
                    used_tools.append(source)

        if not used_tools:
            used_tools = ["drug", "guideline", "literature", "risk"]

        logger.info(f"  将重跑工具: {used_tools}")

        # 执行工具获取新结果
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.executor.run(used_tools, question))
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self.executor.run(used_tools, question))
        except Exception as e:
            logger.error("[ConsultGraph] Rerank 执行工具出错:", e)
            results = []

        if not isinstance(results, list):
            results = []

        logger.info(f"Rerank 完成，新工具结果数: {len(results)}")
        if len(results) > old_results_count:
            logger.info(f"结果增加: {old_results_count} → {len(results)}")
        
        # 更新状态
        state["tool_results"] = results
        state["need_rerank"] = False

        # 标记历史
        reflection_history = state.get("reflection_history", [])
        if reflection_history:
            reflection_history[-1]["rerank_triggered"] = True
            reflection_history[-1]["rerank_tools"] = used_tools
        state["reflection_history"] = reflection_history

        return state