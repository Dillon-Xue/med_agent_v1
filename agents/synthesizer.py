from openai import OpenAI
import os

class Synthesizer:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def run(self, augmented_question, tool_results):
        # 🚀 优先检查：如果 patient 工具返回了确认信息（✅），直接返回
        for res in tool_results:
            if not res or not isinstance(res, dict):
                continue
            if res.get("source") == "patient":
                answer = res.get("answer", "")
                # 如果 patient 返回的是保存/更新成功的确认信息
                if answer.startswith("✅"):
                    return answer
                # 如果 patient 返回的是档案查询结果
                if answer.startswith("📋"):
                    return answer

        # 特殊处理：只有一个工具结果，且来源是 patient，直接返回
        if len(tool_results) == 1:
            res = tool_results[0]
            if res and isinstance(res, dict) and res.get("source") == "patient":
                return res.get("answer", "患者工具未返回有效信息")

        # 否则，正常综合所有工具结果
        context_parts = []
        for res in tool_results:
            if not res or not isinstance(res, dict):
                continue
            answer = res.get("answer", "")
            if not answer:
                continue
            source = res.get("source", "unknown").upper()
            context_parts.append(f"【来自 {source} 工具】\n{answer}")

        if not context_parts:
            return "未找到相关信息，请尝试换一种问法。"

        combined = "\n\n".join(context_parts)

        system_prompt = """你是一个专业的医学知识合成器。回答必须严格遵循以下纯文本格式，不要使用任何 markdown 语法。

输出格式要求：
1. 第一行：核心结论（一句话）。
2. 空一行。
3. 使用 "- " 开头的列表项列出关键信息，每项占一行。
4. 重要警示前加 "⚠️ " 并单独成行。
5. 不同主题之间用空行分隔。
6. 最后一行：总结建议。

注意：不要使用任何加粗符号，不要使用星号。每一条信息末尾标注【来源：xxx】。"""

        user_prompt = f"""对话历史及当前问题：
{augmented_question}

各工具返回结果：
{combined}

请严格按照上述纯文本格式要求给出综合回答，不要使用任何 markdown 语法。"""

        resp = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content