from tools.base_tool import BaseTool
from utils.response import build_response
from utils.embeddings import DashscopeEmbeddings
import time, logging
logger = logging.getLogger(__name__)

class RiskTool(BaseTool):
    def __init__(self, base_dir, api_key):
        super().__init__(base_dir, "risk", api_key)
        self.embeddings = DashscopeEmbeddings()
        self.load_db(self.embeddings)
        logger.info(f"VECTOR READ PATH: {self.db_path}")
        self._trace = []

    def trace(self, stage: str, data: dict):
        self._trace.append({
            "tool": "risk",
            "stage": stage,
            "ts": time.time(),
            "data": data
        })

    def get_trace(self):
        return self._trace

    def clear_trace(self):
        self._trace = []

    def run(self, query: str):
        # 剥离对话历史
        if "当前问题：" in query:
            query = query.rsplit("当前问题：", 1)[-1].strip()
        self.clear_trace()
        t0 = time.time()
        self.trace("run_start", {"query": query})

        try:
            docs = self.retrieve_with_optimization(query, k=5, use_llm_rerank=False)
        except Exception as e:
            self.trace("error", {"stage": "retrieve", "error": str(e)})
            return build_response(
                answer=f"检索失败: {str(e)}",
                source="risk",
                debug={"error": str(e)},
                trace=self.get_trace()
            )

        self.trace("retrieve", {"count": len(docs), "samples": [d.page_content[:200] for d in docs]})
        context, citations = self._build_cited_context(docs)
        self.trace("context", {"length": len(context), "preview": context[:300], "citations": citations})

        prompt = f"""
你是药物风险分析专家，专注于药物相互作用、不良反应、禁忌症。

只能基于给定资料回答，不允许编造。
只回答与问题中提到的药品直接相关的内容。如果资料中出现了其他药品的信息，忽略它们，不纳入回答。
只回答与问题直接相关的内容（如问题问"配伍禁忌"，只回答配伍禁忌信息；问"相互作用"，只回答相互作用信息）。

资料：
{context}

问题：
{query}

重要要求：
1. 只回答与问题直接相关的内容，不输出资料中的其他无关信息
2. 如果资料中出现了其他药品，忽略该药品的所有内容
3. 每条回答内容末尾必须标注精确来源，格式为：(来源: xxx.pdf, 第 N 页，第 M 段)。如果资料中涉及图表信息，格式为：(来源: xxx.pdf, 第 N 页, 图X, 第 M 段) 或 (来源: xxx.pdf, 第 N 页, 表X, 第 M 段)。
4. 来源信息必须直接从资料中复制，不得自行编造
5. 如果某条信息无法对应到资料中的来源，标注：(来源: 模型推理，请核实)
"""
        messages = [
            {"role": "system", "content": "严格药物风险分析，仅依据资料"},
            {"role": "user", "content": prompt}
        ]
        #answer = self._safe_llm_call(messages)
        answer = self._safe_llm_call(messages, fallback_context=context)
        latency = round(time.time() - t0, 3)
        self.trace("final_answer", {"answer": answer, "latency": latency})

        return build_response(
            answer=answer,
            source="risk",
            debug={"retrieved": len(docs), "latency": latency, "citations": citations},
            trace=self.get_trace()
        ) 
