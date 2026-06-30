import os
import time
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.embeddings import DashscopeEmbeddings
from utils.response import build_response

PROJECT_ROOT = os.getenv("MED_AGENT_ROOT", os.getcwd())

class MemoryTool:
    def __init__(self):
        self.db_path = os.path.join(PROJECT_ROOT, "vector_db", "memory")
        self.embeddings = DashscopeEmbeddings()
        self.vectordb = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name="memory"
        )
        self._trace = []
        print(f"[MemoryTool] Loaded from {self.db_path}")

    def trace(self, stage: str, data: dict):
        self._trace.append({"tool": "memory", "stage": stage, "ts": time.time(), "data": data})

    def get_trace(self):
        return self._trace

    def clear_trace(self):
        self._trace = []

    def remember(self, patient_name: str, diagnosis: str, medications: str,
                 assessment: str, medication_goal: str = "", precautions: str = "",
                 doctor_id: str = "default", approval_id: str = "", requester: str = "") -> dict:
        self.clear_trace()
        content = f"""
患者姓名：{patient_name}
诊断：{diagnosis}
用药方案：{medications}
评估结果：{assessment}
用药目标：{medication_goal}
注意事项：{precautions}
审批人：{requester}
审批ID：{approval_id}
医生ID：{doctor_id}
"""
        metadata = {
            "patient_name": patient_name,
            "diagnosis": diagnosis,
            "medications": medications,
            "doctor_id": doctor_id,
            "approval_id": approval_id,
            "requester": requester,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        doc_id = f"{patient_name}_{diagnosis}_{approval_id}"
        doc = Document(page_content=content, metadata=metadata)
        self.vectordb.add_documents(documents=[doc], ids=[doc_id])
        return build_response(
            answer=f"✅ 已记忆病例：{patient_name} - {diagnosis}",
            source="memory",
            debug={"doc_id": doc_id},
            trace=self.get_trace()
        )

    def recall(self, query: str, k: int = 3, doctor_id: str = None, min_similarity: float = 0.0) -> List[dict]:
        self.clear_trace()
        try:
            docs = self.vectordb.similarity_search_with_score(query, k=k * 2)
            if doctor_id:
                docs = [(doc, score) for doc, score in docs if doc.metadata.get("doctor_id") == doctor_id]
            results = []
            for doc, score in docs:
                similarity = 1 - score
                if similarity < min_similarity:
                    continue
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "similarity": similarity
                })
            return results[:k]
        except Exception as e:
            return []

    def run(self, query: str) -> dict:
        results = self.recall(query)
        if not results:
            return build_response(
                answer="未找到相关历史病例",
                source="memory",
                debug={"count": 0},
                trace=self.get_trace()
            )
        lines = ["📋 找到相关历史病例："]
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            lines.append(f"\n{i}. 【{meta.get('patient_name', '未知')}】{meta.get('diagnosis', '未知诊断')}")
            lines.append(f"   用药：{meta.get('medications', '无')}")
            lines.append(f"   时间：{meta.get('created_at', '未知')}")
            if r.get("score"):
                lines.append(f"   相似度：{(1 - r['score']):.2%}")
        return build_response(
            answer="\n".join(lines),
            source="memory",
            debug={"count": len(results)},
            trace=self.get_trace()
        )