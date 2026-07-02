from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """LangGraph 状态定义"""
    question: str
    # 对话消息（自动追加）
    messages: Annotated[List[dict], add_messages]
    # 患者信息（手动管理，不需要 add_messages）
    current_patient: str
    patient_info: dict
    missing_info: List[str]
    tool_results: List[dict]
    final_answer: str
    iteration: int
    max_iterations: int
    # ===== 🆕 反思循环新增字段 =====
    critique_feedback: str          # 反思节点的修改意见
    need_rerank: bool               # 是否触发 LLM Rerank
    reflection_history: List[dict]  # 反思历史记录（用于调试）