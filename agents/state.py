from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph 状态定义"""
    # 用户原始问题
    question: str
    # 对话消息（自动追加）
    messages: Annotated[List[dict], add_messages]
    # 已收集的患者信息
    patient_info: dict
    # 缺失的信息（需要追问的）
    missing_info: List[str]
    # 工具调用结果
    tool_results: List[dict]
    # 最终答案
    final_answer: str
    # 迭代次数
    iteration: int
    # 最大迭代次数
    max_iterations: int
    history: list