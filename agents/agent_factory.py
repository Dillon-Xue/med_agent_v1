import os
from agents.planner import Planner
from agents.executor import Executor
from agents.synthesizer import Synthesizer
from tools.tool_registry import get_tools
import logging
logger = logging.getLogger(__name__)

class SpecialtyAgent:
    def __init__(self, specialty: str, tools: dict, api_key: str):
        self.specialty = specialty
        self.planner = Planner(specialty=specialty)
        self.executor = Executor(tools)
        self.synthesizer = Synthesizer(api_key=api_key, specialty=specialty)

    async def run(self, question: str, history: list = None, trace_callback=None):
        def _wrap_callback(original_callback, specialty):
            if original_callback is None:
                return None
            def wrapped(step_type, data):
                modified_data = data.copy() if isinstance(data, dict) else {'data': data}
                modified_data['agent'] = specialty
                original_callback(step_type, modified_data)
            return wrapped
        
        wrapped_trace = _wrap_callback(trace_callback, self.specialty)
        plan = self.planner.run(question, trace_callback=wrapped_trace)
        tool_list = plan.get("tools", [])
        if not tool_list:
            tool_list = ["drug", "guideline", "literature", "risk"]
        results = await self.executor.run(tool_list, question, trace_callback=wrapped_trace)
        answer = self.synthesizer.run(question, results, trace_callback=wrapped_trace)
        return {
            "specialty": self.specialty,
            "answer": answer,
            "tools_used": tool_list,
            "tool_results": results
        }
        
def create_agent(specialty: str):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    tools = get_tools()
    return SpecialtyAgent(specialty, tools, api_key)

_agent_cache = {}
def get_agent(specialty: str):
    if specialty not in _agent_cache:
        _agent_cache[specialty] = create_agent(specialty)
    return _agent_cache[specialty]