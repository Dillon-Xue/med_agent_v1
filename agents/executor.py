from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class Executor:
    def __init__(self, tools: dict):
        self.tools = tools

    def run(self, tool_list, question):
        # 如果工具列表为空，直接返回空结果
        if not tool_list:
            return []

        results = [None] * len(tool_list)

        with ThreadPoolExecutor(max_workers=len(tool_list)) as executor:
            future_to_idx = {}
            for idx, name in enumerate(tool_list):
                tool = self.tools.get(name)
                if tool:
                    future = executor.submit(tool.run, question)
                    future_to_idx[future] = idx
                else:
                    results[idx] = {
                        "answer": f"工具 {name} 未找到",
                        "source": name,
                        "success": False
                    }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result(timeout=10)
                    if res is not None:
                        results[idx] = res
                except Exception as e:
                    results[idx] = {
                        "answer": f"工具执行失败: {str(e)}",
                        "source": tool_list[idx] if idx < len(tool_list) else "unknown",
                        "success": False
                    }

        return [r for r in results if r is not None]