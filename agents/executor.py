import asyncio
from concurrent.futures import ThreadPoolExecutor

class Executor:
    def __init__(self, tools: dict):
        self.tools = tools
        self._executor = ThreadPoolExecutor(max_workers=10)

    async def run(self, tool_list, question):
        if not tool_list:
            return []

        loop = asyncio.get_event_loop()

        async def run_one(name):
            tool = self.tools.get(name)
            if not tool:
                return {
                    "answer": f"工具 {name} 未找到",
                    "source": name,
                    "success": False
                }
            try:
                res = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, tool.run, question),
                    timeout=60.0
                )
                return res if res is not None else None
            except asyncio.TimeoutError:
                return {
                    "answer": f"工具 {name} 执行超时（10秒）",
                    "source": name,
                    "success": False
                }
            except Exception as e:
                return {
                    "answer": f"工具 {name} 执行失败: {str(e)}",
                    "source": name,
                    "success": False
                }

        tasks = [run_one(name) for name in tool_list]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]