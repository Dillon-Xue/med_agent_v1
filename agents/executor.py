import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)

class Executor:
    def __init__(self, tools: dict):
        self.tools = tools
        self._executor = ThreadPoolExecutor(max_workers=10)

    async def run(self, tool_list, question, trace_callback=None):
        if not tool_list:
            return []

        loop = asyncio.get_event_loop()

        async def run_one(name):
            start_time = time.time()
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
                # 🆕 trace_callback 现在改成了同步函数，直接调用
                if trace_callback:
                    logger.debug(f"[Executor] 正在记录 trace: {name}")
                    trace_callback("executor", {
                        "tool": name,
                        "question": question,
                        "duration": time.time() - start_time,
                        "success": True,
                        "result_preview": res.get("answer", "")[:200] if res else ""
                    })
                return res if res is not None else None
            except asyncio.TimeoutError:
                if trace_callback:
                    trace_callback("executor", {
                        "tool": name,
                        "question": question,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": "Timeout (60s)"
                    })
                return {
                    "answer": f"工具 {name} 执行超时（60秒）",
                    "source": name,
                    "success": False
                }
            except Exception as e:
                if trace_callback:
                    trace_callback("executor", {
                        "tool": name,
                        "question": question,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e)
                    })
                return {
                    "answer": f"工具 {name} 执行失败: {str(e)}",
                    "source": name,
                    "success": False
                }

        tasks = [run_one(name) for name in tool_list]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]