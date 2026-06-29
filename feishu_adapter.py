"""
飞书适配层 - 使用 Channel SDK 长连接模式
将飞书消息转换为 Agent 请求，调用现有 process_question
"""

import os
import asyncio
import logging
import json
import re
import inspect
from dotenv import load_dotenv
from lark_channel import FeishuChannel
from chat import process_question

load_dotenv()

logger = logging.getLogger(__name__)

# 飞书配置
APP_ID = os.getenv("FEISHU_APP_ID")
APP_SECRET = os.getenv("FEISHU_APP_SECRET")
VERIFICATION_TOKEN = os.getenv("FEISHU_VERIFICATION_TOKEN", "")

# 会话存储（简单内存，生产可用 Redis）
sessions = {}

def get_session_id(user_id: str) -> str:
    """为每个飞书用户维护一个 session_id"""
    if user_id not in sessions:
        sessions[user_id] = f"feishu_{user_id}_{int(asyncio.get_event_loop().time())}"
    return sessions[user_id]


# 初始化 Channel
channel = FeishuChannel(
    app_id=APP_ID,
    app_secret=APP_SECRET,
    verification_token=VERIFICATION_TOKEN,
)

# 调试：打印 send 方法签名
print("send signature:", inspect.signature(channel.send))


async def handle_message(msg):
    """处理飞书消息"""
    try:
        chat_id = msg.chat_id
        user_id = msg.sender.open_id
        text = msg.content

        # 提取文本内容
        if hasattr(text, 'text'):
            question = text.text.strip()
        else:
            question = str(text).strip()

        if not question:
            return

        # 去除 @ 机器人
        if "@_" in question or "@" in question:
            question = re.sub(r'@[_\w]+', '', question).strip()

        logger.info(f"[飞书] 收到消息 from {user_id}: {question}")

        session_id = get_session_id(user_id)
        history = []
        result = await process_question(question, history, trace_callback=None)
        answer = result.get("result", {}).get("answer", "处理失败，请重试")

        # 直接传入字符串，框架内部自动处理文本消息
        await channel.send(chat_id, answer)

        logger.info(f"[飞书] 回复已发送")

    except Exception as e:
        logger.error(f"[飞书] 处理消息失败: {e}")
        import traceback
        traceback.print_exc()


# 显式注册事件回调
channel.on("message", handle_message)


async def main():
    logger.info("[飞书] 启动长连接...")
    
    await channel.connect()
    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())