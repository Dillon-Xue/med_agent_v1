from openai import OpenAI
from dotenv import load_dotenv
import os

# 读取 .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {
            "role": "user",
            "content": "请介绍一下你自己"
        }
    ]
)

print(response.choices[0].message.content)
