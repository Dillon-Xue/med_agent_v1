from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.embeddings.create(
    model="text-embedding-v4",
    input="清解散火颗粒"
)

print(len(response.data[0].embedding))
