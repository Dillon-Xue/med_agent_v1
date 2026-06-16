from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@app.get("/ask")
def ask(question: str):
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": question}]
    )
    answer = response.choices[0].message.content
    return {"answer": answer}
