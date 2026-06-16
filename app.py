from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from ingest import DashscopeEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化百炼客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 向量库目录
VECTOR_DB_DIR = "/mnt/d/A_Study/Agent/Med_Agent/vector_db"

# 加载向量库
vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=DashscopeEmbeddings()
)

# FastAPI应用
app = FastAPI(title="本地医药知识库RAG Agent")

# 请求体模型
class QuestionRequest(BaseModel):
    question: str

# 根接口
@app.get("/")
def read_root():
    return {"message": "本地医药知识库Agent在线，访问 /ask 接口进行问答。"}

# 问答接口
@app.post("/ask")
def ask_agent_api(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 检索 Top3 相似文本
    docs = vectordb.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 调用 Qwen 生成答案
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是医药研发知识库助手，请基于上下文回答问题。"},
                {"role": "user", "content": f"上下文：{context}\n问题：{question}"}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型调用失败：{str(e)}")

    return {"question": question, "answer": answer}
