from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from utils.embeddings import DashscopeEmbeddings
from utils.config import get_llm_client
from dotenv import load_dotenv
import os

load_dotenv()

# 🆕 使用统一客户端工厂
client, model = get_llm_client()

# 向量库目录
VECTOR_DB_DIR = "/mnt/d/A_Study/Agent/Med_Agent/vector_db"

# 加载向量库
vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=DashscopeEmbeddings()
)

app = FastAPI(title="本地医药知识库RAG Agent")

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "本地医药知识库Agent在线，访问 /ask 接口进行问答。"}

@app.post("/ask")
def ask_agent_api(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    docs = vectordb.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    try:
        # 🆕 使用 model 变量，不是 self.model
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是医药研发知识库助手，请基于上下文回答问题。"},
                {"role": "user", "content": f"上下文：{context}\n问题：{question}"}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型调用失败：{str(e)}")

    return {"question": question, "answer": answer}