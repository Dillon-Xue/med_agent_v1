import os
from dotenv import load_dotenv
import dashscope
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from utils.embeddings import DashscopeEmbeddings   # 从公共模块导入

load_dotenv()
PROJECT_ROOT = os.getenv("MED_AGENT_ROOT", os.getcwd())
BASE_DATA = os.path.join(PROJECT_ROOT, "data")
BASE_VECTOR = os.path.join(PROJECT_ROOT, "vector_db")
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

embeddings = DashscopeEmbeddings()

CATEGORIES = ["rag", "literature", "drug", "guideline", "risk"]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def load_pdfs(path):
    docs = []
    if not os.path.exists(path):
        print(f"警告: 路径不存在 {path}")
        return docs
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

def build_index(category):
    print(f"\n===== BUILD {category} =====")
    data_path = os.path.join(BASE_DATA, category)
    vector_path = os.path.join(BASE_VECTOR, category)
    docs = load_pdfs(data_path)
    print("pages:", len(docs))
    if not docs:
        print(f"跳过 {category}: 无 PDF 文件")
        return
    chunks = splitter.split_documents(docs)
    print("chunks:", len(chunks))
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vector_path,
        collection_name="langchain"
    )
    print(f"{category} DONE")

if __name__ == "__main__":
    for c in CATEGORIES:
        build_index(c)
    print("\nALL DONE") 
