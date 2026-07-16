import os
from dotenv import load_dotenv
import dashscope
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from utils.embeddings import DashscopeEmbeddings

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
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = file
                page.metadata["page"] = page.metadata.get("page", 0) + 1
            docs.extend(pages)
    return docs

def add_chunk_index(chunks):
    page_chunks = {}
    for chunk in chunks:
        page_key = (chunk.metadata.get("source"), chunk.metadata.get("page", 1))
        if page_key not in page_chunks:
            page_chunks[page_key] = []
        page_chunks[page_key].append(chunk)
    
    for page_key, chunk_list in page_chunks.items():
        for idx, chunk in enumerate(chunk_list, 1):
            chunk.metadata["chunk_index"] = idx
    return chunks

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
    chunks = add_chunk_index(chunks)
    print("chunks:", len(chunks))
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"  Chunk {i}: source={chunk.metadata.get('source')}, page={chunk.metadata.get('page')}, chunk_index={chunk.metadata.get('chunk_index')}")
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