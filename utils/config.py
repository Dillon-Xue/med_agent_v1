import os
from dotenv import load_dotenv

load_dotenv()

def get_llm_model():
    return os.getenv("LLM_MODEL_NAME", "qwen-plus")

def get_embedding_model():
    return os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
