import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm_model():
    return os.getenv("LLM_MODEL_NAME", "qwen-plus")


def get_embedding_model():
    return os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")


def get_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "dashscope")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

def get_response_mode() -> str:
    return os.getenv("LLM_RESPONSE_MODE", "detailed")

def get_response_max_length() -> int:
    return int(os.getenv("LLM_RESPONSE_MAX_LENGTH", "500"))

# 🆕 统一创建 OpenAI 客户端
def get_llm_client(api_key: str = None, timeout: float = 120.0):
    """
    根据 LLM_PROVIDER 创建对应的 OpenAI 客户端
    返回: (client, model_name)
    """
    provider = get_llm_provider()
    print(f"[LLMClient] Provider: {provider}")

    if provider == "ollama":
        client = OpenAI(
            base_url=get_ollama_base_url(),
            api_key="ollama",
            timeout=timeout
        )
        model = get_ollama_model()
        print(f"[LLMClient] Using Ollama model: {model}")
        return client, model
    else:
        # 默认使用 DashScope
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=timeout
        )
        model = get_llm_model()
        print(f"[LLMClient] Using DashScope model: {model}")
        return client, model