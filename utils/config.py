import os, logging
from openai import OpenAI
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

load_dotenv()


def get_llm_model():
    model = os.getenv("LLM_MODEL_NAME")
    if not model:
        raise ValueError("LLM_MODEL_NAME environment variable is not set")
    return model


def get_embedding_model():
    model = os.getenv("EMBEDDING_MODEL_NAME")
    if not model:
        raise ValueError("EMBEDDING_MODEL_NAME environment variable is not set")
    return model


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
    logger.info(f"[LLMClient] Provider: {provider}")

    if provider == "ollama":
        client = OpenAI(
            base_url=get_ollama_base_url(),
            api_key="ollama",
            timeout=timeout
        )
        model = get_ollama_model()
        logger.info(f"[LLMClient] Using Ollama model: {model}")
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
        logger.info(f"[LLMClient] Using DashScope model: {model}")
        return client, model


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", "INFO").upper()

# ===== 向量检索与文档处理配置 =====
def get_chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "500"))

def get_chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "100"))

def get_max_iterations() -> int:
    return int(os.getenv("MAX_ITERATIONS", "3"))

def get_min_text_length() -> int:
    return int(os.getenv("MIN_TEXT_LENGTH", "50"))

def get_max_upload_size() -> int:
    return int(os.getenv("MAX_UPLOAD_SIZE", str(20 * 1024 * 1024)))

def get_allowed_upload_types() -> set:
    types = os.getenv("ALLOWED_UPLOAD_TYPES", "image/png,image/jpeg,image/jpg,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return set(t.strip() for t in types.split(","))

def setup_logging():
    """配置全局日志格式（chat.py 会调用，其他模块无需重复配置）"""
    level = get_log_level()
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 设置第三方库日志级别为 WARNING，减少噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)