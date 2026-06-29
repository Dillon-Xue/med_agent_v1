import logging
from cryptography.fernet import Fernet
import os

logger = logging.getLogger(__name__)

# 密钥从环境变量读取（禁止硬编码！）
CRYPTO_KEY = os.getenv("CRYPTO_KEY")
if not CRYPTO_KEY:
    raise RuntimeError(
        "CRYPTO_KEY 环境变量未设置！\n"
        "请在 .env 文件中添加：CRYPTO_KEY=<your-fernet-key>\n"
        "生成密钥方法：在 Python 中执行 Fernet.generate_key().decode()"
    )

cipher = Fernet(CRYPTO_KEY.encode())


def encrypt_text(text: str) -> str:
    """加密文本"""
    if not text or not isinstance(text, str):
        return text
    return cipher.encrypt(text.encode()).decode()


def decrypt_text(encrypted: str) -> str:
    """解密文本"""
    if not encrypted or not isinstance(encrypted, str):
        return encrypted
    try:
        return cipher.decrypt(encrypted.encode()).decode()
    except Exception:
        return encrypted  # 如果解密失败，返回原文（兼容旧数据）


def encrypt_if_needed(text: str) -> str:
    """加密（如果还没加密）"""
    if not text or not isinstance(text, str):
        return text
    # 简单判断：如果已经是加密格式，不再加密
    if text.startswith("gAAAAA"):
        return text
    return encrypt_text(text)


def decrypt_if_needed(text: str) -> str:
    """解密（如果已加密）"""
    if not text or not isinstance(text, str):
        return text
    if text.startswith("gAAAAA"):
        return decrypt_text(text)
    return text
