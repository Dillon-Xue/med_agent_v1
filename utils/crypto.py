from cryptography.fernet import Fernet
import os

# 密钥从环境变量读取（不要硬编码！）
CRYPTO_KEY = os.getenv("CRYPTO_KEY")
if not CRYPTO_KEY:
    # 首次启动自动生成（仅开发环境）
    CRYPTO_KEY = Fernet.generate_key().decode()
    logger.warning(f"⚠️ 请将 CRYPTO_KEY={CRYPTO_KEY} 添加到 .env")
    os.environ["CRYPTO_KEY"] = CRYPTO_KEY

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