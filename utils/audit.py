import os, json, logging
from datetime import datetime
from utils.database import get_connection
logger = logging.getLogger(__name__)

def get_tenant():
    """获取当前租户ID"""
    try:
        import chat
        if hasattr(chat, 'get_current_tenant'):
            return chat.get_current_tenant()
    except ImportError:
        pass
    return "default"


def get_current_user():
    """获取当前用户"""
    try:
        from utils.thread_context import doctor_id_var
        user = doctor_id_var.get()
        if user:
            return user
    except Exception:
        pass
    try:
        import chat
        if hasattr(chat, 'current_session_user') and chat.current_session_user:
            return chat.current_session_user
    except ImportError:
        pass
    return "system"


def log_audit(action: str, resource_type: str, resource_id: str = None,
              detail: dict = None, ip: str = None):
    """
    记录审计日志

    Args:
        action: 操作类型 (QUERY/UPDATE/CREATE/APPROVE/REJECT)
        resource_type: 资源类型 (patient/approval/conversation)
        resource_id: 资源ID
        detail: 操作详情（会脱敏）
        ip: 客户端IP
    """
    logger.debug(f"[Audit] 被调用: {action} {resource_type} {resource_id}")
    try:
        # 导入脱敏函数
        from utils.response import mask_sensitive

        user_id = get_current_user()
        tenant_id = get_tenant()

        # 对 detail 中的敏感信息脱敏
        if detail:
            # 如果是 dict，转为 JSON 字符串后脱敏
            if isinstance(detail, dict):
                detail_str = json.dumps(detail, ensure_ascii=False)
            else:
                detail_str = str(detail)
            detail_str = mask_sensitive(detail_str)
        else:
            detail_str = None

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO audit_logs
               (user_id, tenant_id, action, resource_type, resource_id, detail, ip_address)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (user_id, tenant_id, action, resource_type, resource_id, detail_str, ip)
        )
        conn.commit()
        conn.close()
        logger.debug(f"[Audit] {action} | {resource_type} | {resource_id} | {user_id}")
    except Exception as e:
        logger.error(f"[Audit] 记录失败: {e}")
