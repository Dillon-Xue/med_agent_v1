import os, re, logging
from utils.response import build_response
from utils.crypto import encrypt_if_needed, decrypt_if_needed
from utils.database import get_connection
logger = logging.getLogger(__name__)

class ApprovalTool:
    def __init__(self):
        pass

    def _get_connection(self):
        return get_connection()

    def _get_doctor_id(self) -> str:
        """获取当前医生ID（用于数据隔离）"""
        try:
            import chat
            if hasattr(chat, 'current_session_user') and chat.current_session_user:
                return chat.current_session_user
        except ImportError:
            pass
        return "default"

    def _get_tenant(self):
        try:
            import chat
            if hasattr(chat, 'get_current_tenant'):
                return chat.get_current_tenant()
        except ImportError:
            pass
        return "default"

    def _get_current_user(self, query: str = "") -> str:
        try:
            import chat
            if hasattr(chat, 'current_session_user') and chat.current_session_user:
                logger.info(f"[ApprovalTool] 从 chat 全局变量获取用户: {chat.current_session_user}")
                return chat.current_session_user
        except ImportError:
            pass
        # 从 query 中提取（备用）
        match = re.search(r'用户[：:]\s*(\S+)', query)
        if match:
            logger.info(f"[ApprovalTool] 从 query 解析用户: {match.group(1)}")
            return match.group(1)
        return "current_user"

    def create(self, title: str, content: str, type: str, requester: str, reviewer: str = None) -> dict:
        import time
        import random
        approval_id = f"APP-{time.strftime('%Y%m%d')}-{random.randint(100, 999)}"
        tenant_id = self._get_tenant()
        doctor_id = self._get_doctor_id()
        reviewer = reviewer or requester
        content_encrypted = encrypt_if_needed(content)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO approvals (id, title, content, type, requester, reviewer, doctor_id, tenant_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (approval_id, title, content_encrypted, type, requester, reviewer, doctor_id, tenant_id)
        )
        conn.commit()
        conn.close()
        return build_response(
            answer=f"✅ 已创建审批项 {approval_id}：{title}",
            source="approval",
            debug={"approval_id": approval_id}
        )

    def list_pending_by_user(self, user: str) -> list:
        logger.info(f"[ApprovalTool.list_pending_by_user] user: {user}")
        tenant_id = self._get_tenant()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, requester, created_at FROM approvals "
            "WHERE tenant_id = %s AND reviewer = %s AND status = 'pending' "
            "ORDER BY created_at DESC",
            (tenant_id, user)
        )
        rows = cursor.fetchall()
        conn.close()
        logger.info(f"[ApprovalTool.list_pending_by_user] 查询到 {len(rows)} 条记录")
        # 将 created_at 转换为字符串格式
        return [{
            "id": row.get("id"),
            "title": row.get("title"),
            "requester": row.get("requester"),
            "created_at": row.get("created_at").strftime("%Y-%m-%d %H:%M:%S") if row.get("created_at") else ""
        } for row in rows]

    def list_pending(self, query: str = "") -> dict:
        user = self._get_current_user(query)
        tenant_id = self._get_tenant()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, requester, created_at FROM approvals "
            "WHERE tenant_id = %s AND reviewer = %s AND status = 'pending' "
            "ORDER BY created_at DESC",
            (tenant_id, user)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return build_response(
                answer="✅ 当前没有待您审批的项",
                source="approval"
            )

        lines = ["📋 待审批列表（等待您审批）："]
        for row in rows:
            lines.append(f"  [{row.get('id')}] {row.get('title')} - 申请人：{row.get('requester')} - {row.get('created_at')}")
        lines.append("\n输入「审批通过 [编号]」或「驳回 [编号] [原因]」")
        return build_response(
            answer="\n".join(lines),
            source="approval",
            debug={"items": rows}
        )

    def list_approved(self, query: str = "") -> dict:
        user = self._get_current_user(query)
        tenant_id = self._get_tenant()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, requester, created_at, reviewed_at FROM approvals "
            "WHERE tenant_id = %s AND reviewer = %s AND status = 'approved' "
            "ORDER BY reviewed_at DESC",
            (tenant_id, user)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return build_response(
                answer="✅ 您当前没有已通过的审批",
                source="approval"
            )

        lines = ["📋 已通过审批列表："]
        for row in rows:
            lines.append(f"  [{row.get('id')}] {row.get('title')} - 申请人：{row.get('requester')} - 审批时间：{row.get('reviewed_at')}")
        return build_response(
            answer="\n".join(lines),
            source="approval",
            debug={"items": rows}
        )

    def list_rejected(self, query: str = "") -> dict:
        user = self._get_current_user(query)
        tenant_id = self._get_tenant()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, requester, comment, created_at, reviewed_at FROM approvals "
            "WHERE tenant_id = %s AND reviewer = %s AND status = 'rejected' "
            "ORDER BY reviewed_at DESC",
            (tenant_id, user)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return build_response(
                answer="✅ 您当前没有已驳回的审批",
                source="approval"
            )

        lines = ["📋 已驳回审批列表："]
        for row in rows:
            comment = row.get('comment') or "未填写原因"
            lines.append(f"  [{row.get('id')}] {row.get('title')} - 申请人：{row.get('requester')} - 驳回原因：{comment}")
        return build_response(
            answer="\n".join(lines),
            source="approval",
            debug={"items": rows}
        )

    def list_done(self, query: str = "") -> dict:
        user = self._get_current_user(query)
        tenant_id = self._get_tenant()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, requester, status, created_at, reviewed_at FROM approvals "
            "WHERE tenant_id = %s AND reviewer = %s AND status IN ('approved', 'rejected') "
            "ORDER BY reviewed_at DESC",
            (tenant_id, user)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return build_response(
                answer="✅ 您当前没有已完成的审批",
                source="approval"
            )

        status_map = {"approved": "✅ 已通过", "rejected": "❌ 已驳回"}
        lines = ["📋 已审批列表（已完成）："]
        for row in rows:
            status_text = status_map.get(row.get('status'), row.get('status'))
            lines.append(f"  [{row.get('id')}] {row.get('title')} - {status_text}")
            lines.append(f"    申请人：{row.get('requester')} | {row.get('reviewed_at')}")
        return build_response(
            answer="\n".join(lines),
            source="approval",
            debug={"items": rows}
        )

    def list_all(self, query: str = "") -> dict:
        user = self._get_current_user(query)
        tenant_id = self._get_tenant()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, requester, status, created_at, reviewed_at FROM approvals "
            "WHERE tenant_id = %s AND reviewer = %s "
            "ORDER BY created_at DESC",
            (tenant_id, user)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return build_response(
                answer="📋 您当前没有任何审批记录",
                source="approval"
            )

        status_map = {"pending": "⏳ 待审批", "approved": "✅ 已通过", "rejected": "❌ 已驳回"}
        lines = ["📋 全部审批列表（您作为审批人）："]
        for row in rows:
            status_text = status_map.get(row.get('status'), row.get('status'))
            lines.append(f"  [{row.get('id')}] {row.get('title')} - {status_text}")
            lines.append(f"    申请人：{row.get('requester')} | {row.get('created_at')}")
        return build_response(
            answer="\n".join(lines),
            source="approval",
            debug={"items": rows}
        )

    def approve(self, approval_id: str, query: str = "") -> dict:
        approval_id = approval_id.strip()
        user = self._get_current_user(query)
        doctor_id = self._get_doctor_id()
        logger.debug(f"[DEBUG] approve - id: {approval_id}, user: {user}, doctor_id: {doctor_id}")

        conn = self._get_connection()
        cursor = conn.cursor()

        # 先查询当前记录
        cursor.execute(
            "SELECT id, reviewer, status FROM approvals WHERE id = %s AND reviewer = %s",
            (approval_id, user)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return build_response(
                answer=f"❌ 未找到审批项 {approval_id}",
                source="approval",
                success=False
            )
        logger.debug(f"[DEBUG] approve - 当前记录: id={row.get('id')}, reviewer={row.get('reviewer')}, status={row.get('status')}")

        if row.get('reviewer') != user:
            conn.close()
            return build_response(
                answer=f"❌ 您不是 {approval_id} 的审批人（当前审批人：{row.get('reviewer')}）",
                source="approval",
                success=False
            )
        if row.get('status') != "pending":
            conn.close()
            return build_response(
                answer=f"❌ {approval_id} 当前状态为 {row.get('status')}，无法审批",
                source="approval",
                success=False
            )

        # 执行更新
        cursor.execute(
            "UPDATE approvals SET status='approved', reviewed_at=NOW() WHERE id = %s AND reviewer = %s AND status = 'pending'",
            (approval_id, user)
        )
        affected = cursor.rowcount
        conn.commit()

        # ===== 🆕 数据回流：审批通过后写入记忆库 =====
        if affected:
            try:
                # 查询完整审批内容
                cursor.execute(
                    "SELECT content, type, requester FROM approvals WHERE id = %s",
                    (approval_id,)
                )
                approval_row = cursor.fetchone()
                if approval_row:
                    content = approval_row.get('content')
                    approval_type = approval_row.get('type')
                    requester = approval_row.get('requester')

                    # 仅对用药评估类型触发数据回流
                    if approval_type == "medication_evaluation":
                        self._write_to_memory(approval_id, content, requester, doctor_id)
            except Exception as e:
                logger.error(f"[数据回流] 写入记忆库失败: {e}")
                # 不影响审批主流程

        conn.close()

        try:
            from utils.audit import log_audit
            log_audit(
                action="APPROVE",
                resource_type="approval",
                resource_id=approval_id,
                detail={"status": "approved", "reviewer": user},
                ip=None
            )
        except Exception as e:
            logger.error(f"[Audit] 审批日志记录失败: {e}")

        logger.info(f"[DEBUG] approve - affected rows: {affected}")

        if affected:
            return build_response(
                answer=f"✅ 审批 {approval_id} 已通过，审批人：{user}",
                source="approval"
            )
        return build_response(
            answer=f"❌ 审批 {approval_id} 失败，请重试",
            source="approval",
            success=False
        )

    # ===== 🆕 数据回流辅助方法 =====
    def _write_to_memory(self, approval_id: str, content: str, requester: str, doctor_id: str):
        """
        将审批通过的内容写入记忆向量库
        """
        import re
        from tools.memory_tool import MemoryTool
        from utils.crypto import decrypt_if_needed

        # 解密 content（存储时是加密的）
        decrypted_content = decrypt_if_needed(content)
        logger.info(f"[数据回流] 审批 {approval_id} 解密后的 content 预览:\n{decrypted_content[:500]}")

        # 解析 content 中的关键字段
        def extract_field(text: str, field_name: str) -> str:
            pattern = rf'{field_name}[：:]\s*([^\n]+)'
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
            else:
                logger.warning(f"[数据回流] 未找到字段 '{field_name}'")
                return ""

        patient_name = extract_field(decrypted_content, "患者姓名")
        diagnosis = extract_field(decrypted_content, "临床诊断")
        medications = extract_field(decrypted_content, "目前用药")
        assessment = extract_field(decrypted_content, "评估结果")
        medication_goal = extract_field(decrypted_content, "用药目标")
        precautions = extract_field(decrypted_content, "用药注意事项")

        if not patient_name or not diagnosis:
            logger.warning(f"[数据回流] 审批 {approval_id} 缺少患者姓名或诊断，跳过写入")
            return

        try:
            memory_tool = MemoryTool()
            memory_tool.remember(
                patient_name=patient_name,
                diagnosis=diagnosis,
                medications=medications,
                assessment=assessment,
                medication_goal=medication_goal,
                precautions=precautions,
                doctor_id=doctor_id,
                approval_id=approval_id,
                requester=requester
            )
            logger.info(f"[数据回流] 审批 {approval_id} 已写入记忆库：{patient_name} - {diagnosis}")
        except Exception as e:
            logger.error(f"[数据回流] 写入记忆库失败: {e}")
            raise

    def reject(self, approval_id: str, comment: str = "", query: str = "") -> dict:
        approval_id = approval_id.strip()
        user = self._get_current_user(query)
        doctor_id = self._get_doctor_id()
        logger.debug(f"[DEBUG] reject - id: {approval_id}, user: {user}, doctor_id: {doctor_id}, comment: {comment}")

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, reviewer, status FROM approvals WHERE id = %s AND reviewer = %s",
            (approval_id, user)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return build_response(
                answer=f"❌ 未找到审批项 {approval_id}",
                source="approval",
                success=False
            )
        logger.debug(f"[DEBUG] reject - 当前记录: id={row.get('id')}, reviewer={row.get('reviewer')}, status={row.get('status')}")

        if row.get('reviewer') != user:
            conn.close()
            return build_response(
                answer=f"❌ 您不是 {approval_id} 的审批人（当前审批人：{row.get('reviewer')}）",
                source="approval",
                success=False
            )
        if row.get('status') != "pending":
            conn.close()
            return build_response(
                answer=f"❌ {approval_id} 当前状态为 {row.get('status')}，无法驳回",
                source="approval",
                success=False
            )

        cursor.execute(
            "UPDATE approvals SET status='rejected', comment=%s, reviewed_at=NOW() "
            "WHERE id = %s AND reviewer = %s AND status = 'pending'",
            (comment, approval_id, user)
        )
        affected = cursor.rowcount
        conn.commit()
        try:
            from utils.audit import log_audit
            log_audit(
                action="REJECT",
                resource_type="approval",
                resource_id=approval_id,
                detail={"status": "rejected", "comment": comment, "reviewer": user},
                ip=None
            )
        except Exception as e:
            logger.error(f"[Audit] 审批日志记录失败: {e}")
        conn.close()
        logger.debug(f"[DEBUG] reject - affected rows: {affected}")

        if affected:
            return build_response(
                answer=f"✅ 审批 {approval_id} 已驳回，审批人：{user}，原因：{comment or '未填写'}",
                source="approval"
            )
        return build_response(
            answer=f"❌ 驳回 {approval_id} 失败，请重试",
            source="approval",
            success=False
        )

    def run(self, query: str) -> dict:
        logger.debug(f"[DEBUG] ApprovalTool.run - 收到查询: {query[:200]}...")

        # 1. 列表查询
        if "待审批" in query:
            return self.list_pending(query)
        if "已驳回" in query or "驳回列表" in query:
            return self.list_rejected(query)
        if "已通过" in query or "通过列表" in query:
            return self.list_approved(query)
        if "已审批" in query:
            return self.list_done(query)
        if "全部列表" in query or "所有审批" in query:
            return self.list_all(query)

        # 2. 审批操作
        match = re.search(r'审批通过\s*[:：]?\s*([A-Z0-9\-]+)', query)
        if match:
            approval_id = match.group(1)
            logger.info(f"[DEBUG] 匹配到审批通过: {approval_id}")
            return self.approve(approval_id, query)

        match = re.search(r'驳回\s*[:：]?\s*([A-Z0-9\-]+)\s*(.*?)$', query)
        if match:
            approval_id = match.group(1)
            comment = match.group(2).strip()
            return self.reject(approval_id, comment, query)

        return build_response(
            answer="""审批工具支持：
- 待审批列表 - 查看您作为审批人的待审批项
- 已通过列表 - 查看您已通过的审批
- 已驳回列表 - 查看您已驳回的审批
- 已审批列表 - 查看您已完成的审批（通过+驳回）
- 全部列表 - 查看您作为审批人的所有审批
- 审批通过 APP-001 - 通过指定审批
- 驳回 APP-001 原因：资料不齐 - 驳回并填写原因""",
            source="approval",
            success=False
        )