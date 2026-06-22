import os
import re
import pymysql
from utils.response import build_response


class ApprovalTool:
    def __init__(self):
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_user = os.getenv("DB_USER", "root")
        self.db_password = os.getenv("DB_PASSWORD", "yourpassword")
        self.db_name = os.getenv("DB_NAME", "patient_db")

    def _get_connection(self):
        return pymysql.connect(
            host=self.db_host,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            charset='utf8mb4'
        )

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
                print(f"[ApprovalTool] 从 chat 全局变量获取用户: {chat.current_session_user}")
                return chat.current_session_user
        except ImportError:
            pass
        # 从 query 中提取（备用）
        match = re.search(r'用户[：:]\s*(\S+)', query)
        if match:
            print(f"[ApprovalTool] 从 query 解析用户: {match.group(1)}")
            return match.group(1)
        return "current_user"

    def create(self, title: str, content: str, type: str, requester: str, reviewer: str = None) -> dict:
        import time
        import random
        approval_id = f"APP-{time.strftime('%Y%m%d')}-{random.randint(100, 999)}"
        tenant_id = self._get_tenant()
        reviewer = reviewer or requester
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO approvals (id, title, content, type, requester, reviewer, tenant_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (approval_id, title, content, type, requester, reviewer, tenant_id)
        )
        conn.commit()
        conn.close()
        return build_response(
            answer=f"✅ 已创建审批项 {approval_id}：{title}",
            source="approval",
            debug={"approval_id": approval_id}
        )

    def list_pending_by_user(self, user: str) -> list:
        print(f"[ApprovalTool.list_pending_by_user] user: {user}")
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
        print(f"[ApprovalTool.list_pending_by_user] 查询到 {len(rows)} 条记录")
        # 将 created_at 转换为字符串格式
        return [{
            "id": row[0],
            "title": row[1],
            "requester": row[2],
            "created_at": row[3].strftime("%Y-%m-%d %H:%M:%S") if row[3] else ""
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
            lines.append(f"  [{row[0]}] {row[1]} - 申请人：{row[2]} - {row[3]}")
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
            lines.append(f"  [{row[0]}] {row[1]} - 申请人：{row[2]} - 审批时间：{row[4]}")
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
            comment = row[3] or "未填写原因"
            lines.append(f"  [{row[0]}] {row[1]} - 申请人：{row[2]} - 驳回原因：{comment}")
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
            status_text = status_map.get(row[3], row[3])
            lines.append(f"  [{row[0]}] {row[1]} - {status_text}")
            lines.append(f"    申请人：{row[2]} | {row[5]}")
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
            status_text = status_map.get(row[3], row[3])
            lines.append(f"  [{row[0]}] {row[1]} - {status_text}")
            lines.append(f"    申请人：{row[2]} | {row[4]}")
        return build_response(
            answer="\n".join(lines),
            source="approval",
            debug={"items": rows}
        )

    def approve(self, approval_id: str, query: str = "") -> dict:
        approval_id = approval_id.strip()
        user = self._get_current_user(query)
        print(f"[DEBUG] approve - id: {approval_id}, user: {user}")

        conn = self._get_connection()
        cursor = conn.cursor()

        # 先查询当前记录
        cursor.execute(
            "SELECT id, reviewer, status FROM approvals WHERE id = %s",
            (approval_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return build_response(
                answer=f"❌ 未找到审批项 {approval_id}",
                source="approval",
                success=False
            )
        print(f"[DEBUG] approve - 当前记录: id={row[0]}, reviewer={row[1]}, status={row[2]}")

        if row[1] != user:
            conn.close()
            return build_response(
                answer=f"❌ 您不是 {approval_id} 的审批人（当前审批人：{row[1]}）",
                source="approval",
                success=False
            )
        if row[2] != "pending":
            conn.close()
            return build_response(
                answer=f"❌ {approval_id} 当前状态为 {row[2]}，无法审批",
                source="approval",
                success=False
            )

        # 执行更新
        cursor.execute(
            "UPDATE approvals SET status='approved', reviewed_at=NOW() WHERE id = %s AND status = 'pending'",
            (approval_id,)
        )
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"[DEBUG] approve - affected rows: {affected}")

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

    def reject(self, approval_id: str, comment: str = "", query: str = "") -> dict:
        approval_id = approval_id.strip()
        user = self._get_current_user(query)
        print(f"[DEBUG] reject - id: {approval_id}, user: {user}, comment: {comment}")

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, reviewer, status FROM approvals WHERE id = %s",
            (approval_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return build_response(
                answer=f"❌ 未找到审批项 {approval_id}",
                source="approval",
                success=False
            )
        print(f"[DEBUG] reject - 当前记录: id={row[0]}, reviewer={row[1]}, status={row[2]}")

        if row[1] != user:
            conn.close()
            return build_response(
                answer=f"❌ 您不是 {approval_id} 的审批人（当前审批人：{row[1]}）",
                source="approval",
                success=False
            )
        if row[2] != "pending":
            conn.close()
            return build_response(
                answer=f"❌ {approval_id} 当前状态为 {row[2]}，无法驳回",
                source="approval",
                success=False
            )

        cursor.execute(
            "UPDATE approvals SET status='rejected', comment=%s, reviewed_at=NOW() "
            "WHERE id = %s AND status = 'pending'",
            (comment, approval_id)
        )
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"[DEBUG] reject - affected rows: {affected}")

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
        # 直接使用原始 query，不提取“当前问题”
        print(f"[DEBUG] ApprovalTool.run - 收到查询: {query[:200]}...")

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

        # 2. 审批操作（直接匹配，不依赖“当前问题”）
        match = re.search(r'审批通过\s*[:：]?\s*([A-Z0-9\-]+)', query)
        if match:
            approval_id = match.group(1)
            print(f"[DEBUG] 匹配到审批通过: {approval_id}")
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