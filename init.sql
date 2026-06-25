-- ============================================
-- 患者表（结构化字段）
-- ============================================
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL COMMENT '患者姓名',
    gender VARCHAR(10) COMMENT '性别（男/女）',
    age VARCHAR(10) COMMENT '年龄（如：35岁）',
    id_card VARCHAR(255) COMMENT '身份证号',
    phone VARCHAR(255) COMMENT '联系方式',
    address VARCHAR(200) COMMENT '家庭住址',
    allergy VARCHAR(200) COMMENT '过敏史',
    medication VARCHAR(200) COMMENT '当前用药史',
    symptoms VARCHAR(500) COMMENT '症状描述',
    diagnosis VARCHAR(500) COMMENT '临床诊断',
    info TEXT COMMENT '原始信息备份',
    doctor_id VARCHAR(100) DEFAULT 'default' COMMENT '所属医生ID',
    tenant_id VARCHAR(50) DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_name_idcard (name, id_card),
    INDEX idx_tenant (tenant_id),
    INDEX idx_name (name)
);

-- ============================================
-- 审批表（保持不变）
-- ============================================
CREATE TABLE IF NOT EXISTS approvals (
    id VARCHAR(20) PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    type VARCHAR(50) NOT NULL,
    requester VARCHAR(100) NOT NULL,
    requester_role VARCHAR(50),
    reviewer VARCHAR(100),
    reviewer_role VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    comment TEXT,
    doctor_id VARCHAR(100) DEFAULT 'default' COMMENT '所属医生ID',   -- 🆕 新增
    tenant_id VARCHAR(50) DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP,
    INDEX idx_tenant (tenant_id),
    INDEX idx_status (status),
    INDEX idx_requester (requester),
    INDEX idx_doctor (doctor_id)   -- 🆕 新增索引
);

-- ============================================
-- 会话历史表（保持不变）
-- ============================================
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL COMMENT '前端生成或用户标识',
    tenant_id VARCHAR(50) DEFAULT 'default',
    role VARCHAR(20) NOT NULL COMMENT 'user / assistant / system',
    content TEXT NOT NULL,
    tools_used JSON NULL COMMENT '使用的工具列表',
    file_name VARCHAR(255) NULL COMMENT '关联的文件名',
    conversation_type VARCHAR(20) DEFAULT 'quick' COMMENT 'quick / consult / approval',
    trace_id VARCHAR(64) NULL COMMENT 'Trace 会话 ID',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id, created_at),
    INDEX idx_tenant (tenant_id),
    INDEX idx_type (conversation_type)
);


-- ============================================
-- 审计日志表（安全合规）
-- ============================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL COMMENT '操作用户',
    tenant_id VARCHAR(50) DEFAULT 'default' COMMENT '租户ID',
    action VARCHAR(50) NOT NULL COMMENT '操作类型: QUERY/UPDATE/CREATE/APPROVE/REJECT',
    resource_type VARCHAR(50) NOT NULL COMMENT '资源类型: patient/approval/conversation',
    resource_id VARCHAR(100) COMMENT '资源ID',
    detail TEXT COMMENT '操作详情(脱敏后)',
    ip_address VARCHAR(45) COMMENT '客户端IP',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_tenant (tenant_id),
    INDEX idx_action (action),
    INDEX idx_created (created_at)
);