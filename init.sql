-- 创建患者表（如果不存在）
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    id_card VARCHAR(20) NULL,
    info TEXT NOT NULL,
    diagnosis TEXT,
    tenant_id VARCHAR(50) DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_name_idcard (name, id_card),
    INDEX idx_tenant (tenant_id)
);

-- 创建审批表（如果不存在）
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
    tenant_id VARCHAR(50) DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP,
    INDEX idx_tenant (tenant_id),
    INDEX idx_status (status),
    INDEX idx_requester (requester)
);


-- 会话历史表
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL COMMENT '前端生成或用户标识',
    tenant_id VARCHAR(50) DEFAULT 'default',
    role VARCHAR(20) NOT NULL COMMENT 'user / assistant / system',
    content TEXT NOT NULL,
    tools_used JSON NULL COMMENT '使用的工具列表',
    file_name VARCHAR(255) NULL COMMENT '关联的文件名',
    conversation_type VARCHAR(20) DEFAULT 'quick' COMMENT 'quick / consult / approval',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id, created_at),
    INDEX idx_tenant (tenant_id),
    INDEX idx_type (conversation_type)
);