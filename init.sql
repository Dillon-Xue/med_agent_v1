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