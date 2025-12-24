-- Create database for fraud analysis
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    amount DECIMAL(15, 2),
    currency VARCHAR(3),
    merchant_id VARCHAR(50),
    merchant_category VARCHAR(100),
    merchant_country VARCHAR(2),
    user_country VARCHAR(2),
    transaction_time TIMESTAMP,
    status VARCHAR(20),
    fraud_score DECIMAL(5, 4),
    features JSONB
);

CREATE TABLE IF NOT EXISTS fraud_cases (
    case_id SERIAL PRIMARY KEY,
    transaction_ids TEXT[],
    fraud_type VARCHAR(50),
    description TEXT,
    amount_lost DECIMAL(15, 2),
    detection_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    status VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,
    policy_type VARCHAR(50),
    region VARCHAR(10),
    version VARCHAR(20),
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    action VARCHAR(100),
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    request_body JSONB,
    response_body JSONB,
    status_code INTEGER,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transactions_time ON transactions(transaction_time);
CREATE INDEX idx_transactions_user ON transactions(user_id);
CREATE INDEX idx_fraud_cases_status ON fraud_cases(status);
CREATE INDEX idx_policies_type_region ON policies(policy_type, region);

-- Insert sample data
INSERT INTO policies (title, content, policy_type, region, version) VALUES
('AML Policy 2024', 'All transactions above $10,000 must be reported within 24 hours. Enhanced due diligence required for high-risk jurisdictions.', 'AML', 'Global', '2024.1'),
('KYC Verification Rules', 'Customer identity must be verified using two independent sources. Biometric verification required for accounts over $50,000.', 'KYC', 'Global', '2024.1'),
('EU Transaction Monitoring', 'Real-time monitoring of cross-border transactions within EU. Flag transactions with velocity > 5x normal pattern.', 'Monitoring', 'EU', '2024.2');