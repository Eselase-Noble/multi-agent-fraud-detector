# Fraud Detection Risk Rules

## Overview
This document outlines the risk scoring rules and thresholds for fraud detection in financial transactions.

## Risk Categories

### 1. Transaction Velocity Risk
**Rule TV-1**: Multiple transactions in short time period
- Threshold: >5 transactions per hour per user
- Risk Score: 0.7
- Action: Flag for review

**Rule TV-2**: Unusual transaction pattern
- Threshold: Transactions 3x average daily amount
- Risk Score: 0.6
- Action: Enhanced monitoring

### 2. Geographic Risk
**Rule GR-1**: Cross-border transaction anomalies
- Threshold: Transaction from country not in user's history
- Risk Score: 0.8
- Action: Require additional verification

**Rule GR-2**: Distance-based risk
- Threshold: >500km from user's usual location
- Risk Score: 0.5
- Action: Monitor closely

### 3. Merchant Risk
**Rule MR-1**: High-risk merchant categories
- Categories: Gambling, Adult Entertainment, Cryptocurrency
- Risk Score: 0.9
- Action: Enhanced due diligence

**Rule MR-2**: New merchant relationship
- Threshold: First transaction with merchant
- Risk Score: 0.4
- Action: Standard monitoring

### 4. Amount-Based Risk
**Rule AR-1**: Large transaction amounts
- Threshold: >$10,000
- Risk Score: 0.7
- Action: Mandatory review

**Rule AR-2**: Round number transactions
- Threshold: Multiple transactions at round amounts ($100, $500, $1000)
- Risk Score: 0.6
- Action: Pattern analysis

### 5. Time-Based Risk
**Rule TR-1**: Unusual transaction hours
- Threshold: Transactions between 2 AM - 5 AM local time
- Risk Score: 0.5
- Action: Time pattern analysis

**Rule TR-2**: Weekend vs weekday patterns
- Threshold: Business transactions on weekends
- Risk Score: 0.4
- Action: Contextual review

## Composite Risk Scoring

### Risk Aggregation Formula
Overall Risk Score = 
  (Velocity Risk × 0.25) +
  (Geographic Risk × 0.20) +
  (Merchant Risk × 0.25) +
  (Amount Risk × 0.20) +
  (Time Risk × 0.10)

### Risk Level Thresholds
- **Low Risk**: 0.0 - 0.3
  - Action: Standard processing
- **Medium Risk**: 0.31 - 0.6
  - Action: Additional verification
- **High Risk**: 0.61 - 0.8
  - Action: Manual review required
- **Critical Risk**: 0.81 - 1.0
  - Action: Block transaction, alert team

## Review and Update Process

### Quarterly Review
All risk rules must be reviewed quarterly by the Risk Committee.

### Exception Handling
Exceptions to rules must be documented and approved by:
1. Risk Analyst
2. Team Lead
3. Compliance Officer

### Version Control
- Version: 2.3
- Effective Date: October 15, 2024
- Previous Version: 2.2
- Next Review: January 15, 2025

## Compliance References
- PCI DSS Requirement 10: Track and monitor access
- GDPR Article 32: Security of processing
- ISO 27001: Information security management

## Contact Information
For questions about these rules, contact:
- Risk Management Team: risk@company.com
- Compliance Department: compliance@company.com
- Fraud Investigation: fraud@company.com