import asyncio
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
import random
from app.db.postgres import PostgreSQLManager
from app.db.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataSeeder:
    """Production-grade data seeder for development and testing."""

    def __init__(self):
        self.db = PostgreSQLManager()
        self.vector_store = VectorStore()

        # Sample data templates
        self.merchant_categories = [
            "Retail", "E-commerce", "Travel", "Entertainment",
            "Food & Dining", "Groceries", "Electronics", "Healthcare",
            "Education", "Gambling", "Cryptocurrency", "Utilities"
        ]

        self.countries = ["US", "UK", "CA", "AU", "DE", "FR", "JP", "SG"]

        self.fraud_types = [
            "card_not_present", "account_takeover", "money_laundering",
            "identity_theft", "merchant_collusion", "friendly_fraud"
        ]

    async def initialize(self):
        """Initialize database connections."""
        await self.db.initialize()
        await self.vector_store.initialize()

    async def seed_transactions(self, count: int = 1000):
        """Seed transaction data."""
        logger.info(f"Seeding {count} transactions...")

        users = [f"user_{i:04d}" for i in range(1, 101)]
        merchants = [f"merchant_{i:04d}" for i in range(1, 51)]

        for i in range(count):
            transaction_data = {
                "transaction_id": f"txn_{datetime.now().strftime('%Y%m%d')}_{i:06d}",
                "user_id": random.choice(users),
                "amount": round(random.uniform(10, 5000), 2),
                "currency": "USD",
                "merchant_id": random.choice(merchants),
                "merchant_category": random.choice(self.merchant_categories),
                "merchant_country": random.choice(self.countries),
                "user_country": random.choice(self.countries),
                "transaction_time": datetime.now() - timedelta(days=random.randint(0, 30)),
                "status": random.choice(["completed", "pending", "failed"]),
                "fraud_score": round(random.uniform(0, 0.3), 4),  # Mostly low fraud scores
                "features": {
                    "device_type": random.choice(["mobile", "desktop", "tablet"]),
                    "ip_country": random.choice(self.countries),
                    "browser": random.choice(["chrome", "firefox", "safari"]),
                    "time_of_day": random.randint(0, 23)
                }
            }

            # Occasionally add high fraud score transactions
            if random.random() < 0.05:  # 5% high fraud
                transaction_data["fraud_score"] = round(random.uniform(0.7, 1.0), 4)
                transaction_data["status"] = "flagged"

            try:
                await self.db.insert_transaction(transaction_data)
            except Exception as e:
                logger.error(f"Failed to seed transaction {i}: {e}")

        logger.info(f"Seeded {count} transactions")

    async def seed_fraud_cases(self, count: int = 50):
        """Seed fraud case data."""
        logger.info(f"Seeding {count} fraud cases...")

        # Get some transaction IDs for fraud cases
        transactions = await self.db.get_transaction_data(limit=count * 3)
        transaction_ids = [tx["transaction_id"] for tx in transactions]

        for i in range(count):
            # Select 1-3 transactions for this case
            case_transactions = random.sample(
                transaction_ids,
                k=random.randint(1, min(3, len(transaction_ids)))
            )

            case_data = {
                "transaction_ids": case_transactions,
                "fraud_type": random.choice(self.fraud_types),
                "description": f"Fraud case #{i + 1}: {random.choice(['Suspicious pattern', 'Multiple chargebacks', 'Unusual activity'])}",
                "amount_lost": round(random.uniform(100, 10000), 2),
                "detection_method": random.choice(["system", "manual", "customer_report"]),
                "status": random.choice(["open", "investigating", "resolved"])
            }

            try:
                await self.db.create_fraud_case(case_data)
            except Exception as e:
                logger.error(f"Failed to seed fraud case {i}: {e}")

        logger.info(f"Seeded {count} fraud cases")

    async def seed_policy_documents(self):
        """Seed policy documents to vector store."""
        logger.info("Seeding policy documents...")

        policy_documents = [
            {
                "content": """Anti-Money Laundering (AML) Policy

Article 1: Scope and Purpose
This policy establishes guidelines to prevent money laundering activities through our financial systems.

Article 2: Customer Due Diligence
All customers must undergo identity verification using at least two independent sources.

Article 3: Transaction Monitoring
Transactions above $10,000 must be reported within 24 hours. Enhanced monitoring required for high-risk jurisdictions.

Article 4: Record Keeping
All transaction records must be maintained for a minimum of 5 years.

Article 5: Compliance Training
All employees must complete AML training annually.

Effective Date: January 1, 2024
Version: 2024.1""",
                "metadata": {
                    "title": "AML Policy 2024",
                    "policy_type": "AML",
                    "region": "Global",
                    "version": "2024.1",
                    "effective_date": "2024-01-01",
                    "source": "compliance_department"
                }
            },
            {
                "content": """Know Your Customer (KYC) Procedures

Section 1: Customer Identification
1.1 Government-issued photo ID required
1.2 Proof of address (utility bill, bank statement)
1.3 Tax identification number for business accounts

Section 2: Risk Assessment
2.1 Customers scored based on:
   - Geographic location
   - Business type
   - Transaction patterns
   - Political exposure

Section 3: Enhanced Due Diligence
Required for:
- High-value accounts (>$50,000)
- High-risk countries
- Politically exposed persons

Section 4: Ongoing Monitoring
Regular review of customer transactions and profile updates.

Version: 2024.2
Applicable: All regions""",
                "metadata": {
                    "title": "KYC Procedures Manual",
                    "policy_type": "KYC",
                    "region": "Global",
                    "version": "2024.2",
                    "effective_date": "2024-03-15",
                    "source": "compliance_department"
                }
            },
            {
                "content": """Fraud Detection and Prevention Guidelines

Chapter 1: Transaction Monitoring Rules
1. Velocity monitoring: Flag >5 transactions/hour
2. Geographic anomalies: Transactions from unusual locations
3. Amount monitoring: Unusually large or small transactions
4. Time patterns: Transactions outside normal hours

Chapter 2: Fraud Typologies
2.1 Card Not Present (CNP) fraud
2.2 Account takeover
2.3 Identity theft
2.4 Merchant collusion
2.5 Friendly fraud

Chapter 3: Investigation Procedures
3.1 Initial assessment within 2 hours
3.2 Evidence collection requirements
3.3 Reporting timelines
3.4 Customer communication protocols

Chapter 4: Prevention Measures
4.1 Multi-factor authentication
4.2 Behavioral analytics
4.3 Machine learning models
4.4 Manual review thresholds

Last Updated: June 2024""",
                "metadata": {
                    "title": "Fraud Detection Guidelines",
                    "policy_type": "Fraud",
                    "region": "Global",
                    "version": "2024.3",
                    "effective_date": "2024-06-01",
                    "source": "fraud_department"
                }
            },
            {
                "content": """EU Payment Services Directive (PSD2) Compliance

Article 1: Strong Customer Authentication
1.1 Two-factor authentication required for all electronic payments
1.2 Exceptions for low-risk transactions (<€30)
1.3 Dynamic linking of transaction details

Article 2: Access to Payment Accounts
2.1 Third-party provider requirements
2.2 Secure communication standards
2.3 Liability provisions

Article 3: Transaction Risk Analysis
3.1 Real-time risk scoring
3.2 Exemption handling procedures
3.3 Fraud rate monitoring

Article 4: Regulatory Reporting
4.1 Quarterly fraud reporting
4.2 Incident notification requirements
4.3 Audit trail maintenance

Applicable Region: European Union
Effective: January 2024""",
                "metadata": {
                    "title": "PSD2 Compliance Requirements",
                    "policy_type": "Compliance",
                    "region": "EU",
                    "version": "2024.1",
                    "effective_date": "2024-01-01",
                    "source": "legal_department"
                }
            },
            {
                "content": """Risk Management Framework

Section A: Risk Assessment Methodology
A.1 Quantitative risk scoring (1-100)
A.2 Qualitative risk factors
A.3 Combined risk rating

Section B: Risk Mitigation Controls
B.1 Preventive controls
B.2 Detective controls
B.3 Corrective controls

Section C: Risk Monitoring
C.1 Daily risk dashboards
C.2 Monthly risk committee reviews
C.3 Quarterly risk assessments

Section D: Risk Reporting
D.1 Executive risk reports
D.2 Regulatory reporting
D.3 Audit reporting

Framework Version: 3.0
Approval Date: March 2024""",
                "metadata": {
                    "title": "Enterprise Risk Management Framework",
                    "policy_type": "Risk",
                    "region": "Global",
                    "version": "3.0",
                    "effective_date": "2024-03-01",
                    "source": "risk_department"
                }
            }
        ]

        try:
            document_ids = await self.vector_store.add_documents(policy_documents)
            logger.info(f"Seeded {len(document_ids)} policy documents")
        except Exception as e:
            logger.error(f"Failed to seed policy documents: {e}")

    async def seed_audit_logs(self, count: int = 200):
        """Seed audit log data."""
        logger.info(f"Seeding {count} audit logs...")

        users = [f"analyst_{i:03d}" for i in range(1, 21)]
        actions = [
            "login", "logout", "transaction_view", "transaction_update",
            "fraud_case_create", "fraud_case_update", "policy_search",
            "report_generate", "user_create", "user_update"
        ]

        resource_types = ["transaction", "fraud_case", "policy", "user", "report"]

        for i in range(count):
            audit_data = {
                "user_id": random.choice(users),
                "action": random.choice(actions),
                "resource_type": random.choice(resource_types),
                "resource_id": f"res_{random.randint(1000, 9999)}",
                "request_body": {
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {
                        "page": random.randint(1, 10),
                        "limit": random.choice([10, 25, 50, 100]),
                        "filter": random.choice(["active", "inactive", "all"])
                    }
                },
                "response_body": {
                    "status": "success",
                    "count": random.randint(1, 100),
                    "data": ["item1", "item2", "item3"]
                },
                "status_code": random.choice([200, 201, 400, 401, 404, 500]),
                "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "user_agent": random.choice([
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
                ])
            }

            try:
                await self.db.log_audit_event(audit_data)
            except Exception as e:
                logger.error(f"Failed to seed audit log {i}: {e}")

        logger.info(f"Seeded {count} audit logs")

    async def seed_all_data(self):
        """Seed all types of data."""
        logger.info("Starting comprehensive data seeding...")

        try:
            await self.initialize()

            # Seed in order (transactions first, then fraud cases that reference them)
            await self.seed_transactions(1000)
            await self.seed_fraud_cases(50)
            await self.seed_policy_documents()
            await self.seed_audit_logs(200)

            logger.info("Data seeding completed successfully")

        except Exception as e:
            logger.error(f"Data seeding failed: {e}")
            raise

    async def generate_test_investigation(self) -> Dict[str, Any]:
        """Generate a test investigation scenario."""
        # Get some suspicious transactions
        transactions = await self.db.get_transaction_data(limit=5)

        # Enhance with simulated fraud indicators
        for tx in transactions:
            tx["analysis"] = {
                "velocity_anomaly": {
                    "score": round(random.uniform(0.6, 0.9), 2),
                    "description": "High transaction velocity detected",
                    "confidence": 0.85
                },
                "geo_anomaly": {
                    "score": round(random.uniform(0.7, 1.0), 2),
                    "description": "Unusual geographic pattern",
                    "confidence": 0.9
                },
                "merchant_risk": {
                    "score": round(random.uniform(0.5, 0.8), 2),
                    "description": "Medium risk merchant category",
                    "confidence": 0.7
                }
            }
            tx["overall_risk"] = round(random.uniform(0.6, 0.9), 2)
            tx["confidence"] = round(random.uniform(0.7, 0.95), 2)

        # Simulate policy search results
        policy_results = {
            "answer": "Based on AML Policy 2024, transactions above $10,000 require enhanced due diligence and must be reported within 24 hours.",
            "sources": [
                {
                    "document_id": "policy_001",
                    "title": "AML Policy 2024",
                    "policy_type": "AML",
                    "relevance_score": 0.89
                }
            ],
            "confidence": 0.85,
            "document_count": 3
        }

        # Simulate fraud reasoning
        fraud_reasoning = {
            "signals": [
                {
                    "signal_type": "velocity_anomaly",
                    "description": "5x normal transaction velocity",
                    "strength": 0.85,
                    "source": "transaction_analysis"
                },
                {
                    "signal_type": "geo_anomaly",
                    "description": "Transaction from high-risk jurisdiction",
                    "strength": 0.92,
                    "source": "transaction_analysis"
                }
            ],
            "pattern_matches": [
                {
                    "pattern_id": "cnp_fraud",
                    "pattern_name": "Card Not Present Fraud",
                    "match_score": 0.78,
                    "risk_level": "HIGH"
                }
            ],
            "overall_assessment": {
                "fraud_likelihood": 0.82,
                "impact_score": 0.65,
                "confidence": 0.88,
                "risk_level": "HIGH",
                "key_findings": [
                    "Multiple high-strength fraud signals",
                    "Matches known fraud pattern",
                    "Transaction amount exceeds normal patterns"
                ]
            }
        }

        return {
            "transactions": transactions,
            "policies": policy_results,
            "fraud_reasoning": fraud_reasoning,
            "investigation_id": f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }

    async def cleanup_test_data(self):
        """Clean up all test data."""
        logger.info("Cleaning up test data...")

        try:
            # Delete from vector store
            await self.vector_store.delete_collection()

            # Truncate database tables
            truncate_queries = [
                "TRUNCATE TABLE transactions CASCADE",
                "TRUNCATE TABLE fraud_cases CASCADE",
                "TRUNCATE TABLE audit_logs CASCADE"
            ]

            for query in truncate_queries:
                await self.db.execute_custom_query(query)

            logger.info("Test data cleanup completed")

        except Exception as e:
            logger.error(f"Test data cleanup failed: {e}")


async def main():
    """Main function for data seeding."""
    seeder = DataSeeder()

    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "seed":
            await seeder.seed_all_data()
        elif command == "cleanup":
            await seeder.cleanup_test_data()
        elif command == "test":
            test_scenario = await seeder.generate_test_investigation()
            print(json.dumps(test_scenario, indent=2, default=str))
        else:
            print("Usage: python seed_data.py [seed|cleanup|test]")
    else:
        # Default: seed all data
        await seeder.seed_all_data()


if __name__ == "__main__":
    asyncio.run(main())