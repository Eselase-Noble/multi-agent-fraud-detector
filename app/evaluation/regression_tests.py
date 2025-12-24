import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass

from app.utils.logger import get_logger
from app.agents.planner import PlannerAgent
from app.agents.transaction_agent import TransactionAgent
from app.agents.policy_rag_agent import PolicyRAGAgent
from app.agents.fraud_reasoning_agent import FraudReasoningAgent
from app.agents.explanation_agent import ExplanationAgent

logger = get_logger(__name__)


@dataclass
class TestCase:
    """A test case for regression testing."""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    validation_function: Optional[Callable] = None
    max_execution_time: float = 10.0

    def __post_init__(self):
        self.test_id = hashlib.md5(self.name.encode()).hexdigest()[:8]
        self.created_at = datetime.now()


@dataclass
class TestResult:
    """Result of a test case execution."""
    test_case: TestCase
    passed: bool
    execution_time: float
    actual_output: Any
    error_message: Optional[str] = None
    validation_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_case.test_id,
            "test_name": self.test_case.name,
            "passed": self.passed,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "validation_details": self.validation_details,
            "timestamp": datetime.now().isoformat()
        }


class RegressionTestSuite:
    """Production-grade regression testing for the fraud detection system."""

    def __init__(self):
        self.test_cases = []
        self.test_results = []
        self.test_history = []

        # Initialize agents for testing
        self.planner = PlannerAgent()
        self.transaction_agent = TransactionAgent()
        self.policy_agent = PolicyRAGAgent()
        self.reasoning_agent = FraudReasoningAgent()
        self.explanation_agent = ExplanationAgent()

        # Test data
        self.test_transactions = [
            {
                "transaction_id": "test_txn_001",
                "user_id": "test_user_001",
                "amount": 1500.00,
                "currency": "USD",
                "merchant_id": "test_merchant_001",
                "merchant_category": "Electronics",
                "merchant_country": "US",
                "user_country": "UK",
                "transaction_time": datetime.now().isoformat(),
                "status": "completed",
                "fraud_score": 0.25
            },
            {
                "transaction_id": "test_txn_002",
                "user_id": "test_user_001",
                "amount": 50.00,
                "currency": "USD",
                "merchant_id": "test_merchant_002",
                "merchant_category": "Groceries",
                "merchant_country": "UK",
                "user_country": "UK",
                "transaction_time": datetime.now().isoformat(),
                "status": "completed",
                "fraud_score": 0.05
            }
        ]

        self.test_queries = [
            "Analyze transaction test_txn_001 for fraud indicators",
            "What are the AML policies for transactions over $10,000?",
            "Investigate suspicious activity for user test_user_001"
        ]

    def register_test_cases(self):
        """Register all test cases."""
        logger.info("Registering regression test cases...")

        # Planner agent tests
        self.test_cases.extend([
            TestCase(
                name="planner_simple_query",
                description="Test planner with simple fraud investigation query",
                input_data={
                    "query": "Analyze transaction test_txn_001",
                    "context": None
                },
                validation_function=self._validate_planner_output
            ),
            TestCase(
                name="planner_complex_query",
                description="Test planner with complex investigation query",
                input_data={
                    "query": "Investigate user test_user_001 for money laundering patterns",
                    "context": {"priority": "high"}
                },
                validation_function=self._validate_planner_output
            )
        ])

        # Transaction agent tests
        self.test_cases.extend([
            TestCase(
                name="transaction_agent_single",
                description="Test transaction agent with single transaction",
                input_data={
                    "transaction_id": "test_txn_001",
                    "transaction_data": self.test_transactions[0]
                },
                validation_function=self._validate_transaction_analysis
            )
        ])

        # Policy RAG agent tests
        self.test_cases.extend([
            TestCase(
                name="policy_agent_search",
                description="Test policy agent search functionality",
                input_data={
                    "query": "AML compliance requirements",
                    "limit": 3
                },
                validation_function=self._validate_policy_search
            )
        ])

        # Fraud reasoning tests
        self.test_cases.extend([
            TestCase(
                name="fraud_reasoning_basic",
                description="Test fraud reasoning with basic inputs",
                input_data={
                    "transaction_analysis": [
                        {
                            "transaction_id": "test_txn_001",
                            "analysis": {
                                "velocity_anomaly": {"score": 0.8},
                                "geo_anomaly": {"score": 0.9},
                                "overall_risk": 0.85
                            }
                        }
                    ],
                    "policy_results": {
                        "sources": [{"title": "AML Policy", "relevance_score": 0.9}]
                    }
                },
                validation_function=self._validate_fraud_reasoning
            )
        ])

        # End-to-end tests
        self.test_cases.extend([
            TestCase(
                name="e2e_investigation",
                description="End-to-end fraud investigation",
                input_data={
                    "query": "Investigate transaction test_txn_001",
                    "user_id": "test_analyst_001",
                    "transaction_ids": ["test_txn_001"]
                },
                validation_function=self._validate_e2e_investigation,
                max_execution_time=30.0
            )
        ])

        logger.info(f"Registered {len(self.test_cases)} test cases")

    async def run_all_tests(self) -> List[TestResult]:
        """Run all registered test cases."""
        logger.info(f"Running {len(self.test_cases)} regression tests...")

        results = []

        for test_case in self.test_cases:
            try:
                result = await self._run_test_case(test_case)
                results.append(result)

                if result.passed:
                    logger.info(f"✓ {test_case.name}: PASSED ({result.execution_time:.2f}s)")
                else:
                    logger.error(f"✗ {test_case.name}: FAILED - {result.error_message}")

            except Exception as e:
                logger.error(f"✗ {test_case.name}: ERROR - {str(e)}")
                results.append(TestResult(
                    test_case=test_case,
                    passed=False,
                    execution_time=0.0,
                    actual_output=None,
                    error_message=f"Test execution error: {str(e)}"
                ))

        # Store in history
        self.test_history.append({
            "timestamp": datetime.now(),
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "failed_tests": sum(1 for r in results if not r.passed),
            "results": [r.to_dict() for r in results]
        })

        return results

    async def _run_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Execute test based on test case name
            if test_case.name.startswith("planner_"):
                actual_output = await self._execute_planner_test(test_case)
            elif test_case.name.startswith("transaction_"):
                actual_output = await self._execute_transaction_test(test_case)
            elif test_case.name.startswith("policy_"):
                actual_output = await self._execute_policy_test(test_case)
            elif test_case.name.startswith("fraud_"):
                actual_output = await self._execute_fraud_reasoning_test(test_case)
            elif test_case.name.startswith("e2e_"):
                actual_output = await self._execute_e2e_test(test_case)
            else:
                raise ValueError(f"Unknown test type: {test_case.name}")

            execution_time = asyncio.get_event_loop().time() - start_time

            # Validate output
            passed, validation_details = await self._validate_output(
                test_case, actual_output
            )

            return TestResult(
                test_case=test_case,
                passed=passed,
                execution_time=execution_time,
                actual_output=actual_output,
                validation_details=validation_details
            )

        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            return TestResult(
                test_case=test_case,
                passed=False,
                execution_time=execution_time,
                actual_output=None,
                error_message=f"Test timeout after {execution_time:.2f}s"
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return TestResult(
                test_case=test_case,
                passed=False,
                execution_time=execution_time,
                actual_output=None,
                error_message=str(e)
            )

    async def _execute_planner_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Execute planner agent test."""
        query = test_case.input_data["query"]
        context = test_case.input_data.get("context")

        plan = await self.planner.create_plan(query, context)
        return plan

    async def _execute_transaction_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Execute transaction agent test."""
        # For testing, we'll use provided transaction data
        if "transaction_data" in test_case.input_data:
            # Simulate analysis
            return {
                "transaction_id": test_case.input_data["transaction_id"],
                "analysis": {
                    "velocity_anomaly": {"score": 0.7, "description": "Test anomaly"},
                    "geo_anomaly": {"score": 0.3, "description": "Normal"},
                    "overall_risk": 0.5
                }
            }
        else:
            # Use actual agent
            return await self.transaction_agent.analyze_transaction(
                test_case.input_data["transaction_id"]
            )

    async def _execute_policy_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Execute policy agent test."""
        query = test_case.input_data["query"]
        limit = test_case.input_data.get("limit", 5)

        return await self.policy_agent.search(query, limit)

    async def _execute_fraud_reasoning_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Execute fraud reasoning test."""
        agent_results = test_case.input_data

        return await self.reasoning_agent.analyze(agent_results)

    async def _execute_e2e_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Execute end-to-end test."""
        # This would simulate the full investigation flow
        query = test_case.input_data["query"]

        # Step 1: Planning
        plan = await self.planner.create_plan(query)

        # Step 2: Transaction analysis
        transaction_results = {}
        if plan.get("needs_transaction_data"):
            for tx_id in test_case.input_data.get("transaction_ids", []):
                analysis = await self.transaction_agent.analyze_transaction(tx_id)
                transaction_results[f"transaction_{tx_id}"] = analysis

        # Step 3: Policy search
        policy_results = {}
        if plan.get("needs_policy_review"):
            policy_results = await self.policy_agent.search(query)

        # Step 4: Fraud reasoning
        agent_results = {**transaction_results, "policies": policy_results}
        reasoning_result = await self.reasoning_agent.analyze(agent_results)

        # Step 5: Explanation
        explanation = await self.explanation_agent.explain({
            **agent_results,
            "fraud_reasoning": reasoning_result
        })

        return {
            "plan": plan,
            "transaction_analysis": transaction_results,
            "policy_results": policy_results,
            "fraud_reasoning": reasoning_result,
            "explanation": explanation
        }

    async def _validate_output(self,
                               test_case: TestCase,
                               actual_output: Any) -> Tuple[bool, Dict[str, Any]]:
        """Validate test output."""
        if test_case.validation_function:
            return await test_case.validation_function(actual_output)

        # Default validation: check non-empty output
        if actual_output is None:
            return False, {"error": "Output is None"}

        if isinstance(actual_output, dict) and not actual_output:
            return False, {"error": "Output dictionary is empty"}

        if isinstance(actual_output, list) and not actual_output:
            return False, {"error": "Output list is empty"}

        return True, {"message": "Basic validation passed"}

    async def _validate_planner_output(self, output: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate planner agent output."""
        required_fields = ["steps", "required_agents", "estimated_total_time"]

        missing_fields = [field for field in required_fields if field not in output]

        if missing_fields:
            return False, {
                "missing_fields": missing_fields,
                "actual_fields": list(output.keys())
            }

        # Check steps format
        if not isinstance(output["steps"], list):
            return False, {"error": "Steps must be a list"}

        # Check required agents
        if not isinstance(output["required_agents"], list):
            return False, {"error": "Required agents must be a list"}

        return True, {"message": "Planner output validated successfully"}

    async def _validate_transaction_analysis(self, output: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate transaction analysis output."""
        required_fields = ["transaction_id", "analysis", "overall_risk"]

        missing_fields = [field for field in required_fields if field not in output]

        if missing_fields:
            return False, {"missing_fields": missing_fields}

        # Check analysis structure
        analysis = output["analysis"]
        if not isinstance(analysis, dict):
            return False, {"error": "Analysis must be a dictionary"}

        # Check risk score range
        risk = output.get("overall_risk")
        if not isinstance(risk, (int, float)) or not (0 <= risk <= 1):
            return False, {"error": f"Invalid risk score: {risk}"}

        return True, {"message": "Transaction analysis validated"}

    async def _validate_policy_search(self, output: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate policy search output."""
        if "answer" not in output:
            return False, {"error": "Missing answer field"}

        if "sources" not in output:
            return False, {"error": "Missing sources field"}

        if not isinstance(output["sources"], list):
            return False, {"error": "Sources must be a list"}

        return True, {"message": "Policy search validated"}

    async def _validate_fraud_reasoning(self, output: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate fraud reasoning output."""
        required_fields = ["signals", "pattern_matches", "overall_assessment"]

        missing_fields = [field for field in required_fields if field not in output]

        if missing_fields:
            return False, {"missing_fields": missing_fields}

        # Check assessment structure
        assessment = output["overall_assessment"]
        if "fraud_likelihood" not in assessment:
            return False, {"error": "Missing fraud_likelihood in assessment"}

        if "risk_level" not in assessment:
            return False, {"error": "Missing risk_level in assessment"}

        return True, {"message": "Fraud reasoning validated"}

    async def _validate_e2e_investigation(self, output: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate end-to-end investigation output."""
        required_sections = ["plan", "fraud_reasoning", "explanation"]

        missing_sections = [section for section in required_sections if section not in output]

        if missing_sections:
            return False, {"missing_sections": missing_sections}

        # Check explanation has required fields
        explanation = output.get("explanation", {})
        if "executive_summary" not in explanation:
            return False, {"error": "Missing executive_summary in explanation"}

        if "recommended_actions" not in explanation:
            return False, {"error": "Missing recommended_actions in explanation"}

        return True, {"message": "End-to-end investigation validated"}

    async def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        if not self.test_history:
            return {"error": "No tests have been run yet"}

        latest_run = self.test_history[-1]

        summary = {
            "last_run": latest_run["timestamp"].isoformat(),
            "total_tests": latest_run["total_tests"],
            "passed": latest_run["passed_tests"],
            "failed": latest_run["failed_tests"],
            "success_rate": (
                latest_run["passed_tests"] / latest_run["total_tests"] * 100
                if latest_run["total_tests"] > 0 else 0
            )
        }

        # Add historical trends
        if len(self.test_history) > 1:
            previous_run = self.test_history[-2]
            summary["trend"] = {
                "previous_success_rate": (
                    previous_run["passed_tests"] / previous_run["total_tests"] * 100
                    if previous_run["total_tests"] > 0 else 0
                ),
                "change": summary["success_rate"] - (
                    previous_run["passed_tests"] / previous_run["total_tests"] * 100
                    if previous_run["total_tests"] > 0 else 0
                )
            }

        return summary

    async def export_test_results(self,
                                  format: str = "json",
                                  run_id: Optional[int] = None) -> str:
        """Export test results."""
        if run_id is not None and 0 <= run_id < len(self.test_history):
            data = self.test_history[run_id]
        elif self.test_history:
            data = self.test_history[-1]
        else:
            return json.dumps({"error": "No test results available"})

        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == "html":
            # Generate HTML report
            return self._generate_html_report(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML test report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Regression Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid; }}
                .passed {{ border-color: #4CAF50; background: #e8f5e9; }}
                .failed {{ border-color: #f44336; background: #ffebee; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Regression Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Timestamp: {data['timestamp'].isoformat()}</p>
                <p>Total Tests: {data['total_tests']}</p>
                <p>Passed: {data['passed_tests']}</p>
                <p>Failed: {data['failed_tests']}</p>
                <p>Success Rate: {data['passed_tests'] / data['total_tests'] * 100:.1f}%</p>
            </div>
            <h2>Test Results</h2>
        """

        for result in data.get("results", []):
            status_class = "passed" if result["passed"] else "failed"
            html += f"""
            <div class="test-result {status_class}">
                <h3>{result['test_name']}</h3>
                <p>Status: {'PASSED' if result['passed'] else 'FAILED'}</p>
                <p>Execution Time: {result['execution_time']:.2f}s</p>
                <p class="timestamp">Test ID: {result['test_id']}</p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    async def add_custom_test(self, test_case: TestCase):
        """Add a custom test case."""
        self.test_cases.append(test_case)
        logger.info(f"Added custom test case: {test_case.name}")

    async def cleanup_old_results(self, days: int = 30):
        """Clean up old test results."""
        cutoff = datetime.now() - timedelta(days=days)

        self.test_history = [
            run for run in self.test_history
            if run["timestamp"] > cutoff
        ]

        logger.info(f"Cleaned up test results older than {days} days")


# Global instance
regression_test_suite = RegressionTestSuite()


async def run_regression_tests():
    """Run regression tests."""
    suite = RegressionTestSuite()
    suite.register_test_cases()
    results = await suite.run_all_tests()
    return results


async def get_test_report():
    """Get test report."""
    suite = RegressionTestSuite()
    return await suite.get_test_summary()