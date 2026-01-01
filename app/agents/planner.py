import json
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
from enum import Enum

from app.models.openai_client import OpenAIClient
from app.models.deepseek_client import DeepSeekClient
from app.utils.logger import get_logger
from app.api.schemas import AgentType, AgentPlan


#author: Noble Eselase Vulley
#version: 1.0.0


logger = get_logger(__name__)


class DataSource(str, Enum):
    TRANSACTIONS = "transactions"
    POLICIES = "policies"
    FRAUD_CASES = "fraud_cases"
    THREAT_INTEL = "threat_intel"


@dataclass
class PlanStep:
    step_id: str
    description: str
    agent_type: AgentType
    data_sources: List[DataSource]
    dependencies: List[str]
    estimated_time: float


class PlannerAgent:
    def __init__(self):
        self.openai_client = OpenAIClient()
        self.deepseek_client = DeepSeekClient()
        self.system_prompt = """You are a Planner Agent for a financial fraud investigation system.

        Your task:
        1. Decompose the user request into investigation steps.
        2. Decide which agents must be invoked.
        3. Specify required data sources.

        Available agents:
        - TransactionDataAgent: Analyzes transaction data for anomalies
        - PolicyRAGAgent: Retrieves compliance and fraud policies
        - FraudReasoningAgent: Correlates evidence for fraud detection
        - ExplanationAgent: Generates human-readable explanations

        Rules:
        - Minimize unnecessary LLM calls.
        - Prefer structured data before documents.
        - Output a JSON plan only.

        Output format:
        {{
            "steps": [
                {{
                    "step_id": "step_1",
                    "description": "description",
                    "agent": "agent_type",
                    "data_sources": ["source1", "source2"],
                    "dependencies": [],
                    "estimated_time": 2.5
                }}
            ],
            "required_agents": ["agent1", "agent2"],
            "estimated_total_time": 10.0,
            "priority": "high|medium|low",
            "needs_transaction_data": true|false,
            "needs_policy_review": true|false
        }}

        User Query: {user_query}"""

    async def create_plan(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Create an execution plan for investigation."""
        try:
            logger.info(f"Creating plan for query: {user_query[:100]}...")

            # Use DeepSeek for planning (cost-effective)
            messages = [
                {"role": "system", "content": self.system_prompt.format(user_query=user_query)},
                {"role": "user", "content": user_query}
            ]

            if context:
                messages.append({
                    "role": "system",
                    "content": f"Additional context: {json.dumps(context)}"
                })

            response = await self.deepseek_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )

            # Parse JSON response
            plan_data = self._parse_json_response(response)

            # Validate and enrich plan
            validated_plan = self._validate_plan(plan_data, user_query)

            logger.info(f"Plan created: {len(validated_plan['steps'])} steps")
            return validated_plan

        except Exception as e:
            logger.error(f"Failed to create plan: {e}", exc_info=True)
            # Return fallback plan
            return self._create_fallback_plan(user_query)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Extract JSON from markdown code blocks
            import re

            # Look for JSON in code blocks
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Try to clean and parse again
            cleaned = self._clean_json_string(response)
            return json.loads(cleaned)

    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string for parsing."""
        import re

        # Remove non-JSON content
        lines = json_str.strip().split('\n')
        json_lines = []
        in_json = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{') or stripped.startswith('['):
                in_json = True
            if in_json:
                json_lines.append(line)
            if stripped.endswith('}') or stripped.endswith(']'):
                in_json = False

        cleaned = '\n'.join(json_lines)

        # Fix common JSON issues
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)

        return cleaned

    def _validate_plan(self, plan_data: Dict, user_query: str) -> Dict[str, Any]:
        """Validate and enrich plan data."""
        # Ensure required fields
        default_plan = {
            "steps": [],
            "required_agents": [],
            "estimated_total_time": 0.0,
            "priority": "medium",
            "needs_transaction_data": False,
            "needs_policy_review": False
        }

        # Merge with provided data
        validated = {**default_plan, **plan_data}

        # Analyze query for additional requirements
        query_lower = user_query.lower()

        if any(word in query_lower for word in ['transaction', 'txn', 'payment', 'purchase']):
            validated["needs_transaction_data"] = True

        if any(word in query_lower for word in ['policy', 'compliance', 'regulation', 'rule', 'aml', 'kyc']):
            validated["needs_policy_review"] = True

        # Calculate estimated time
        total_time = sum(step.get("estimated_time", 2.0) for step in validated["steps"])
        validated["estimated_total_time"] = max(total_time, 5.0)

        # Set priority based on keywords
        if any(word in query_lower for word in ['urgent', 'critical', 'emergency', 'immediate']):
            validated["priority"] = "high"
        elif any(word in query_lower for word in ['suspicious', 'fraud', 'anomaly']):
            validated["priority"] = "medium"
        else:
            validated["priority"] = "low"

        return validated

    def _create_fallback_plan(self, user_query: str) -> Dict[str, Any]:
        """Create fallback plan when LLM fails."""
        logger.warning("Creating fallback plan")

        return {
            "steps": [
                {
                    "step_id": "step_1",
                    "description": "Analyze transaction data for anomalies",
                    "agent": AgentType.TRANSACTION.value,
                    "data_sources": [DataSource.TRANSACTIONS.value],
                    "dependencies": [],
                    "estimated_time": 3.0
                },
                {
                    "step_id": "step_2",
                    "description": "Check relevant compliance policies",
                    "agent": AgentType.POLICY.value,
                    "data_sources": [DataSource.POLICIES.value],
                    "dependencies": ["step_1"],
                    "estimated_time": 2.5
                },
                {
                    "step_id": "step_3",
                    "description": "Correlate evidence for fraud reasoning",
                    "agent": AgentType.REASONING.value,
                    "data_sources": [DataSource.TRANSACTIONS.value, DataSource.POLICIES.value],
                    "dependencies": ["step_1", "step_2"],
                    "estimated_time": 4.0
                },
                {
                    "step_id": "step_4",
                    "description": "Generate investigation report",
                    "agent": AgentType.EXPLANATION.value,
                    "data_sources": [],
                    "dependencies": ["step_3"],
                    "estimated_time": 2.0
                }
            ],
            "required_agents": [
                AgentType.TRANSACTION.value,
                AgentType.POLICY.value,
                AgentType.REASONING.value,
                AgentType.EXPLANATION.value
            ],
            "estimated_total_time": 11.5,
            "priority": "medium",
            "needs_transaction_data": True,
            "needs_policy_review": True
        }

    async def optimize_plan(self, plan: Dict[str, Any], available_agents: List[str]) -> Dict[str, Any]:
        """Optimize plan based on available agents and resources."""
        # Remove steps for unavailable agents
        available_steps = [
            step for step in plan["steps"]
            if step["agent"] in available_agents
        ]

        # Update dependencies
        step_ids = {step["step_id"] for step in available_steps}
        for step in available_steps:
            step["dependencies"] = [
                dep for dep in step["dependencies"]
                if dep in step_ids
            ]

        plan["steps"] = available_steps
        plan["required_agents"] = list(set(
            step["agent"] for step in available_steps
        ))

        return plan