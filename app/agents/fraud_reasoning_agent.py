import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

from app.models.deepseek_client import DeepSeekClient
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)


class FraudType(str, Enum):
    CARD_NOT_PRESENT = "card_not_present"
    ACCOUNT_TAKEOVER = "account_takeover"
    MONEY_LAUNDERING = "money_laundering"
    IDENTITY_THEFT = "identity_theft"
    MERCHANT_COLLUSION = "merchant_collusion"
    FRIENDLY_FRAUD = "friendly_fraud"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    TRIANGULATION = "triangulation"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FraudSignal:
    """Individual fraud signal."""
    signal_type: str
    description: str
    strength: float  # 0.0 to 1.0
    evidence: List[str]
    source: str


@dataclass
class FraudPatternMatch:
    """Match with known fraud pattern."""
    pattern_id: str
    pattern_name: str
    match_score: float
    matched_signals: List[str]
    description: str
    risk_level: RiskLevel


@dataclass
class ReasoningStep:
    """Step in fraud reasoning process."""
    step_number: int
    description: str
    input_data: Dict[str, Any]
    analysis: Dict[str, Any]
    conclusion: str
    confidence: float


class FraudReasoningAgent:
    def __init__(self):
        self.deepseek_client = DeepSeekClient()

        self.system_prompt = """You are a Fraud Reasoning Agent.

        Goal:
        Correlate transaction anomalies with fraud patterns and policy violations.

        Steps:
        1. Identify suspicious signals from inputs
        2. Match signals to known fraud typologies
        3. Assess likelihood and impact
        4. Generate structured reasoning chain

        Rules:
        - Think step by step
        - Do NOT generate final user explanation
        - Output structured reasoning JSON
        - Consider both transaction data and policy constraints
        - Account for false positive indicators

        Available fraud patterns:
        - Card Not Present: Online transactions without physical card
        - Account Takeover: Unauthorized account access
        - Money Laundering: Layering transactions to obscure origin
        - Identity Theft: Using stolen identity information
        - Merchant Collusion: Merchant involved in fraudulent transactions
        - Friendly Fraud: Legitimate customer falsely claims fraud
        - Synthetic Identity: Fake identity created from real information
        - Triangulation: Using legitimate business as front

        Output format:
        {
            "signals": [
                {
                    "signal_type": "velocity_anomaly",
                    "description": "description",
                    "strength": 0.85,
                    "evidence": ["evidence1", "evidence2"],
                    "source": "transaction_analysis"
                }
            ],
            "pattern_matches": [
                {
                    "pattern_id": "cnp_fraud",
                    "pattern_name": "Card Not Present Fraud",
                    "match_score": 0.78,
                    "matched_signals": ["signal1", "signal2"],
                    "description": "Pattern match description",
                    "risk_level": "high"
                }
            ],
            "reasoning_chain": [
                {
                    "step_number": 1,
                    "description": "Step description",
                    "input_data": {...},
                    "analysis": {...},
                    "conclusion": "conclusion",
                    "confidence": 0.9
                }
            ],
            "overall_assessment": {
                "fraud_likelihood": 0.75,
                "impact_score": 0.6,
                "confidence": 0.85,
                "risk_level": "high",
                "key_findings": ["finding1", "finding2"],
                "false_positive_indicators": ["indicator1", "indicator2"]
            },
            "recommendations": {
                "immediate": ["action1", "action2"],
                "short_term": ["action3"],
                "long_term": ["action4"]
            }
        }

        Inputs:
        - Transaction signals
        - Policy excerpts
        - Historical fraud patterns
        """

        # Known fraud patterns database (simplified)
        self.fraud_patterns = self._load_fraud_patterns()

    async def analyze(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fraud reasoning analysis."""
        try:
            logger.info("Starting fraud reasoning analysis...")

            # Step 1: Extract and normalize inputs
            normalized_inputs = await self._normalize_inputs(agent_results)

            # Step 2: Identify fraud signals
            signals = await self._identify_signals(normalized_inputs)

            # Step 3: Match to fraud patterns
            pattern_matches = await self._match_patterns(signals, normalized_inputs)

            # Step 4: Generate reasoning chain using DeepSeek
            reasoning_output = await self._llm_reasoning(
                signals,
                pattern_matches,
                normalized_inputs
            )

            # Step 5: Calculate overall assessment
            assessment = await self._calculate_assessment(
                signals,
                pattern_matches,
                reasoning_output,
                normalized_inputs
            )

            # Step 6: Generate recommendations
            recommendations = await self._generate_recommendations(assessment)

            # Combine all results
            result = {
                "signals": [asdict(s) for s in signals],
                "pattern_matches": [asdict(p) for p in pattern_matches],
                "reasoning_chain": reasoning_output.get("reasoning_chain", []),
                "overall_assessment": assessment,
                "recommendations": recommendations,
                "metadata": {
                    "model_used": "deepseek",
                    "analysis_timestamp": asyncio.get_event_loop().time(),
                    "input_sources": list(agent_results.keys())
                }
            }

            logger.info(f"Fraud reasoning completed: risk_level={assessment['risk_level']}")
            return result

        except Exception as e:
            logger.error(f"Fraud reasoning failed: {e}", exc_info=True)
            return self._create_error_result(str(e))

    async def _normalize_inputs(self, agent_results: Dict) -> Dict[str, Any]:
        """Normalize and structure inputs from different agents."""
        normalized = {
            "transaction_data": [],
            "policy_data": [],
            "historical_patterns": [],
            "context": {}
        }

        # Extract transaction analysis
        for key, value in agent_results.items():
            if key.startswith("transaction_"):
                if isinstance(value, dict) and "analysis" in value:
                    normalized["transaction_data"].append(value)

            elif key == "policies":
                if isinstance(value, dict) and "sources" in value:
                    normalized["policy_data"] = value.get("sources", [])

            elif key == "historical_context":
                normalized["historical_patterns"] = value

        # Add any additional context
        if "context" in agent_results:
            normalized["context"] = agent_results["context"]

        return normalized

    async def _identify_signals(self, inputs: Dict) -> List[FraudSignal]:
        """Identify potential fraud signals from inputs."""
        signals = []

        # Analyze transaction data
        for tx_analysis in inputs["transaction_data"]:
            # Velocity signals
            velocity_anomaly = tx_analysis.get("analysis", {}).get("velocity_anomaly", {})
            if velocity_anomaly.get("score", 0) > 0.6:
                signals.append(FraudSignal(
                    signal_type="velocity_anomaly",
                    description=velocity_anomaly.get("description", "High transaction velocity"),
                    strength=velocity_anomaly.get("score", 0),
                    evidence=[f"Transaction {tx_analysis.get('transaction_id', 'unknown')}"],
                    source="transaction_analysis"
                ))

            # Geo signals
            geo_anomaly = tx_analysis.get("analysis", {}).get("geo_anomaly", {})
            if geo_anomaly.get("score", 0) > 0.7:
                signals.append(FraudSignal(
                    signal_type="geo_anomaly",
                    description=geo_anomaly.get("description", "Unusual geographic location"),
                    strength=geo_anomaly.get("score", 0),
                    evidence=[f"Cross-border transaction detected"],
                    source="transaction_analysis"
                ))

            # Merchant risk signals
            merchant_risk = tx_analysis.get("analysis", {}).get("merchant_risk", {})
            if merchant_risk.get("score", 0) > 0.8:
                signals.append(FraudSignal(
                    signal_type="high_risk_merchant",
                    description=merchant_risk.get("description", "High-risk merchant category"),
                    strength=merchant_risk.get("score", 0),
                    evidence=[f"Merchant category flagged as high-risk"],
                    source="transaction_analysis"
                ))

        # Policy violation signals
        for policy in inputs["policy_data"]:
            if isinstance(policy, dict) and policy.get("policy_type") == "AML":
                signals.append(FraudSignal(
                    signal_type="aml_policy_relevant",
                    description="AML policy context available",
                    strength=0.5,
                    evidence=[f"Policy: {policy.get('title', 'Unknown')}"],
                    source="policy_rag"
                ))

        # Historical pattern signals
        historical = inputs.get("historical_patterns", {})

        # Ensure we handle both dict and list cases
        if isinstance(historical, dict):
            if historical.get("fraud_history"):
                signals.append(FraudSignal(
                    signal_type="prior_fraud_history",
                    description="User has prior fraud incidents",
                    strength=0.7,
                    evidence=["Historical fraud cases found"],
                    source="historical_data"
                ))

        elif isinstance(historical, list):
            # If multiple historical entries, check each one
            for entry in historical:
                if isinstance(entry, dict) and entry.get("fraud_history"):
                    signals.append(FraudSignal(
                        signal_type="prior_fraud_history",
                        description="User has prior fraud incidents",
                        strength=0.7,
                        evidence=["Historical fraud cases found"],
                        source="historical_data"
                    ))

        return signals

    async def _match_patterns(self, signals: List[FraudSignal],
                              inputs: Dict) -> List[FraudPatternMatch]:
        """Match signals to known fraud patterns."""
        matches = []

        # Check each known pattern
        for pattern_id, pattern in self.fraud_patterns.items():
            match_score, matched_signals = self._calculate_pattern_match(
                pattern, signals, inputs
            )

            if match_score > 0.3:  # Threshold for inclusion
                matches.append(FraudPatternMatch(
                    pattern_id=pattern_id,
                    pattern_name=pattern["name"],
                    match_score=match_score,
                    matched_signals=matched_signals,
                    description=pattern["description"],
                    risk_level=self._determine_risk_level(match_score, pattern)
                ))

        # Sort by match score
        matches.sort(key=lambda x: x.match_score, reverse=True)

        return matches[:5]  # Return top 5 matches

    async def _llm_reasoning(self, signals: List[FraudSignal],
                             pattern_matches: List[FraudPatternMatch],
                             inputs: Dict) -> Dict[str, Any]:
        """Use DeepSeek for advanced reasoning."""
        # Prepare context for LLM
        context = {
            "signals": [asdict(s) for s in signals],
            "pattern_matches": [asdict(p) for p in pattern_matches],
            "transaction_data": inputs["transaction_data"],
            "policy_data": inputs["policy_data"],
            "historical_context": inputs.get("historical_patterns", {})
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(context, default=str)}
        ]

        response = await self.deepseek_client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, using rule-based reasoning")
            return self._rule_based_reasoning(signals, pattern_matches)

    async def _calculate_assessment(self, signals: List[FraudSignal],
                                    pattern_matches: List[FraudPatternMatch],
                                    reasoning_output: Dict, inputs: Dict) -> Dict[str, Any]:
        """Calculate overall fraud assessment."""
        # Calculate fraud likelihood
        signal_strengths = [s.strength for s in signals]
        pattern_scores = [p.match_score for p in pattern_matches]

        if signal_strengths:
            avg_signal_strength = sum(signal_strengths) / len(signal_strengths)
        else:
            avg_signal_strength = 0

        if pattern_scores:
            avg_pattern_score = sum(pattern_scores) / len(pattern_scores)
        else:
            avg_pattern_score = 0

        # Weighted combination
        fraud_likelihood = (avg_signal_strength * 0.4 + avg_pattern_score * 0.6)

        # Calculate impact score
        impact_score = self._calculate_impact(signals, pattern_matches)

        # Determine risk level
        risk_level = self._calculate_risk_level(fraud_likelihood, impact_score)

        # Extract key findings
        key_findings = []
        for signal in signals[:3]:  # Top 3 signals
            if signal.strength > 0.6:
                key_findings.append(signal.description)

        for pattern in pattern_matches[:2]:  # Top 2 patterns
            if pattern.match_score > 0.5:
                key_findings.append(f"Matches {pattern.pattern_name} pattern")

        # Identify false positive indicators
        false_positives = self._identify_false_positives(signals, inputs)

        return {
            "fraud_likelihood": float(fraud_likelihood),
            "impact_score": float(impact_score),
            "confidence": reasoning_output.get("confidence", 0.7),
            "risk_level": risk_level.value,
            "key_findings": key_findings,
            "false_positive_indicators": false_positives,
            "signal_count": len(signals),
            "pattern_match_count": len(pattern_matches)
        }

    async def _generate_recommendations(self, assessment: Dict) -> Dict[str, List[str]]:
        """Generate recommendations based on assessment."""
        recommendations = {
            "immediate": [],
            "short_term": [],
            "long_term": []
        }

        risk_level = assessment.get("risk_level", "low")
        fraud_likelihood = assessment.get("fraud_likelihood", 0)

        # Immediate actions
        if risk_level == "CRITICAL" or fraud_likelihood > 0.8:
            recommendations["immediate"].extend([
                "Block transaction immediately",
                "Freeze associated accounts",
                "Alert fraud investigation team",
                "Initiate customer contact procedure"
            ])
        elif risk_level == "HIGH" or fraud_likelihood > 0.6:
            recommendations["immediate"].extend([
                "Flag transaction for manual review",
                "Request additional authentication",
                "Monitor account for further suspicious activity"
            ])

        # Short-term actions
        if fraud_likelihood > 0.4:
            recommendations["short_term"].extend([
                "Update customer risk profile",
                "Review transaction patterns for similar cases",
                "Enhance monitoring rules based on findings"
            ])

        # Long-term actions
        if len(assessment.get("key_findings", [])) > 2:
            recommendations["long_term"].extend([
                "Review and update fraud detection models",
                "Consider policy changes based on patterns",
                "Train analysts on identified fraud typology"
            ])

        return recommendations

    def _calculate_pattern_match(self, pattern: Dict, signals: List[FraudSignal],
                                 inputs: Dict) -> Tuple[float, List[str]]:
        """Calculate match score between signals and a fraud pattern."""
        required_signals = pattern.get("required_signals", [])
        weighted_signals = pattern.get("weighted_signals", {})

        matched = []
        total_weight = 0
        matched_weight = 0

        # Check required signals
        for req_signal in required_signals:
            matching = [s for s in signals if req_signal in s.signal_type]
            if matching:
                matched.append(req_signal)
                matched_weight += 1
            total_weight += 1

        # Check weighted signals
        for signal_type, weight in weighted_signals.items():
            matching = [s for s in signals if signal_type in s.signal_type]
            if matching:
                matched.append(signal_type)
                matched_weight += weight
            total_weight += weight

        # Calculate match score
        if total_weight == 0:
            return 0.0, matched

        match_score = matched_weight / total_weight

        # Adjust based on transaction data
        tx_data = inputs.get("transaction_data", [])
        if tx_data:
            avg_amount = sum(tx.get("amount", 0) for tx in tx_data) / len(tx_data)
            if avg_amount > pattern.get("typical_amount_threshold", 1000):
                match_score *= 1.2  # Boost for high amounts

        return min(match_score, 1.0), matched

    def _determine_risk_level(self, match_score: float, pattern: Dict) -> RiskLevel:
        """Determine risk level based on match score and pattern."""
        base_risk = pattern.get("base_risk", "medium")

        if match_score > 0.8:
            return RiskLevel.CRITICAL
        elif match_score > 0.6:
            return RiskLevel.HIGH
        elif match_score > 0.4:
            if base_risk == "high":
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        elif match_score > 0.2:
            return RiskLevel.LOW
        return RiskLevel.LOW

    def _calculate_impact(self, signals: List[FraudSignal],
                          pattern_matches: List[FraudPatternMatch]) -> float:
        """Calculate potential impact score."""
        impact_factors = []

        # Signal-based impact
        high_strength_signals = [s for s in signals if s.strength > 0.7]
        impact_factors.append(min(len(high_strength_signals) / 3, 1.0))

        # Pattern-based impact
        high_risk_patterns = [p for p in pattern_matches if p.risk_level in ["HIGH", "CRITICAL"]]
        impact_factors.append(min(len(high_risk_patterns) / 2, 1.0))

        # Combined impact
        if impact_factors:
            return sum(impact_factors) / len(impact_factors)
        return 0.0

    def _calculate_risk_level(self, likelihood: float, impact: float) -> RiskLevel:
        """Calculate overall risk level."""
        risk_score = (likelihood * 0.6 + impact * 0.4)

        if risk_score > 0.8:
            return RiskLevel.CRITICAL
        elif risk_score > 0.6:
            return RiskLevel.HIGH
        elif risk_score > 0.4:
            return RiskLevel.MEDIUM
        elif risk_score > 0.2:
            return RiskLevel.LOW
        return RiskLevel.LOW

    def _identify_false_positives(self, signals: List[FraudSignal], inputs: Dict) -> List[str]:
        """Identify indicators that might suggest false positive."""
        false_positives = []

        # Check for legitimate explanations
        tx_data = inputs.get("transaction_data", [])
        for tx in tx_data:
            features = tx.get("features", {})

            # Travel indicator
            if features.get("geo_distance_km", 0) > 1000:
                # Check if this is typical travel pattern
                if "travel" in tx.get("merchant_category", "").lower():
                    false_positives.append("Possible travel-related transaction")

            # Large but legitimate purchase
            if tx.get("amount", 0) > 5000:
                merchant_cat = tx.get("merchant_category", "").lower()
                if any(cat in merchant_cat for cat in ["jewelry", "electronics", "furniture"]):
                    false_positives.append("High-value but typical purchase category")

        return false_positives

    def _rule_based_reasoning(self, signals: List[FraudSignal],
                              pattern_matches: List[FraudPatternMatch]) -> Dict[str, Any]:
        """Fallback rule-based reasoning when LLM fails."""
        reasoning_chain = []

        # Step 1: Signal analysis
        reasoning_chain.append(ReasoningStep(
            step_number=1,
            description="Analyze identified fraud signals",
            input_data={"signal_count": len(signals)},
            analysis={"high_strength_signals": [s for s in signals if s.strength > 0.7]},
            conclusion=f"Found {len(signals)} potential fraud signals",
            confidence=0.8
        ))

        # Step 2: Pattern matching
        reasoning_chain.append(ReasoningStep(
            step_number=2,
            description="Match signals to known fraud patterns",
            input_data={"pattern_count": len(pattern_matches)},
            analysis={"top_matches": [p.pattern_name for p in pattern_matches[:3]]},
            conclusion=f"Matched {len(pattern_matches)} fraud patterns",
            confidence=0.7
        ))

        # Step 3: Risk assessment
        reasoning_chain.append(ReasoningStep(
            step_number=3,
            description="Assess overall fraud risk",
            input_data={"signals": [s.signal_type for s in signals[:5]]},
            analysis={"risk_factors": ["Multiple high-strength signals", "Pattern matches"]},
            conclusion="Moderate to high fraud risk indicated",
            confidence=0.75
        ))

        return {
            "reasoning_chain": [asdict(step) for step in reasoning_chain],
            "confidence": 0.7
        }

    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            "error": True,
            "error_message": error,
            "signals": [],
            "pattern_matches": [],
            "reasoning_chain": [],
            "overall_assessment": {
                "fraud_likelihood": 0.0,
                "impact_score": 0.0,
                "confidence": 0.0,
                "risk_level": "LOW",
                "key_findings": ["Analysis failed due to error"],
                "false_positive_indicators": []
            },
            "recommendations": {
                "immediate": ["Manual review required", "Check system logs"],
                "short_term": [],
                "long_term": []
            }
        }

    def _load_fraud_patterns(self) -> Dict[str, Dict]:
        """Load known fraud patterns."""
        return {
            "cnp_fraud": {
                "name": "Card Not Present Fraud",
                "description": "Fraudulent online or phone transactions without physical card",
                "required_signals": ["velocity_anomaly", "geo_anomaly"],
                "weighted_signals": {"high_risk_merchant": 0.8, "behavioral_anomaly": 0.6},
                "typical_amount_threshold": 500,
                "base_risk": "high"
            },
            "account_takeover": {
                "name": "Account Takeover",
                "description": "Unauthorized access to legitimate user account",
                "required_signals": ["behavioral_anomaly", "prior_fraud_history"],
                "weighted_signals": {"velocity_anomaly": 0.7, "geo_anomaly": 0.9},
                "typical_amount_threshold": 1000,
                "base_risk": "critical"
            },
            "money_laundering": {
                "name": "Money Laundering",
                "description": "Structured transactions to obscure fund origins",
                "required_signals": ["velocity_anomaly"],
                "weighted_signals": {"aml_policy_relevant": 0.9, "structured_transactions": 0.8},
                "typical_amount_threshold": 10000,
                "base_risk": "high"
            },
            "friendly_fraud": {
                "name": "Friendly Fraud",
                "description": "Legitimate customer falsely claims transaction was fraudulent",
                "required_signals": ["behavioral_anomaly"],
                "weighted_signals": {"dispute_history": 0.9, "high_value": 0.6},
                "typical_amount_threshold": 200,
                "base_risk": "medium"
            }
        }