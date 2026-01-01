import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

from app.models.openai_client import OpenAIClient
from app.utils.logger import get_logger
from app.utils.config import settings

#author: Noble Eselase Vulley
#version: 1.0.0

logger = get_logger(__name__)


@dataclass
class EvidenceItem:
    """Individual piece of evidence."""
    description: str
    source: str
    confidence: float
    relevance: str


@dataclass
class InvestigationSummary:
    """Structured investigation summary."""
    overview: str
    timeline: List[str]
    key_decisions: List[str]
    outcome: str

    def to_dict(self):
        return {
            "overview": self.overview,
            "timeline": self.timeline,
            "key_decisions": self.key_decisions,
            "outcome": self.outcome,
        }


class ExplanationAgent:
    def __init__(self):
        self.openai_client = OpenAIClient()

        self.system_prompt = """You are an Explanation Agent for fraud analysts.

        Task:
        - Produce a clear, concise explanation of fraud investigation findings
        - Reference evidence from provided sources
        - Avoid speculation and stay factual
        - Structure information for decision-making

        Must include:
        1. Executive Summary: Brief overview of findings
        2. Key Evidence: Bulleted list of supporting evidence
        3. Risk Assessment: Risk score and level explanation
        4. Fraud Pattern Analysis: Matched fraud patterns if any
        5. Confidence Level: Certainty of conclusions
        6. Recommended Actions: Prioritized action items
        7. Supporting Details: Additional context as needed

        Formatting rules:
        - Use clear section headers
        - Bold important terms
        - Include source citations where applicable
        - Keep technical terms to a minimum

        If confidence < 0.7:
        - Explicitly state the conclusion is uncertain
        - Highlight evidence gaps
        - Recommend additional verification

        Evidence and Findings:
        {aggregated_results}"""

    async def explain(self, investigation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation of investigation results."""
        try:
            logger.info("Generating investigation explanation...")

            # Extract and structure evidence
            evidence = await self._extract_evidence(investigation_results)

            # Generate summary using OpenAI
            explanation = await self._generate_explanation(
                investigation_results,
                evidence
            )

            # Add confidence-based warnings
            if explanation.get("confidence", 0) < settings.CONFIDENCE_THRESHOLD:
                explanation = self._add_uncertainty_warnings(explanation)

            # Format for different audiences
            formatted_explanation = await self._format_for_audience(
                explanation,
                audience="analyst"
            )

            # Add metadata and timestamps
            formatted_explanation.update({
                "explanation_timestamp": asyncio.get_event_loop().time(),
                "model_used": "openai",
                "explanation_version": "1.0"
            })

            logger.info("Explanation generation completed")
            return formatted_explanation

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}", exc_info=True)
            return self._create_error_explanation(str(e))

    async def _extract_evidence(self, results: Dict[str, Any]) -> List[EvidenceItem]:
        """Extract and structure evidence from investigation results."""
        evidence_items = []

        # Extract from transaction analysis
        for key, value in results.items():
            if key.startswith("transaction_"):
                analysis = value.get("analysis", {})

                # Velocity evidence
                velocity = analysis.get("velocity_anomaly", {})
                if velocity.get("score", 0) > 0.5:
                    evidence_items.append(EvidenceItem(
                        description=f"Transaction velocity anomaly: {velocity.get('description', 'High velocity')}",
                        source=f"Transaction {value.get('transaction_id', 'unknown')}",
                        confidence=velocity.get("confidence", 0.7),
                        relevance="High" if velocity.get("score", 0) > 0.7 else "Medium"
                    ))

                # Geo evidence
                geo = analysis.get("geo_anomaly", {})
                if geo.get("score", 0) > 0.6:
                    evidence_items.append(EvidenceItem(
                        description=f"Geographic anomaly: {geo.get('description', 'Unusual location')}",
                        source=f"Transaction {value.get('transaction_id', 'unknown')}",
                        confidence=geo.get("confidence", 0.7),
                        relevance="High" if geo.get("score", 0) > 0.7 else "Medium"
                    ))

        # Extract from fraud reasoning
        reasoning = results.get("fraud_reasoning", {})
        if reasoning:
            signals = reasoning.get("signals", [])
            for signal in signals[:5]:  # Limit to top 5 signals
                evidence_items.append(EvidenceItem(
                    description=f"Fraud signal: {signal.get('description', 'Unknown signal')}",
                    source="Fraud Reasoning Agent",
                    confidence=signal.get("strength", 0.5),
                    relevance="High" if signal.get("strength", 0) > 0.7 else "Medium"
                ))

            patterns = reasoning.get("pattern_matches", [])
            for pattern in patterns[:3]:  # Top 3 patterns
                evidence_items.append(EvidenceItem(
                    description=f"Matched fraud pattern: {pattern.get('pattern_name', 'Unknown pattern')} (score: {pattern.get('match_score', 0):.2f})",
                    source="Pattern Matching",
                    confidence=pattern.get("match_score", 0.5),
                    relevance="Critical" if pattern.get("risk_level") == "CRITICAL" else "High"
                ))

        # Extract from policy results
        policies = results.get("policies", {})
        if policies and "sources" in policies:
            for source in policies["sources"][:3]:  # Top 3 policy sources
                evidence_items.append(EvidenceItem(
                    description=f"Policy reference: {source.get('title', 'Unknown policy')}",
                    source=f"Policy: {source.get('policy_type', 'General')}",
                    confidence=source.get("relevance_score", 0.6),
                    relevance="Medium"
                ))

        # Sort by relevance and confidence
        evidence_items.sort(
            key=lambda x: (
                0 if x.relevance == "Critical" else
                1 if x.relevance == "High" else
                2 if x.relevance == "Medium" else 3,
                x.confidence
            )
        )

        return evidence_items

    async def _generate_explanation(self, results: Dict[str, Any],
                                    evidence: List[EvidenceItem]) -> Dict[str, Any]:
        """Generate explanation using OpenAI."""
        # Prepare context for LLM
        context = {
            "investigation_summary": self._create_investigation_summary(results).to_dict(),
            "key_evidence": [e.__dict__ for e in evidence],
            "fraud_assessment": results.get("fraud_reasoning", {}).get("overall_assessment", {}),
            "recommendations": results.get("fraud_reasoning", {}).get("recommendations", {}),
            "transaction_count": sum(1 for k in results.keys() if k.startswith("transaction_")),
            "confidence_scores": self._extract_confidence_scores(results)
        }

        messages = [
            {"role": "system", "content": self.system_prompt.format(
                aggregated_results=json.dumps(context, indent=2)
            )},
            {"role": "user", "content": "Generate a comprehensive investigation explanation."}
        ]

        response = await self.openai_client.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )

        # Parse and structure the response
        structured_response = self._structure_llm_response(response, context)

        return structured_response

    def _structure_llm_response(self, llm_response: str, context: Dict) -> Dict[str, Any]:
        """Structure the LLM response into organized sections."""
        # Parse sections from LLM response
        sections = self._parse_sections(llm_response)

        # Calculate overall confidence
        confidence_scores = context.get("confidence_scores", [])
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

        # Extract risk assessment
        fraud_assessment = context.get("fraud_assessment", {})

        return {
            "executive_summary": sections.get("executive_summary", llm_response[:500]),
            "key_evidence": self._format_evidence_list(context.get("key_evidence", [])),
            "risk_assessment": {
                "risk_score": fraud_assessment.get("fraud_likelihood", 0.0),
                "risk_level": fraud_assessment.get("risk_level", "UNKNOWN"),
                "confidence": overall_confidence,
                "key_factors": fraud_assessment.get("key_findings", [])
            },
            "fraud_pattern_analysis": self._extract_pattern_analysis(context),
            "confidence_level": {
                "score": overall_confidence,
                "interpretation": self._interpret_confidence(overall_confidence),
                "limitations": self._identify_limitations(context)
            },
            "recommended_actions": self._prioritize_actions(context.get("recommendations", {})),
            "detailed_analysis": sections.get("detailed_analysis", ""),
            "supporting_details": self._extract_supporting_details(context),
            "raw_llm_response": llm_response  # Keep for debugging
        }

    async def _format_for_audience(self, explanation: Dict[str, Any],
                                   audience: str = "analyst") -> Dict[str, Any]:
        """Format explanation for different audiences."""
        base_explanation = explanation.copy()

        if audience == "analyst":
            # Detailed version for analysts
            formatted = {
                "audience": "fraud_analyst",
                "technical_level": "detailed",
                "sections": [
                    {
                        "title": "Investigation Summary",
                        "content": base_explanation["executive_summary"],
                        "priority": "high"
                    },
                    {
                        "title": "Evidence Analysis",
                        "content": base_explanation["key_evidence"],
                        "priority": "high"
                    },
                    {
                        "title": "Risk Assessment",
                        "content": base_explanation["risk_assessment"],
                        "priority": "high"
                    },
                    {
                        "title": "Fraud Pattern Details",
                        "content": base_explanation["fraud_pattern_analysis"],
                        "priority": "medium"
                    },
                    {
                        "title": "Recommended Actions",
                        "content": base_explanation["recommended_actions"],
                        "priority": "high"
                    },
                    {
                        "title": "Confidence Assessment",
                        "content": base_explanation["confidence_level"],
                        "priority": "medium"
                    },
                    {
                        "title": "Technical Details",
                        "content": base_explanation.get("detailed_analysis", ""),
                        "priority": "low"
                    }
                ]
            }

        elif audience == "manager":
            # Concise version for managers
            formatted = {
                "audience": "management",
                "technical_level": "summary",
                "key_points": [
                    base_explanation["executive_summary"],
                    f"Risk Level: {base_explanation['risk_assessment']['risk_level']}",
                    f"Confidence: {base_explanation['confidence_level']['score']:.0%}",
                    f"Recommended Action: {base_explanation['recommended_actions'].get('immediate', ['Review'])[0]}"
                ],
                "action_required": base_explanation["risk_assessment"]["risk_level"] in ["HIGH", "CRITICAL"]
            }

        else:  # Default technical version
            formatted = base_explanation

        # Add metadata
        formatted["formatted_timestamp"] = asyncio.get_event_loop().time()
        formatted["original_explanation_id"] = id(base_explanation)

        return formatted

    def _create_investigation_summary(self, results: Dict) -> InvestigationSummary:
        """Create structured investigation summary."""
        transaction_count = sum(1 for k in results.keys() if k.startswith("transaction_"))

        reasoning = results.get("fraud_reasoning", {})
        assessment = reasoning.get("overall_assessment", {})

        return InvestigationSummary(
            overview=f"Analysis of {transaction_count} transactions identified potential fraud indicators.",
            timeline=[
                "Transaction data collected and analyzed",
                "Policy references retrieved",
                "Fraud pattern matching performed",
                "Risk assessment completed"
            ],
            key_decisions=[
                f"Risk level determined as {assessment.get('risk_level', 'UNKNOWN')}",
                f"Confidence score: {assessment.get('confidence', 0):.0%}"
            ],
            outcome=f"Investigation {'requires immediate action' if assessment.get('risk_level') in ['HIGH', 'CRITICAL'] else 'recommends monitoring'}"
        )

    def _extract_confidence_scores(self, results: Dict) -> List[float]:
        """Extract confidence scores from various sources."""
        confidence_scores = []

        # Transaction analysis confidence
        for key, value in results.items():
            if key.startswith("transaction_"):
                confidence_scores.append(value.get("confidence", 0.5))

        # Fraud reasoning confidence
        reasoning = results.get("fraud_reasoning", {})
        if reasoning:
            confidence_scores.append(reasoning.get("overall_assessment", {}).get("confidence", 0.5))

        # Policy retrieval confidence
        policies = results.get("policies", {})
        if policies:
            confidence_scores.append(policies.get("confidence", 0.6))

        return confidence_scores if confidence_scores else [0.5]

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Parse LLM response into sections."""
        sections = {}
        current_section = "executive_summary"
        current_content = []

        lines = text.split('\n')
        for line in lines:
            line = line.strip()

            # Detect section headers
            if line.lower().startswith(("executive summary", "summary:")):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = "executive_summary"
                current_content = []
            elif line.lower().startswith(("key evidence", "evidence:")):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = "key_evidence"
                current_content = []
            elif line.lower().startswith(("risk assessment", "risk:")):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = "risk_assessment"
                current_content = []
            elif line.lower().startswith(("recommended", "actions:", "next steps")):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = "recommended_actions"
                current_content = []
            elif line.lower().startswith(("detailed", "analysis:", "technical")):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = "detailed_analysis"
                current_content = []
            elif line:
                current_content.append(line)

        # Add final section
        if current_content:
            sections[current_section] = '\n'.join(current_content)

        return sections

    def _format_evidence_list(self, evidence_items: List[Dict]) -> List[Dict]:
        """Format evidence list for presentation."""
        formatted = []
        for item in evidence_items[:10]:  # Limit to top 10
            formatted.append({
                "description": item.get("description", "Unknown evidence"),
                "source": item.get("source", "Unknown"),
                "confidence": item.get("confidence", 0.5),
                "relevance": item.get("relevance", "Medium"),
                "impact": "High" if item.get("confidence", 0) > 0.7 else "Medium"
            })
        return formatted

    def _extract_pattern_analysis(self, context: Dict) -> Dict[str, Any]:
        """Extract fraud pattern analysis from context."""
        reasoning = context.get("fraud_assessment", {})

        return {
            "patterns_identified": reasoning.get("pattern_match_count", 0),
            "primary_pattern": reasoning.get("key_findings", [])[0] if reasoning.get("key_findings") else "None",
            "match_strength": reasoning.get("fraud_likelihood", 0),
            "pattern_details": context.get("fraud_reasoning", {}).get("pattern_matches", [])[:3]
        }

    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score for humans."""
        if confidence >= 0.9:
            return "Very High - Strong evidence supports conclusions"
        elif confidence >= 0.7:
            return "High - Good evidence supports conclusions"
        elif confidence >= 0.5:
            return "Moderate - Some evidence supports conclusions"
        elif confidence >= 0.3:
            return "Low - Limited evidence, conclusions uncertain"
        else:
            return "Very Low - Insufficient evidence for reliable conclusions"

    def _identify_limitations(self, context: Dict) -> List[str]:
        """Identify limitations in the analysis."""
        limitations = []

        # Check data completeness
        tx_count = context.get("transaction_count", 0)
        if tx_count == 0:
            limitations.append("No transaction data available for analysis")
        elif tx_count < 3:
            limitations.append("Limited transaction history for pattern analysis")

        # Check evidence strength
        evidence = context.get("key_evidence", [])
        strong_evidence = [e for e in evidence if e.get("confidence", 0) > 0.7]
        if len(strong_evidence) < 2:
            limitations.append("Limited strong evidence points")

        # Check confidence scores
        conf_scores = context.get("confidence_scores", [])
        if any(score < 0.5 for score in conf_scores):
            limitations.append("Some analysis components have low confidence")

        return limitations if limitations else ["No significant limitations identified"]

    def _prioritize_actions(self, recommendations: Dict) -> Dict[str, List[str]]:
        """Prioritize recommended actions."""
        prioritized = {
            "immediate": recommendations.get("immediate", []),
            "short_term": recommendations.get("short_term", []),
            "long_term": recommendations.get("long_term", []),
            "monitoring": [
                "Continue standard monitoring",
                "Document investigation findings"
            ]
        }

        # Ensure we always have at least some recommendations
        if not any(prioritized.values()):
            prioritized["monitoring"].append("Schedule follow-up review")

        return prioritized

    def _extract_supporting_details(self, context: Dict) -> Dict[str, Any]:
        """Extract supporting details from context."""
        return {
            "transaction_analysis_count": context.get("transaction_count", 0),
            "evidence_items_count": len(context.get("key_evidence", [])),
            "analysis_timestamp": asyncio.get_event_loop().time(),
            "methodology": "Multi-agent RAG system with transaction analysis, policy retrieval, and fraud pattern matching",
            "assumptions": [
                "Transaction data is accurate and complete",
                "Policy documents are current and applicable",
                "Fraud patterns are based on historical data"
            ]
        }

    def _add_uncertainty_warnings(self, explanation: Dict) -> Dict:
        """Add warnings when confidence is low."""
        warnings = [
            "⚠️ LOW CONFIDENCE: Conclusions should be verified with additional evidence",
            "⚠️ Consider manual review before taking action",
            "⚠️ Evidence gaps identified - recommend further investigation"
        ]

        explanation["warnings"] = warnings
        explanation["requires_verification"] = True

        # Update executive summary to reflect uncertainty
        if "executive_summary" in explanation:
            explanation["executive_summary"] = (
                    "⚠️ LOW CONFIDENCE - " + explanation["executive_summary"]
            )

        return explanation

    def _create_error_explanation(self, error: str) -> Dict[str, Any]:
        """Create error explanation structure."""
        return {
            "error": True,
            "error_message": error,
            "executive_summary": f"Explanation generation failed: {error}",
            "key_evidence": [],
            "risk_assessment": {
                "risk_score": 0.0,
                "risk_level": "UNKNOWN",
                "confidence": 0.0,
                "key_factors": ["System error prevented complete analysis"]
            },
            "confidence_level": {
                "score": 0.0,
                "interpretation": "Unable to assess confidence due to error",
                "limitations": ["System failure", "Missing data"]
            },
            "recommended_actions": {
                "immediate": ["Check system logs", "Manual investigation required"],
                "short_term": ["Review error and retry"],
                "long_term": ["Investigate system stability"]
            },
            "warnings": ["CRITICAL: Explanation generation failed - manual review required"]
        }