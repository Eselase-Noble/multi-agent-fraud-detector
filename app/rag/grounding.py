import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)


@dataclass
class GroundingEvidence:
    """Evidence supporting a grounded claim."""
    claim: str
    source_documents: List[Dict[str, Any]]
    confidence: float
    citations: List[str]
    contradictions: List[str]


@dataclass
class GroundingResult:
    """Result of grounding evaluation."""
    response: str
    evidence: List[GroundingEvidence]
    overall_confidence: float
    hallucinations: List[str]
    missing_sources: List[str]
    is_grounded: bool


class ResponseGrounder:
    """Production-grade response grounding and hallucination detection."""

    def __init__(self):
        self.min_confidence = 0.7
        self.citation_pattern = re.compile(r'\[(\d+)\]|\(([^)]+)\)')

        # Known hallucination patterns
        self.hallucination_patterns = [
            r'according to (?:our|the) (?:research|analysis|data)',
            r'studies (?:show|indicate) that',
            r'it is (?:widely|generally) (?:known|accepted)',
            r'experts (?:agree|say)',
            r'as (?:previously|earlier) mentioned',
            r'based on (?:the|this) information',
        ]

    async def ground_response(self,
                              response: str,
                              source_documents: List[Dict[str, Any]],
                              query: Optional[str] = None) -> GroundingResult:
        """Ground response in source documents and detect hallucinations."""
        logger.info("Grounding response in source documents...")

        try:
            # Extract claims from response
            claims = await self._extract_claims(response)

            # Match claims to source documents
            evidence_list = await self._match_claims_to_sources(
                claims, source_documents
            )

            # Detect hallucinations
            hallucinations = await self._detect_hallucinations(
                response, evidence_list, source_documents
            )

            # Identify missing sources
            missing_sources = await self._identify_missing_sources(
                evidence_list, source_documents
            )

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(evidence_list)

            # Check if response is grounded
            is_grounded = overall_confidence >= self.min_confidence and not hallucinations

            result = GroundingResult(
                response=response,
                evidence=evidence_list,
                overall_confidence=overall_confidence,
                hallucinations=hallucinations,
                missing_sources=missing_sources,
                is_grounded=is_grounded
            )

            logger.info(f"Grounding complete: confidence={overall_confidence:.2f}, grounded={is_grounded}")
            return result

        except Exception as e:
            logger.error(f"Grounding failed: {e}", exc_info=True)
            return self._create_error_result(response, str(e))

    async def _extract_claims(self, response: str) -> List[str]:
        """Extract individual claims from response."""
        # Split by sentences
        sentences = re.split(r'[.!?]+', response)

        # Clean and filter sentences
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum claim length
                # Remove citation markers
                clean_sentence = re.sub(self.citation_pattern, '', sentence).strip()
                if clean_sentence:
                    claims.append(clean_sentence)

        # Remove duplicates while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            if claim not in seen:
                seen.add(claim)
                unique_claims.append(claim)

        return unique_claims

    async def _match_claims_to_sources(self,
                                       claims: List[str],
                                       sources: List[Dict[str, Any]]) -> List[GroundingEvidence]:
        """Match each claim to relevant source documents."""
        evidence_list = []

        for claim in claims:
            matched_sources = []
            citations = []
            contradictions = []

            for source in sources:
                # Check if claim is supported by source
                support_score = await self._calculate_support_score(claim, source)

                if support_score > 0.7:
                    matched_sources.append({
                        "id": source.get("id", "unknown"),
                        "content": source.get("content", "")[:200],
                        "support_score": support_score,
                        "metadata": source.get("metadata", {})
                    })

                    # Extract citation
                    citation = self._extract_citation(claim, source)
                    if citation:
                        citations.append(citation)

                # Check for contradictions
                contradiction_score = await self._calculate_contradiction_score(claim, source)
                if contradiction_score > 0.8:
                    contradictions.append({
                        "source_id": source.get("id", "unknown"),
                        "contradiction": f"Claim contradicts source: {source.get('content', '')[:100]}...",
                        "score": contradiction_score
                    })

            # Calculate claim confidence
            confidence = self._calculate_claim_confidence(matched_sources, contradictions)

            evidence_list.append(GroundingEvidence(
                claim=claim,
                source_documents=matched_sources,
                confidence=confidence,
                citations=citations,
                contradictions=contradictions
            ))

        return evidence_list

    async def _calculate_support_score(self, claim: str, source: Dict[str, Any]) -> float:
        """Calculate how well source supports claim."""
        source_content = source.get("content", "").lower()
        claim_lower = claim.lower()

        # Check for exact matches
        if claim_lower in source_content:
            return 1.0

        # Check for semantic similarity (simplified)
        claim_words = set(claim_lower.split())
        source_words = set(source_content.split())

        # Calculate word overlap
        overlap = claim_words.intersection(source_words)
        if not claim_words:
            return 0.0

        overlap_ratio = len(overlap) / len(claim_words)

        # Boost score for key terms
        key_terms = self._extract_key_terms(claim)
        key_term_matches = sum(1 for term in key_terms if term in source_content)

        if key_terms:
            key_term_score = key_term_matches / len(key_terms)
            overlap_ratio = (overlap_ratio * 0.6) + (key_term_score * 0.4)

        return min(overlap_ratio, 1.0)

    async def _calculate_contradiction_score(self, claim: str, source: Dict[str, Any]) -> float:
        """Check if source contradicts claim."""
        # Simplified contradiction detection
        # In production, would use NLI models

        source_content = source.get("content", "").lower()
        claim_lower = claim.lower()

        # Check for negation patterns
        negation_indicators = [
            "not", "never", "no", "none", "nothing", "nowhere",
            "contrary to", "opposite of", "disproves", "refutes"
        ]

        # Check if source contains negation of claim terms
        claim_terms = claim_lower.split()
        for term in claim_terms:
            for negation in negation_indicators:
                if f"{negation} {term}" in source_content:
                    return 0.9

        return 0.0

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }

        words = text.lower().split()
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]

        return key_terms[:5]  # Limit to top 5

    def _extract_citation(self, claim: str, source: Dict[str, Any]) -> Optional[str]:
        """Extract citation information."""
        source_id = source.get("id", "unknown")
        metadata = source.get("metadata", {})

        # Try to get meaningful citation
        if "title" in metadata:
            return f"{metadata['title']} [{source_id}]"
        elif "author" in metadata:
            return f"{metadata['author']} [{source_id}]"
        else:
            return f"Source {source_id}"

    def _calculate_claim_confidence(self,
                                    matched_sources: List[Dict],
                                    contradictions: List[Dict]) -> float:
        """Calculate confidence for a claim."""
        if contradictions:
            return 0.0  # Contradictions kill confidence

        if not matched_sources:
            return 0.0  # No sources

        # Average support scores
        support_scores = [source.get("support_score", 0) for source in matched_sources]
        avg_support = sum(support_scores) / len(support_scores)

        # Apply source count multiplier
        source_multiplier = min(len(matched_sources) / 3, 1.0)  # Cap at 3 sources

        return avg_support * source_multiplier

    async def _detect_hallucinations(self,
                                     response: str,
                                     evidence_list: List[GroundingEvidence],
                                     sources: List[Dict[str, Any]]) -> List[str]:
        """Detect hallucinations in response."""
        hallucinations = []

        # Check for unsupported claims
        for evidence in evidence_list:
            if evidence.confidence < 0.3 and not evidence.contradictions:
                hallucinations.append(
                    f"Unsupported claim: {evidence.claim[:100]}..."
                )

        # Check for hallucination patterns
        for pattern in self.hallucination_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                hallucinations.append(
                    f"Hallucination pattern detected: '{match.group()}'"
                )

        # Check for fabricated numbers/dates
        fabricated = self._detect_fabricated_content(response, sources)
        hallucinations.extend(fabricated)

        return hallucinations

    def _detect_fabricated_content(self, response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Detect fabricated numbers, dates, or specific details."""
        fabricated = []

        # Extract numbers from response
        response_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)

        # Check if numbers appear in sources
        all_source_content = ' '.join([s.get('content', '') for s in sources])
        for number in response_numbers:
            if number not in all_source_content:
                # Check if it's a common number (year, percentage, etc.)
                if not self._is_common_number(number):
                    fabricated.append(f"Fabricated number: {number}")

        # Check for specific named entities not in sources
        # (Simplified - in production would use NER)

        return fabricated

    def _is_common_number(self, number: str) -> bool:
        """Check if number is common (not necessarily fabricated)."""
        try:
            num = float(number)

            # Common years (1900-2100)
            if 1900 <= num <= 2100 and num % 1 == 0:
                return True

            # Common percentages
            if 0 <= num <= 100:
                return True

            # Common amounts (round numbers)
            if num % 100 == 0 or num % 1000 == 0:
                return True

        except ValueError:
            pass

        return False

    async def _identify_missing_sources(self,
                                        evidence_list: List[GroundingEvidence],
                                        sources: List[Dict[str, Any]]) -> List[str]:
        """Identify claims that need additional sources."""
        missing = []

        for evidence in evidence_list:
            if evidence.confidence < 0.5 and not evidence.source_documents:
                missing.append(
                    f"Claim needs source: {evidence.claim[:100]}..."
                )

        return missing

    def _calculate_overall_confidence(self, evidence_list: List[GroundingEvidence]) -> float:
        """Calculate overall grounding confidence."""
        if not evidence_list:
            return 0.0

        # Weight by claim importance (longer claims are more important)
        weights = []
        confidences = []

        for evidence in evidence_list:
            claim_length = len(evidence.claim.split())
            weight = min(claim_length / 10, 1.0)  # Cap weight at 1.0

            weights.append(weight)
            confidences.append(evidence.confidence)

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(w * c for w, c in zip(weights, confidences))
        return weighted_sum / total_weight

    def _create_error_result(self, response: str, error: str) -> GroundingResult:
        """Create error result when grounding fails."""
        return GroundingResult(
            response=response,
            evidence=[],
            overall_confidence=0.0,
            hallucinations=[f"Grounding error: {error}"],
            missing_sources=[],
            is_grounded=False
        )

    async def add_citations(self,
                            response: str,
                            grounding_result: GroundingResult) -> str:
        """Add citations to response based on grounding evidence."""
        if not grounding_result.evidence:
            return response

        # Create citation mapping
        citation_map = {}
        citation_counter = 1

        for evidence in grounding_result.evidence:
            if evidence.citations and evidence.confidence > 0.5:
                for citation in evidence.citations:
                    if citation not in citation_map:
                        citation_map[citation] = citation_counter
                        citation_counter += 1

        # Add citations to response
        lines = response.split('\n')
        cited_lines = []

        for line in lines:
            cited_line = line

            # Find which evidence applies to this line
            for evidence in grounding_result.evidence:
                if evidence.claim in line and evidence.citations:
                    # Add citations
                    citation_numbers = []
                    for citation in evidence.citations:
                        if citation in citation_map:
                            citation_numbers.append(str(citation_map[citation]))

                    if citation_numbers:
                        citation_str = f"[{','.join(citation_numbers)}]"
                        if not line.endswith(citation_str):
                            cited_line = f"{line} {citation_str}"

            cited_lines.append(cited_line)

        # Add references section
        if citation_map:
            cited_lines.append("\n--- References ---")
            for citation, number in sorted(citation_map.items(), key=lambda x: x[1]):
                cited_lines.append(f"[{number}] {citation}")

        return '\n'.join(cited_lines)

    async def validate_response(self,
                                response: str,
                                sources: List[Dict[str, Any]],
                                min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """Comprehensive response validation."""
        min_conf = min_confidence or self.min_confidence

        grounding_result = await self.ground_response(response, sources)
        cited_response = await self.add_citations(response, grounding_result)

        return {
            "original_response": response,
            "cited_response": cited_response,
            "is_grounded": grounding_result.is_grounded,
            "overall_confidence": grounding_result.overall_confidence,
            "confidence_passed": grounding_result.overall_confidence >= min_conf,
            "hallucination_count": len(grounding_result.hallucinations),
            "missing_source_count": len(grounding_result.missing_sources),
            "evidence_count": len(grounding_result.evidence),
            "supported_claims": sum(1 for e in grounding_result.evidence if e.confidence > 0.5),
            "unsupported_claims": sum(1 for e in grounding_result.evidence if e.confidence <= 0.5),
            "details": {
                "hallucinations": grounding_result.hallucinations,
                "missing_sources": grounding_result.missing_sources,
                "evidence": [
                    {
                        "claim": e.claim[:100],
                        "confidence": e.confidence,
                        "source_count": len(e.source_documents)
                    }
                    for e in grounding_result.evidence
                ]
            }
        }