#routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Coroutine
import uuid
import json

from app.agents.planner import PlannerAgent
from app.agents.transaction_agent import TransactionAgent
from app.agents.policy_rag_agent import PolicyRAGAgent
from app.agents.fraud_reasoning_agent import FraudReasoningAgent
from app.agents.explanation_agent import ExplanationAgent
from app.security.pii_masking import PIIMasker
from app.security.audit_log import AuditLogger, AuditAction
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
audit_logger = AuditLogger()
pii_masker = PIIMasker()


# Request/Response Models
class InvestigationRequest(BaseModel):
    query: str = Field(..., description="The investigation query")
    user_id: str = Field(..., description="User ID of the analyst")
    priority: str = Field("medium", description="Investigation priority")
    transaction_ids: Optional[List[str]] = Field(None, description="Specific transaction IDs to investigate")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class InvestigationResponse(BaseModel):
    investigation_id: str = Field(..., description="Unique investigation ID")
    status: str = Field(..., description="Investigation status")
    result: Optional[Dict[str, Any]] = Field(None, description="Investigation result")
    confidence: float = Field(..., description="Overall confidence score")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Cited sources")
    execution_time: float = Field(..., description="Execution time in seconds")



class StreamChunk(BaseModel):
    type: str = Field(..., description="Chunk type: plan, agent_result, explanation")
    content: Dict[str, Any] = Field(..., description="Chunk content")
    agent: Optional[str] = Field(None, description="Agent name")


@router.post("/investigate", response_model=InvestigationResponse)
async def start_investigation(
        request: InvestigationRequest,
        background_tasks: BackgroundTasks,
        stream: bool = False
):
    """Start a new fraud investigation."""
    investigation_id = str(uuid.uuid4())

    # Log the request (with PII masking)
    masked_request = pii_masker.mask_dict(request.dict())
    await audit_logger.log(
        user_id=request.user_id,
        action=AuditAction.INVESTIGATION_START,
        resource_type="investigation",
        resource_id=investigation_id,
        request_body=masked_request
    )

    if stream:
        # Return streaming response
        return StreamingResponse(
            stream_investigation(investigation_id, request),
            media_type="text/event-stream"
        )
    else:
        # Process investigation synchronously
        result = await process_investigation(investigation_id, request)

        # Log the response
        await audit_logger.log(
            user_id=request.user_id,
            action=AuditAction.INVESTIGATION_COMPLETE,
            resource_type="investigation",
            resource_id=investigation_id,
            response_body=result.model_dump()
        )

        return result


@router.get("/investigations/{investigation_id}")
async def get_investigation(investigation_id: str):
    """Get investigation results by ID."""
    # In production, this would fetch from database
    return {
        "investigation_id": investigation_id,
        "status": "completed",
        "result": {"message": "Investigation data would be retrieved from database"}
    }


@router.get("/transactions/{transaction_id}/analysis")
async def analyze_transaction(transaction_id: str):
    """Analyze a specific transaction."""
    transaction_agent = TransactionAgent()
    result = await transaction_agent.analyze_transaction(transaction_id)

    return {
        "transaction_id": transaction_id,
        "analysis": result
    }


@router.post("/policies/search")
async def search_policies(query: str, limit: int = 5):
    """Search compliance policies."""
    policy_agent = PolicyRAGAgent()
    results = await policy_agent.search(query, limit)

    return {
        "query": query,
        "results": results
    }


@router.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics."""
    from app.evaluation.rag_metrics import RAGMetricsCollector

    collector = RAGMetricsCollector()
    metrics = await collector.get_aggregated_metrics()

    return metrics


# Helper functions
async def process_investigation(investigation_id: str, request: InvestigationRequest) -> InvestigationResponse:
    """Process an investigation request."""
    import time
    start_time = time.time()

    try:
        # Initialize agents
        planner = PlannerAgent()
        transaction_agent = TransactionAgent()
        policy_agent = PolicyRAGAgent()
        reasoning_agent = FraudReasoningAgent()
        explanation_agent = ExplanationAgent()

        # Step 1: Planner creates execution plan
        plan = await planner.create_plan(request.query, request.context)

        # Step 2: Execute plan based on agent requirements
        agent_results = {}

        if plan.get("needs_transaction_data"):
            transaction_ids = request.transaction_ids or await extract_transaction_ids(request.query)
            for tx_id in transaction_ids:
                analysis = await transaction_agent.analyze_transaction(tx_id)
                agent_results[f"transaction_{tx_id}"] = analysis

        if plan.get("needs_policy_review"):
            policy_query = await generate_policy_query(request.query, agent_results)
            policy_results = await policy_agent.search(policy_query)
            agent_results["policies"] = policy_results

        # Step 3: Fraud reasoning
        reasoning_result = await reasoning_agent.analyze(agent_results)
        agent_results["reasoning"] = reasoning_result

        # Step 4: Generate explanation
        explanation = await explanation_agent.explain(agent_results)

        # Calculate confidence
        confidence = calculate_confidence(reasoning_result, explanation)

        execution_time = time.time() - start_time

        return  InvestigationResponse(
            investigation_id=investigation_id,
            status="completed",
            result=explanation,
            confidence=confidence,
            sources=extract_sources(agent_results),
            execution_time=execution_time
        )

    except Exception as e:
        logger.error(f"Investigation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_investigation(investigation_id: str, request: InvestigationRequest):
    """Stream investigation progress."""
    import json
    import asyncio

    try:
        planner = PlannerAgent()

        # Stream planning phase
        plan = await planner.create_plan(request.query, request.context)
        yield f"data: {json.dumps({'type': 'plan', 'content': plan})}\n\n"

        # Execute agents based on plan
        if plan.get("needs_transaction_data"):
            transaction_agent = TransactionAgent()
            transaction_ids = request.transaction_ids or await extract_transaction_ids(request.query)

            for tx_id in transaction_ids:
                analysis = await transaction_agent.analyze_transaction(tx_id)
                yield f"data: {json.dumps({'type': 'agent_result', 'agent': 'transaction', 'content': {'transaction_id': tx_id, 'analysis': analysis}})}\n\n"

        if plan.get("needs_policy_review"):
            policy_agent = PolicyRAGAgent()
            policy_results = await policy_agent.search(request.query)
            yield f"data: {json.dumps({'type': 'agent_result', 'agent': 'policy', 'content': policy_results})}\n\n"

        # Final explanation
        explanation_agent = ExplanationAgent()
        explanation = await explanation_agent.explain({
            "query": request.query,
            "plan": plan
        })

        yield f"data: {json.dumps({'type': 'explanation', 'content': explanation})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': {'message': str(e)}})}\n\n"


async def extract_transaction_ids(query: str) -> List[str]:
    """Extract transaction IDs from query."""
    # Simple regex extraction - in production, use NLP
    import re
    return re.findall(r'transaction[_\s]*(?:id|#)?[_\s]*([A-Z0-9]{8,})', query, re.IGNORECASE)


async def generate_policy_query(user_query: str, agent_results: Dict) -> str:
    """Generate policy search query from user query and agent results."""
    # Extract key terms for policy search
    fraud_terms = ["fraud", "suspicious", "anomaly", "unusual", "flagged"]
    query_terms = user_query.lower().split()

    relevant_terms = [term for term in query_terms if len(term) > 3]
    relevant_terms.extend([term for term in fraud_terms if term in user_query.lower()])

    return " OR ".join(relevant_terms)


def calculate_confidence(reasoning_result: Dict, explanation: Dict) -> float:
    """Calculate overall confidence score."""
    if not reasoning_result or not explanation:
        return 0.0

    # Base confidence from reasoning
    reasoning_confidence = reasoning_result.get("confidence", 0.0)

    # Adjust based on explanation completeness
    explanation_score = min(1.0, len(explanation.get("key_evidence", [])) / 3)

    # Combined confidence
    return (reasoning_confidence * 0.7 + explanation_score * 0.3)


def extract_sources(agent_results: Dict) -> List[Dict[str, Any]]:
    """Extract and format sources from agent results."""
    sources = []

    for key, result in agent_results.items():
        if isinstance(result, dict) and "sources" in result:
            sources.extend(result["sources"])
        elif key.startswith("transaction_"):
            sources.append({
                "type": "transaction",
                "id": key.replace("transaction_", ""),
                "summary": f"Transaction analysis for {key}"
            })
        elif key == "policies":
            for policy in result.get("documents", []):
                sources.append({
                    "type": "policy",
                    "id": policy.get("policy_id"),
                    "title": policy.get("title"),
                    "section": policy.get("section")
                })

    return sources[:10]  # Limit to top 10 sources