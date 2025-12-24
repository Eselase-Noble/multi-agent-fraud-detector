from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentType(str, Enum):
    PLANNER = "planner"
    TRANSACTION = "transaction"
    POLICY = "policy"
    REASONING = "reasoning"
    EXPLANATION = "explanation"


class InvestigationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TransactionSchema(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(..., description="Currency code")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category code")
    merchant_country: str = Field(..., description="Merchant country code")
    user_country: str = Field(..., description="User country code")
    transaction_time: datetime = Field(..., description="Transaction timestamp")
    status: str = Field(..., description="Transaction status")
    fraud_score: Optional[float] = Field(None, description="Fraud probability score")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features")


class PolicyDocument(BaseModel):
    policy_id: str = Field(..., description="Policy document identifier")
    title: str = Field(..., description="Policy title")
    content: str = Field(..., description="Policy content")
    policy_type: str = Field(..., description="Policy type (AML, KYC, etc.)")
    region: str = Field(..., description="Applicable region")
    version: str = Field(..., description="Policy version")
    effective_date: Optional[datetime] = Field(None, description="Effective date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FraudPattern(BaseModel):
    pattern_id: str = Field(..., description="Pattern identifier")
    name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="Pattern description")
    indicators: List[str] = Field(..., description="Key indicators")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    examples: Optional[List[Dict[str, Any]]] = Field(None, description="Example cases")


class AgentPlan(BaseModel):
    steps: List[Dict[str, Any]] = Field(..., description="Execution steps")
    required_agents: List[AgentType] = Field(..., description="Required agents")
    estimated_time: float = Field(..., description="Estimated execution time in seconds")
    data_sources: List[str] = Field(..., description="Required data sources")


class InvestigationResult(BaseModel):
    summary: str = Field(..., description="Investigation summary")
    key_evidence: List[str] = Field(..., description="Key evidence points")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")
    recommended_actions: List[str] = Field(..., description="Recommended actions")
    sources: List[Dict[str, Any]] = Field(..., description="Cited sources")
    fraud_patterns: Optional[List[FraudPattern]] = Field(None, description="Matched fraud patterns")
    transaction_analysis: Optional[Dict[str, Any]] = Field(None, description="Transaction analysis details")


class AgentResponse(BaseModel):
    agent_type: AgentType = Field(..., description="Agent type")
    result: Dict[str, Any] = Field(..., description="Agent result")
    execution_time: float = Field(..., description="Execution time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent confidence")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Agent sources")


class StreamingChunk(BaseModel):
    chunk_type: str = Field(..., description="Chunk type")
    agent: Optional[AgentType] = Field(None, description="Agent name")
    content: Union[str, Dict[str, Any]] = Field(..., description="Chunk content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Chunk timestamp")

    @validator('chunk_type')
    def validate_chunk_type(cls, v):
        valid_types = {'plan', 'agent_result', 'explanation', 'error', 'warning', 'progress'}
        if v not in valid_types:
            raise ValueError(f'chunk_type must be one of {valid_types}')
        return v


class MetricsResponse(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    rag_metrics: Dict[str, float] = Field(..., description="RAG evaluation metrics")
    agent_metrics: Dict[str, Dict[str, float]] = Field(..., description="Agent performance metrics")
    system_metrics: Dict[str, float] = Field(..., description="System resource metrics")


class AuditLogEntry(BaseModel):
    log_id: str = Field(..., description="Log entry identifier")
    user_id: str = Field(..., description="User identifier")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource identifier")
    timestamp: datetime = Field(..., description="Log timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")

    class Config:
        from_attributes = True