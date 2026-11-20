"""Typed outputs shared by the five AURA workflow agents."""
from pydantic import BaseModel, Field
from typing import Literal


class AnalysisStart(BaseModel):
    analysis_run_id: str
    tenant_id: str
    user_id: str
    customer_pk: str
    customer_id: str
    features: dict
    events: list[dict]
    tenant_settings: dict
    mcia_weights: dict | None = None


class RiskSignal(BaseModel):
    signal: str
    severity: str
    supporting_feature: str
    observed_value: str
    evidence_strength: str
    detected_at: str


class DetectResult(BaseModel):
    customer_id: str
    signals: list[RiskSignal]
    overall_severity: str
    summary: str


class DiagnosisItem(BaseModel):
    potential_reason: str
    supporting_features: list[str]
    relevant_events: list[str]
    explanation: str
    uncertainty: str
    confidence: float = Field(ge=0, le=1)
    causal_claim: bool = False


class DiagnoseResult(BaseModel):
    items: list[DiagnosisItem]
    summary: str
    provenance: str


class StrategyNarrativeItem(BaseModel):
    action_id: str
    rationale: str


class StrategyNarrative(BaseModel):
    items: list[StrategyNarrativeItem]


class ApprovalResume(BaseModel):
    decision: Literal["approved", "rejected"]
    recommendation_id: str
    approval_id: str
    reviewer_id: str
    note: str | None = None


class AgentState(dict):
    """Pydantic convenience marker; LangGraph state itself uses a typed mapping schema."""
