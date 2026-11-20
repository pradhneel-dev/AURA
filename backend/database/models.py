"""Tenant-aware persistence models for AURA application workflows."""
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow(): return datetime.now(timezone.utc)


class Base(DeclarativeBase): pass


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    industry: Mapped[str | None] = mapped_column(String(120))
    settings_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    auth_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    memberships: Mapped[list["Membership"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id", name="uq_membership_tenant_user"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    user: Mapped[User] = relationship(back_populates="memberships")


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (UniqueConstraint("tenant_id", "customer_id", name="uq_customer_tenant_external_id"),
                      Index("ix_customers_tenant_segment", "tenant_id", "segment"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    customer_id: Mapped[str] = mapped_column(String(160), nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    segment: Mapped[str | None] = mapped_column(String(100))
    dataset_version_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("tenant_id", "version", name="uq_dataset_tenant_version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_report: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class CustomerEvent(Base):
    __tablename__ = "customer_events"
    __table_args__ = (Index("ix_events_tenant_customer_timestamp", "tenant_id", "customer_pk", "timestamp"),)
    id: Mapped[str] = mapped_column(String(180), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    customer_pk: Mapped[str] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    severity: Mapped[str | None] = mapped_column(String(30))
    provenance: Mapped[str] = mapped_column(String(80), nullable=False)


class ModelArtifact(Base):
    __tablename__ = "model_metadata"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    active_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_tenant_customer_date", "tenant_id", "customer_pk", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    customer_pk: Mapped[str | None] = mapped_column(ForeignKey("customers.id", ondelete="SET NULL"))
    customer_id_snapshot: Mapped[str] = mapped_column(String(160), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_churn: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (Index("ix_analysis_tenant_created", "tenant_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    customer_pk: Mapped[str] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentExecution(Base):
    __tablename__ = "agent_executions"
    __table_args__ = (Index("ix_agent_exec_tenant_run", "tenant_id", "analysis_run_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    input_json: Mapped[dict | None] = mapped_column(JSON)
    output_json: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (UniqueConstraint("tenant_id", "idempotency_key", name="uq_recommendation_idempotency"),
                      Index("ix_recommendations_tenant_status", "tenant_id", "status"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    customer_pk: Mapped[str] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[str | None] = mapped_column(ForeignKey("analysis_runs.id", ondelete="SET NULL"))
    action_id: Mapped[str] = mapped_column(String(100), nullable=False)
    action_name: Mapped[str] = mapped_column(String(200), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    mcia: Mapped[dict] = mapped_column(JSON, nullable=False)
    eligibility: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(160), nullable=False)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Approval(Base):
    __tablename__ = "approvals"
    __table_args__ = (UniqueConstraint("recommendation_id", name="uq_approval_recommendation"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    recommendation_id: Mapped[str] = mapped_column(ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ActionExecution(Base):
    __tablename__ = "action_executions"
    __table_args__ = (UniqueConstraint("recommendation_id", name="uq_execution_once"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    recommendation_id: Mapped[str] = mapped_column(ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False)
    executed_by: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    simulated: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    result: Mapped[dict] = mapped_column(JSON, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class OutcomeFeedback(Base):
    __tablename__ = "outcome_feedback"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    recommendation_id: Mapped[str] = mapped_column(ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False, index=True)
    customer_pk: Mapped[str] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    outcome_status: Mapped[str] = mapped_column(String(40), nullable=False)
    customer_response: Mapped[str | None] = mapped_column(String(100))
    notes: Mapped[str | None] = mapped_column(Text)
    follow_up_status: Mapped[str | None] = mapped_column(String(100))
    recorded_by: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    feedback_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_tenant_time", "tenant_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(180))
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class LLMUsage(Base):
    __tablename__ = "llm_usage"
    __table_args__ = (Index("ix_llm_usage_tenant_created", "tenant_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[str | None] = mapped_column(ForeignKey("analysis_runs.id", ondelete="SET NULL"))
    agent_name: Mapped[str] = mapped_column(String(80), nullable=False)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
