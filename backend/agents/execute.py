"""Approval-verifying safe simulation executor with database idempotency."""
from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from backend.database.models import (Recommendation, Approval, ActionExecution, Customer, Membership, AuditLog)


class ExecuteAgent:
    def run(self, db, tenant_id: str, recommendation_id: str, reviewer_id: str,
            expected_customer_id: str | None = None) -> dict:
        recommendation = db.scalar(select(Recommendation).where(Recommendation.id == recommendation_id,
            Recommendation.tenant_id == tenant_id).with_for_update())
        if recommendation is None: raise HTTPException(404, "Recommendation not found")
        existing = db.scalar(select(ActionExecution).where(ActionExecution.recommendation_id == recommendation.id))
        if existing:
            return {"execution_id":existing.id,"status":existing.status,"simulated":existing.simulated,"idempotent_replay":True,"result":existing.result}
        if recommendation.status != "approved": raise HTTPException(409, "Only an approved recommendation can be executed")
        approval = db.scalar(select(Approval).where(Approval.recommendation_id == recommendation.id,
            Approval.tenant_id == tenant_id, Approval.decision == "approved"))
        if approval is None or approval.reviewer_id != reviewer_id:
            raise HTTPException(403, "An authorized human approval is required")
        customer = db.scalar(select(Customer).where(Customer.id == recommendation.customer_pk,
            Customer.tenant_id == tenant_id))
        if customer is None or (expected_customer_id and customer.customer_id != expected_customer_id):
            raise HTTPException(404, "Recommendation customer does not belong to this tenant")
        membership = db.scalar(select(Membership).where(Membership.tenant_id == tenant_id, Membership.user_id == reviewer_id))
        roles = {r.strip().upper() for r in membership.role.split(",")} if membership else set()
        if not roles.intersection({"OWNER", "ADMIN", "APPROVER"}): raise HTTPException(403, "Reviewer no longer has approval permission")
        result = {"message": "Simulated action execution recorded; no CRM, messaging, billing, or support system was contacted.",
                  "action_id": recommendation.action_id, "customer_id": customer.customer_id,
                  "customer_retention_outcome": "unknown; execution is not evidence of retention"}
        execution = ActionExecution(id=str(uuid4()), tenant_id=tenant_id, recommendation_id=recommendation.id,
            executed_by=reviewer_id, status="executed", simulated=True, result=result)
        recommendation.status = "executed"
        db.add(execution)
        db.add(AuditLog(id=str(uuid4()), tenant_id=tenant_id, user_id=reviewer_id,
            action="recommendation.simulated_execution", resource_type="recommendation", resource_id=recommendation.id,
            details={"execution_id":execution.id,"action_id":recommendation.action_id,"simulated":True}))
        try: db.commit()
        except IntegrityError:
            db.rollback()
            existing = db.scalar(select(ActionExecution).where(ActionExecution.recommendation_id == recommendation_id))
            if existing: return {"execution_id":existing.id,"status":existing.status,"simulated":existing.simulated,"idempotent_replay":True,"result":existing.result}
            raise
        return {"execution_id":execution.id,"status":execution.status,"simulated":True,"idempotent_replay":False,"result":result}
