"""Durable Detect→Diagnose→Predict→Strategy→Approval→Execute LangGraph workflow."""
from pathlib import Path
import logging
import json
import sqlite3
import threading
from contextvars import ContextVar
from typing import Any
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langgraph.errors import GraphInterrupt
from pydantic import ValidationError
from sqlalchemy import select, func

from backend.agents.detect import run_detect
from backend.agents.diagnose import run_diagnose
from backend.agents.predict import PredictAgent
from backend.agents.strategy import run_strategy
from backend.agents.execute import ExecuteAgent
from backend.agents.schemas import AnalysisStart, ApprovalResume
from backend.database.models import (AgentExecution, AnalysisRun, Recommendation, LLMUsage, AuditLog, Prediction)
from backend.database.session import SessionLocal
from backend.llm.gateway import LLMGateway
from ml.inference.predict import ROOT

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    analysis_run_id: str
    tenant_id: str
    user_id: str
    customer_pk: str
    customer_id: str
    features: dict[str, Any]
    events: list[dict]
    tenant_settings: dict
    mcia_weights: dict | None
    detect: dict
    diagnose: dict
    prediction: dict
    strategy: list[dict]
    recommendation_ids: list[str]
    reviewed_recommendation_ids: list[str]
    rejected_recommendation_ids: list[str]
    approval: dict
    execution: dict
    status: str
    errors: list[dict]


def _log_agent(state, name, status, input_json=None, output_json=None, error=None):
    from uuid import uuid5, NAMESPACE_URL
    from datetime import datetime, timezone
    with _sessions()() as db:
        execution_id = str(uuid5(NAMESPACE_URL, f"{state['analysis_run_id']}:{name}"))
        row = db.get(AgentExecution, execution_id)
        if row is None:
            row = AgentExecution(id=execution_id, tenant_id=state["tenant_id"], analysis_run_id=state["analysis_run_id"],
                                 agent_name=name, status=status, input_json=input_json, output_json=output_json,
                                 error=error, completed_at=datetime.now(timezone.utc))
            db.add(row)
        else:
            row.status, row.input_json, row.output_json, row.error = status, input_json, output_json, error
            row.completed_at = datetime.now(timezone.utc)
        db.commit()


def _guarded(name, function):
    def run(state):
        try:
            return function(state)
        except GraphInterrupt:
            # Human approval is the expected pause point, not an agent failure.
            raise
        except Exception as exc:
            logger.exception("Workflow agent failed", extra={"agent_name": name,
                "analysis_run_id": state.get("analysis_run_id"), "tenant_id": state.get("tenant_id"),
                "user_id": state.get("user_id")})
            try:
                _log_agent(state, name, "failed", error=f"{type(exc).__name__}: see server logs for diagnostics")
            except Exception:
                logger.exception("Could not persist workflow failure", extra={"agent_name": name,
                    "analysis_run_id": state.get("analysis_run_id"), "tenant_id": state.get("tenant_id")})
            raise
    return run


def _record_llm(state, agent_name, result):
    with _sessions()() as db:
        db.add(LLMUsage(id=__import__("uuid").uuid4().__str__(), tenant_id=state["tenant_id"],
            analysis_run_id=state["analysis_run_id"], agent_name=agent_name, provider=result.provider or "unconfigured",
            model=result.model or "", prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
            estimated_cost=result.estimated_cost, status=result.status))
        db.commit()


def _llm_settings(state):
    settings = dict(state.get("tenant_settings") or {})
    if not settings.get("llm_enabled"): return settings
    daily_limit = int(settings.get("llm_daily_call_limit", 50))
    utc_start = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    with _sessions()() as db:
        used = db.scalar(select(func.count(LLMUsage.id)).where(LLMUsage.tenant_id==state["tenant_id"],
            LLMUsage.created_at>=utc_start, LLMUsage.status!="fallback_no_authorized_provider")) or 0
    if used >= daily_limit:
        settings["llm_enabled"] = False
        settings["llm_budget_exhausted"] = True
    return settings


def _detect(state: WorkflowState):
    result = run_detect(state["customer_id"], state["features"], state["events"])
    output = result.model_dump(mode="json")
    _log_agent(state, "Detect", "completed", {"feature_count":len(state["features"]),"event_count":len(state["events"])}, output)
    return {"detect": output, "status": "detect_completed"}


def _diagnose(state: WorkflowState):
    gateway = LLMGateway()
    result, usage = run_diagnose(state["detect"], state["features"], state["events"], gateway, _llm_settings(state))
    _record_llm(state, "Diagnose", usage)
    output = result.model_dump(mode="json")
    _log_agent(state, "Diagnose", "completed", {"signal_count":len(state["detect"]["signals"])}, output)
    return {"diagnose": output, "status": "diagnose_completed"}


def _predict(state: WorkflowState):
    result = PredictAgent().run(state["customer_id"], state["features"])
    with _sessions()() as db:
        db.add(Prediction(id=__import__("uuid").uuid4().__str__(),tenant_id=state["tenant_id"],
            customer_pk=state["customer_pk"],customer_id_snapshot=state["customer_id"],
            probability=result["churn_probability"],decision_threshold=result["decision_threshold"],
            predicted_churn=result["predicted_churn"],risk_level=result["risk_level"],
            model_version=result["model_version"]))
        db.commit()
    _log_agent(state, "Predict", "completed", {"model_version":result["model_version"],"input_feature_count":len(state["features"])}, result)
    return {"prediction": result, "status": "predict_completed"}


def _strategy(state: WorkflowState):
    gateway = LLMGateway()
    candidates, usage = run_strategy(state["features"], state["diagnose"], state["prediction"],
                                     state.get("mcia_weights"), _llm_settings(state), gateway)
    _record_llm(state, "Strategy", usage)
    recommendation_ids = []
    with _sessions()() as db:
        run = db.get(AnalysisRun, state["analysis_run_id"])
        for action in candidates[:8]:
            key = f"{state['analysis_run_id']}:{action['id']}"
            row = db.scalar(select(Recommendation).where(Recommendation.tenant_id == state["tenant_id"],
                                                        Recommendation.idempotency_key == key))
            if row is None:
                row = Recommendation(id=__import__("uuid").uuid4().__str__(), tenant_id=state["tenant_id"],
                    customer_pk=state["customer_pk"], analysis_run_id=state["analysis_run_id"],
                    action_id=action["id"], action_name=action["name"], rationale=action["rationale"],
                    mcia={k:action[k] for k in ("money","time","relation","feasibility","mcia_score","weights","estimated_cost","cost_currency")},
                    eligibility=action["eligibility_reason"], status="pending_approval", idempotency_key=key,
                    created_by=state["user_id"])
                db.add(row); db.flush()
            action["recommendation_id"] = row.id
            recommendation_ids.append(row.id)
        run.status = "pending_approval"
        run.result = {"detect":state["detect"],"diagnose":state["diagnose"],"prediction":state["prediction"],
                      "strategy":candidates,"recommendation_ids":recommendation_ids}
        db.add(AuditLog(id=__import__("uuid").uuid4().__str__(), tenant_id=state["tenant_id"], user_id=state["user_id"],
                        action="recommendations.proposed", resource_type="analysis_run", resource_id=run.id,
                        details={"recommendation_ids":recommendation_ids,"count":len(recommendation_ids)}))
        db.commit()
    output = {"candidates":candidates,"recommendation_ids":recommendation_ids}
    _log_agent(state, "Strategy", "completed", {"diagnosis_items":len(state["diagnose"]["items"])}, output)
    return {"strategy":candidates,"recommendation_ids":recommendation_ids,"status":"pending_approval"}


def _approval_gate(state: WorkflowState):
    reviewed = set(state.get("reviewed_recommendation_ids", []))
    pending_ids = [item for item in state.get("recommendation_ids", []) if item not in reviewed]
    payload = {"message":"Human review is required before any action can execute.",
        "analysis_run_id":state["analysis_run_id"],"customer_id":state["customer_id"],
        "recommendation_ids":pending_ids,
        "recommendations":[{"id":item["recommendation_id"],"action_id":item["id"],"name":item["name"],"score":item["mcia_score"]}
                            for item in state.get("strategy",[]) if item.get("recommendation_id") in pending_ids]}
    _log_agent(state, "HumanApproval", "awaiting_approval",
        {"recommendation_ids":pending_ids}, payload)
    raw = interrupt(payload, response_schema=ApprovalResume)
    decision = ApprovalResume.model_validate(raw)
    if decision.recommendation_id not in pending_ids:
        raise ValueError("Approval references a recommendation outside this analysis run")
    _log_agent(state, "HumanApproval", decision.decision, {"recommendation_ids":state.get("recommendation_ids",[])}, decision.model_dump())
    rejected = list(state.get("rejected_recommendation_ids", []))
    if decision.decision == "rejected": rejected.append(decision.recommendation_id)
    return {"approval":decision.model_dump(),"execution":None,
            "reviewed_recommendation_ids":[*state.get("reviewed_recommendation_ids",[]),decision.recommendation_id],
            "rejected_recommendation_ids":rejected,
            "status":"approved" if decision.decision == "approved" else "rejected"}


def _route_after_approval(state: WorkflowState):
    if state.get("approval",{}).get("decision") == "approved": return "execute"
    return "approval_gate" if len(state.get("reviewed_recommendation_ids",[])) < len(state.get("recommendation_ids",[])) else "finish"


def _route_after_execute(state: WorkflowState):
    return "approval_gate" if len(state.get("reviewed_recommendation_ids",[])) < len(state.get("recommendation_ids",[])) else "finish"


def _route_after_strategy(state: WorkflowState):
    return "approval_gate" if state.get("recommendation_ids") else "finish_empty"


def _finish_empty(state: WorkflowState):
    with _sessions()() as db:
        run = db.get(AnalysisRun, state["analysis_run_id"])
        run.status = "completed_no_recommendations"
        run.completed_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        db.commit()
    return {"status":"completed_no_recommendations"}


def _execute(state: WorkflowState):
    approval = ApprovalResume.model_validate(state["approval"])
    with _sessions()() as db:
        result = ExecuteAgent().run(db, tenant_id=state["tenant_id"], recommendation_id=approval.recommendation_id,
                                    reviewer_id=approval.reviewer_id, expected_customer_id=state["customer_id"])
        run = db.get(AnalysisRun, state["analysis_run_id"])
        run.status = "pending_approval"
        db.commit()
    _log_agent(state, f"Execute:{approval.recommendation_id}", "completed", {"recommendation_id":approval.recommendation_id}, result)
    return {"execution":result,"status":"executed"}


def _finish(state: WorkflowState):
    with _sessions()() as db:
        run = db.get(AnalysisRun, state["analysis_run_id"])
        run.status = "completed"
        run.completed_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        db.commit()
    return {"status":"completed"}


def _build_graph():
    path = Path(__import__("os").getenv("AURA_GRAPH_CHECKPOINT_PATH", ROOT / "data/aura_graph_checkpoints.sqlite"))
    if not path.is_absolute(): path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    builder = StateGraph(WorkflowState)
    for node, function in (("detect",_guarded("Detect",_detect)),("diagnose",_guarded("Diagnose",_diagnose)),
                           ("predict",_guarded("Predict",_predict)),("strategy",_guarded("Strategy",_strategy)),
                           ("approval_gate",_guarded("HumanApproval",_approval_gate)),
                           ("execute",_guarded("Execute",_execute)),("finish",_finish),
                           ("finish_empty",_finish_empty)):
        builder.add_node(node,function)
    builder.add_edge(START,"detect"); builder.add_edge("detect","diagnose"); builder.add_edge("diagnose","predict")
    builder.add_edge("predict","strategy")
    builder.add_conditional_edges("strategy",_route_after_strategy,{"approval_gate":"approval_gate","finish_empty":"finish_empty"})
    builder.add_conditional_edges("approval_gate",_route_after_approval,
                                  {"execute":"execute","approval_gate":"approval_gate","finish":"finish"})
    builder.add_conditional_edges("execute",_route_after_execute,{"approval_gate":"approval_gate","finish":"finish"})
    builder.add_edge("finish_empty",END); builder.add_edge("finish",END)
    return builder.compile(checkpointer=checkpointer)


GRAPH = _build_graph()
_GRAPH_LOCK = threading.RLock()
_SESSION_FACTORY = ContextVar("aura_workflow_session_factory", default=SessionLocal)


def _sessions(): return _SESSION_FACTORY.get()


def start_analysis(payload: AnalysisStart, session_factory=None) -> dict:
    state = payload.model_dump()
    config = {"configurable":{"thread_id":payload.analysis_run_id},"recursion_limit":12}
    token = _SESSION_FACTORY.set(session_factory or SessionLocal)
    with _GRAPH_LOCK:
        try: return GRAPH.invoke(state, config)
        finally: _SESSION_FACTORY.reset(token)


def resume_analysis(analysis_run_id: str, approval: ApprovalResume, session_factory=None) -> dict:
    config = {"configurable":{"thread_id":analysis_run_id},"recursion_limit":12}
    token = _SESSION_FACTORY.set(session_factory or SessionLocal)
    with _GRAPH_LOCK:
        try:
            state_snapshot = GRAPH.get_state(config)
            if not state_snapshot.values: raise ValueError("No paused workflow exists for this analysis run")
            return GRAPH.invoke(Command(resume=approval.model_dump()), config)
        finally: _SESSION_FACTORY.reset(token)
