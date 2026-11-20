"""Initial local FastAPI surface: health, actual model prediction, and customers."""
from pathlib import Path
import pandas as pd
import json
import re
from uuid import uuid4, uuid5, NAMESPACE_URL
from datetime import datetime, timezone
from io import BytesIO
import asyncio
import logging
import time
from jose import JWTError, jwt
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlalchemy import select, func
from sqlalchemy.orm import Session, sessionmaker
from fastapi import Depends, Request, UploadFile, File
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from ml.inference.predict import ChurnPredictor, ROOT
from backend.services.event_simulator import simulate_events
from backend.graph_engine.customer_graph import build_customer_graph
from backend.mcia.optimizer import rank_actions
from backend.database.models import (Tenant, User, Membership, Customer, DatasetVersion, Prediction as PredictionRecord,
                                     CustomerEvent, AnalysisRun, AgentExecution, Recommendation, Approval, ActionExecution,
                                     OutcomeFeedback, AuditLog)
from backend.database.session import get_db, SessionLocal
from backend.security.auth import (AuthContext, current_context, require_roles, hash_password, verify_password,
                                   issue_token, token_ttl_seconds)
from backend.llm.gateway import LLMGateway
from backend.agents.schemas import AnalysisStart, ApprovalResume

app = FastAPI(title="AURA — AI-Powered User Retention Analytics", version="0.1.0")
_predictor = None
logger = logging.getLogger(__name__)


@app.middleware("http")
async def request_diagnostics(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if re.fullmatch(r"[A-Za-z0-9._:-]{1,80}", supplied) else str(uuid4())
    request.state.request_id = request_id
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled API exception", extra={"request_id": request_id,
            "method": request.method, "path": request.url.path,
            "tenant_id": getattr(request.state, "tenant_id", None),
            "user_id": getattr(request.state, "user_id", None)})
        from fastapi.responses import JSONResponse
        response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
    response.headers["X-Request-ID"] = request_id
    logger.info("API request completed", extra={"request_id": request_id, "method": request.method,
        "path": request.url.path, "status_code": response.status_code,
        "duration_ms": round((time.monotonic() - started) * 1000, 2),
        "tenant_id": getattr(request.state, "tenant_id", None), "user_id": getattr(request.state, "user_id", None)})
    return response


@app.websocket("/api/events")
async def tenant_events(websocket: WebSocket, token: str):
    """Authenticated tenant event feed. Event payloads contain ids/status only."""
    from backend.security.auth import JWT_SECRET, ALGORITHM
    from backend.database.models import utcnow
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        with SessionLocal() as db:
            user = db.get(User, claims.get("sub")); tenant = db.get(Tenant, claims.get("tid"))
            membership = db.scalar(select(Membership).where(Membership.user_id == claims.get("sub"),
                Membership.tenant_id == claims.get("tid")))
            if (not user or not tenant or not membership or not user.is_active
                    or claims.get("ver") != user.auth_version):
                raise ValueError("invalid session")
            tenant_id = tenant.id
    except (JWTError, ValueError, TypeError):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    cursor = datetime.now(timezone.utc)
    try:
        while True:
            await asyncio.sleep(2)
            with SessionLocal() as db:
                audit_rows = db.scalars(select(AuditLog).where(AuditLog.tenant_id == tenant_id,
                    AuditLog.created_at > cursor).order_by(AuditLog.created_at).limit(100)).all()
                agent_rows = db.scalars(select(AgentExecution).where(AgentExecution.tenant_id == tenant_id,
                    AgentExecution.started_at > cursor).order_by(AgentExecution.started_at).limit(100)).all()
                events = [{"kind":"audit","type":row.action,"resource_type":row.resource_type,
                    "resource_id":row.resource_id,"at":row.created_at.isoformat()} for row in audit_rows]
                events += [{"kind":"agent","type":"agent.completed" if row.status == "completed" else "agent.updated",
                    "analysis_run_id":row.analysis_run_id,"agent":row.agent_name,"status":row.status,
                    "at":row.started_at.isoformat()} for row in agent_rows]
                if audit_rows or agent_rows:
                    cursor = max([row.created_at for row in audit_rows] + [row.started_at for row in agent_rows])
            for event in events:
                await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.get("/api/reports/tenant.pdf")
def tenant_pdf_report(context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    output = BytesIO(); pdf = canvas.Canvas(output, pagesize=letter)
    y = 740
    customer_count = db.scalar(select(func.count(Customer.id)).where(Customer.tenant_id == context.tenant.id)) or 0
    run_count = db.scalar(select(func.count(AnalysisRun.id)).where(AnalysisRun.tenant_id == context.tenant.id)) or 0
    rec_count = db.scalar(select(func.count(Recommendation.id)).where(Recommendation.tenant_id == context.tenant.id)) or 0
    execution_count = db.scalar(select(func.count(ActionExecution.id)).where(ActionExecution.tenant_id == context.tenant.id)) or 0
    lines = ["AURA Tenant Activity Report", f"Organization: {context.tenant.name}",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}", "", f"Customers: {customer_count}",
        f"Analysis runs: {run_count}", f"Recommendations: {rec_count}", f"Simulated executions: {execution_count}",
        "", "Executions are simulations and do not establish customer retention or ROI."]
    pdf.setTitle("AURA Tenant Activity Report"); pdf.setFont("Helvetica", 11)
    for line in lines:
        pdf.drawString(54, y, line[:110]); y -= 22
    pdf.save()
    return Response(output.getvalue(), media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=aura_tenant_report.pdf"})


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str
    features: dict


class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
    organization_name: str
    industry: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str
    tenant_id: str | None = None


class UserInviteRequest(BaseModel):
    email: str
    full_name: str
    password: str
    roles: list[str] = Field(default_factory=lambda: ["ANALYST"])


class MCIAWeightsRequest(BaseModel):
    money: float = Field(ge=0)
    time: float = Field(ge=0)
    relation: float = Field(ge=0)
    feasibility: float = Field(ge=0)


class BatchAnalysisRequest(BaseModel):
    customer_ids: list[str] = Field(min_length=1, max_length=100)


class ReviewRequest(BaseModel):
    note: str | None = Field(default=None, max_length=2000)


class OutcomeRequest(BaseModel):
    outcome_status: str
    customer_response: str | None = None
    notes: str | None = Field(default=None, max_length=4000)
    follow_up_status: str | None = None


class LLMSettingsRequest(BaseModel):
    enabled: bool
    daily_call_limit: int = Field(default=50, ge=1, le=1000)


def _validate_upload(content: bytes, filename: str):
    if not filename.lower().endswith(".csv"): raise HTTPException(415, "Upload a .csv file")
    if len(content) > 25 * 1024 * 1024: raise HTTPException(413, "CSV upload limit is 25 MB")
    try: frame = pd.read_csv(BytesIO(content))
    except Exception as exc: raise HTTPException(422, "Unable to parse uploaded CSV") from exc
    dictionary = pd.read_csv(ROOT / "data/raw/AURA_Data_Dictionary.csv")
    inputs = dictionary.loc[dictionary.role.eq("Input feature"), "parameter"].tolist()
    required = {"customer_id", *inputs}
    actual = set(frame.columns)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - {"churn"})
    errors = []
    if missing: errors.append({"type": "missing_columns", "columns": missing})
    if unexpected: errors.append({"type": "unexpected_columns", "columns": unexpected})
    if "customer_id" in frame and frame.customer_id.isna().any(): errors.append({"type": "missing_customer_ids"})
    if "customer_id" in frame and frame.customer_id.duplicated().any(): errors.append({"type": "duplicate_customer_ids"})
    if frame.isna().any().any(): errors.append({"type": "missing_values", "columns": frame.columns[frame.isna().any()].tolist()})
    if "churn" in frame and not set(frame.churn.dropna().unique()).issubset({0, 1}): errors.append({"type": "invalid_target_labels"})
    numeric = dictionary.loc[dictionary.role.eq("Input feature") & dictionary.type.isin(["int64", "float64"]), "parameter"].tolist()
    for column in numeric:
        if column in frame and not pd.api.types.is_numeric_dtype(frame[column]): errors.append({"type": "invalid_type", "column": column, "expected": "numeric"})
    for column in ("age", "tenure_months", "monthly_charges", "total_charges", "days_to_renewal", "login_frequency_30d"):
        if column in frame and pd.api.types.is_numeric_dtype(frame[column]) and frame[column].lt(0).any():
            errors.append({"type": "out_of_range", "column": column, "minimum": 0})
    reference = pd.read_csv(ROOT / "data/raw/AURA_Synthetic_Customer_Retention_Dataset.csv", nrows=12000)
    category_warnings = {}
    for column in dictionary.loc[dictionary.role.eq("Input feature") & dictionary.type.eq("object"), "parameter"]:
        if column in frame:
            novel = sorted(set(frame[column].dropna().astype(str)) - set(reference[column].dropna().astype(str)))
            if novel: category_warnings[column] = novel[:30]
    report = {"valid": not errors, "row_count": int(len(frame)), "column_count": int(len(frame.columns)),
              "columns": list(frame.columns), "missing_columns": missing, "unexpected_columns": unexpected,
              "errors": errors, "category_warnings_vs_supplied_demo": category_warnings,
              "target_present": "churn" in frame, "churned_labels": int(frame.churn.sum()) if "churn" in frame and not errors else None}
    preview = frame.head(10).astype(object).where(pd.notna(frame.head(10)), None).to_dict(orient="records")
    return frame, report, preview


@app.post("/api/auth/signup", status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email): raise HTTPException(422, "Enter a valid email address")
    if len(payload.password) < 10: raise HTTPException(422, "Password must contain at least 10 characters")
    if not payload.full_name.strip() or not payload.organization_name.strip(): raise HTTPException(422, "Name and organization are required")
    if db.scalar(select(User.id).where(User.email == email)): raise HTTPException(409, "An account with this email already exists")
    tenant = Tenant(id=str(uuid4()), name=payload.organization_name.strip(), industry=payload.industry)
    user = User(id=str(uuid4()), email=email, password_hash=hash_password(payload.password), full_name=payload.full_name.strip())
    membership = Membership(id=str(uuid4()), tenant_id=tenant.id, user_id=user.id, role="OWNER")
    db.add_all([tenant, user, membership])
    db.commit()
    return {"access_token": issue_token(user, tenant, {"OWNER"}), "token_type": "bearer", "expires_in": token_ttl_seconds(),
            "user": {"id": user.id, "email": user.email, "full_name": user.full_name},
            "tenant": {"id": tenant.id, "name": tenant.name}, "roles": ["OWNER"]}


@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    memberships = db.scalars(select(Membership).where(Membership.user_id == user.id)).all()
    if payload.tenant_id:
        membership = next((item for item in memberships if item.tenant_id == payload.tenant_id), None)
    else:
        if len(memberships) > 1:
            raise HTTPException(409, detail={"message": "Choose an organization", "tenants": [
                {"id": item.tenant_id, "name": db.get(Tenant, item.tenant_id).name} for item in memberships]})
        membership = memberships[0] if memberships else None
    if membership is None: raise HTTPException(403, "No access to the selected organization")
    tenant = db.get(Tenant, membership.tenant_id)
    roles = {role.strip().upper() for role in membership.role.replace(";", ",").split(",") if role.strip()}
    return {"access_token": issue_token(user, tenant, roles), "token_type": "bearer", "expires_in": token_ttl_seconds(),
            "user": {"id": user.id, "email": user.email, "full_name": user.full_name},
            "tenant": {"id": tenant.id, "name": tenant.name}, "roles": sorted(roles)}


@app.get("/api/auth/me")
def auth_me(context: AuthContext = Depends(current_context)):
    return {"user": {"id": context.user.id, "email": context.user.email, "full_name": context.user.full_name},
            "tenant": {"id": context.tenant.id, "name": context.tenant.name}, "roles": sorted(context.roles)}


@app.get("/api/tenant/users")
def tenant_users(context: AuthContext = Depends(require_roles("OWNER", "ADMIN")), db: Session = Depends(get_db)):
    members = db.scalars(select(Membership).where(Membership.tenant_id == context.tenant.id)).all()
    return {"items": [{"user_id": member.user.id, "email": member.user.email, "full_name": member.user.full_name,
                       "roles": [role.strip().upper() for role in member.role.split(",") if role.strip()],
                       "is_active": member.user.is_active} for member in members]}


@app.post("/api/tenant/users", status_code=201)
def invite_user(payload: UserInviteRequest, context: AuthContext = Depends(require_roles("OWNER", "ADMIN")),
                db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email): raise HTTPException(422, "Enter a valid email address")
    if len(payload.password) < 10: raise HTTPException(422, "Password must contain at least 10 characters")
    allowed = {"OWNER", "ADMIN", "ANALYST", "APPROVER"}
    roles = {role.strip().upper() for role in payload.roles}
    if not roles or not roles.issubset(allowed): raise HTTPException(422, "Roles must be Owner, Admin, Analyst, or Approver")
    if "OWNER" in roles and "OWNER" not in context.roles: raise HTTPException(403, "Only an owner can assign the owner role")
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(id=str(uuid4()), email=email, password_hash=hash_password(payload.password), full_name=payload.full_name.strip())
        db.add(user); db.flush()
    elif not user.is_active: raise HTTPException(409, "This account is disabled")
    existing = db.scalar(select(Membership).where(Membership.tenant_id == context.tenant.id, Membership.user_id == user.id))
    if existing: raise HTTPException(409, "User is already a member of this organization")
    db.add(Membership(id=str(uuid4()), tenant_id=context.tenant.id, user_id=user.id,
                      role=",".join(sorted(roles))))
    db.add(AuditLog(id=str(uuid4()), tenant_id=context.tenant.id, user_id=context.user.id,
                    action="tenant.user_invite", resource_type="user", resource_id=user.id,
                    details={"roles": sorted(roles)}))
    db.commit()
    return {"user_id": user.id, "email": user.email, "roles": sorted(roles)}


@app.get("/api/settings")
def tenant_settings(context: AuthContext = Depends(current_context)):
    return {"tenant": {"id": context.tenant.id, "name": context.tenant.name, "industry": context.tenant.industry},
            "mcia_weights": context.tenant.settings_json.get("mcia_weights", {"money":0.25,"time":0.25,"relation":0.25,"feasibility":0.25}),
            "active_prediction_threshold": (_predictor.threshold if _predictor else ChurnPredictor().threshold)}


@app.put("/api/settings/mcia")
def update_mcia_weights(payload: MCIAWeightsRequest, context: AuthContext = Depends(require_roles("OWNER", "ADMIN")),
                        db: Session = Depends(get_db)):
    weights = payload.model_dump()
    if sum(weights.values()) <= 0: raise HTTPException(422, "At least one weight must be positive")
    tenant = db.get(Tenant, context.tenant.id)
    settings = dict(tenant.settings_json or {})
    previous = settings.get("mcia_weights", {"money":0.25,"time":0.25,"relation":0.25,"feasibility":0.25})
    settings["mcia_weights"] = weights
    tenant.settings_json = settings
    db.add(AuditLog(id=str(uuid4()), tenant_id=tenant.id, user_id=context.user.id,
        action="tenant.mcia_weights.update", resource_type="tenant_settings", resource_id=tenant.id,
        details={"previous":previous,"updated":weights}))
    db.commit()
    return {"mcia_weights": weights, "note": "Weights affect future candidate rankings; historical recommendations retain their recorded breakdown."}


@app.get("/api/settings/llm")
def llm_settings(context: AuthContext = Depends(require_roles("OWNER", "ADMIN"))):
    return {**LLMGateway().status(context.tenant.settings_json),
            "daily_call_limit":context.tenant.settings_json.get("llm_daily_call_limit",50)}


@app.put("/api/settings/llm")
def update_llm_authorization(payload: LLMSettingsRequest, context: AuthContext = Depends(require_roles("OWNER", "ADMIN")),
                             db: Session = Depends(get_db)):
    settings = dict(context.tenant.settings_json or {})
    gateway_status = LLMGateway().status({**settings,"llm_enabled":payload.enabled})
    if payload.enabled and not gateway_status["provider_configured"]:
        raise HTTPException(422, "No server-side LLM provider credentials/model are configured")
    settings["llm_enabled"] = bool(payload.enabled)
    settings["llm_daily_call_limit"] = payload.daily_call_limit
    tenant = db.get(Tenant, context.tenant.id); tenant.settings_json = settings
    db.add(AuditLog(id=str(uuid4()), tenant_id=tenant.id, user_id=context.user.id,
                    action="tenant.llm_authorization.update", resource_type="tenant_settings", resource_id=tenant.id,
                    details={"enabled":bool(payload.enabled),"daily_call_limit":payload.daily_call_limit,
                             "provider":gateway_status.get("provider")}))
    db.commit()
    return LLMGateway().status(tenant.settings_json)


@app.post("/api/auth/logout")
def logout(context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    user = db.get(User, context.user.id)
    user.auth_version += 1
    db.commit()
    return {"status": "logged_out"}


def _customer_record(db: Session, tenant_id: str, customer_id: str):
    record = db.scalar(select(Customer).where(Customer.tenant_id == tenant_id, Customer.customer_id == customer_id))
    if record is None: raise HTTPException(status_code=404, detail="Customer not found")
    return record


def _customer_payload(record: Customer):
    return {"customer_id": record.customer_id, **record.features}


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": (ROOT / "ml/artifacts/best_model.pkl").exists()}


@app.post("/api/datasets/demo", status_code=201)
def import_demo_dataset(context: AuthContext = Depends(require_roles("OWNER", "ADMIN")), db: Session = Depends(get_db)):
    version_name = "aura-synthetic-v1"
    existing = db.scalar(select(DatasetVersion).where(DatasetVersion.tenant_id == context.tenant.id,
                                                       DatasetVersion.version == version_name))
    if existing:
        return {"dataset_version": existing.version, "row_count": existing.row_count, "already_imported": True}
    frame = pd.read_csv(ROOT / "data/raw/AURA_Synthetic_Customer_Retention_Dataset.csv")
    version = DatasetVersion(id=str(uuid4()), tenant_id=context.tenant.id, version=version_name,
        filename="AURA_Synthetic_Customer_Retention_Dataset.csv", row_count=len(frame), created_by=context.user.id,
        validation_report={"source": "supplied synthetic AURA dataset", "churned_labels": int(frame.churn.sum()),
                           "non_churned_labels": int((frame.churn == 0).sum()), "labels_stored_on_customer": False})
    db.add(version); db.flush()
    chunk_size = 1000
    input_columns = [column for column in frame.columns if column not in ("customer_id", "churn")]
    for start in range(0, len(frame), chunk_size):
        rows = frame.iloc[start:start+chunk_size]
        db.add_all([Customer(id=str(uuid5(NAMESPACE_URL, f"{context.tenant.id}:{row.customer_id}")),
                             tenant_id=context.tenant.id, customer_id=str(row.customer_id),
                             segment=str(row.customer_segment), dataset_version_id=version.id,
                             features={key: row[key].item() if hasattr(row[key], "item") else row[key] for key in input_columns})
                    for _, row in rows.iterrows()])
    db.add(AuditLog(id=str(uuid4()), tenant_id=context.tenant.id, user_id=context.user.id,
                    action="dataset.demo_import", resource_type="dataset_version", resource_id=version.id,
                    details={"rows": len(frame), "version": version_name}))
    db.commit()
    return {"dataset_version": version_name, "row_count": len(frame), "already_imported": False,
            "provenance": "supplied synthetic development dataset"}


@app.post("/api/datasets/validate")
async def validate_dataset(file: UploadFile = File(...), context: AuthContext = Depends(require_roles("OWNER", "ADMIN"))):
    content = await file.read(25 * 1024 * 1024 + 1)
    _, report, preview = _validate_upload(content, file.filename or "")
    return {"validation": report, "preview": preview}


@app.post("/api/datasets/upload", status_code=201)
async def upload_dataset(file: UploadFile = File(...), context: AuthContext = Depends(require_roles("OWNER", "ADMIN")),
                         db: Session = Depends(get_db)):
    content = await file.read(25 * 1024 * 1024 + 1)
    frame, report, _ = _validate_upload(content, file.filename or "")
    if not report["valid"]: raise HTTPException(status_code=422, detail={"message": "Dataset validation failed", "validation": report})
    version_id = str(uuid4())
    version_name = f"upload-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{version_id[:8]}"
    version = DatasetVersion(id=version_id, tenant_id=context.tenant.id, version=version_name,
        filename=(file.filename or "dataset.csv")[:255], row_count=len(frame), created_by=context.user.id,
        validation_report=report)
    db.add(version); db.flush()
    input_columns = pd.read_csv(ROOT / "data/raw/AURA_Data_Dictionary.csv").query("role == 'Input feature'").parameter.tolist()
    for start in range(0, len(frame), 1000):
        rows = frame.iloc[start:start+1000]
        db.add_all([Customer(id=str(uuid5(NAMESPACE_URL, f"{context.tenant.id}:{version_id}:{row.customer_id}")),
            tenant_id=context.tenant.id, customer_id=str(row.customer_id), segment=str(row.get("customer_segment", "")),
            dataset_version_id=version_id,
            features={key: row[key].item() if hasattr(row[key], "item") else row[key] for key in input_columns})
            for _, row in rows.iterrows()])
    db.add(AuditLog(id=str(uuid4()), tenant_id=context.tenant.id, user_id=context.user.id,
                    action="dataset.upload", resource_type="dataset_version", resource_id=version_id,
                    details={"row_count": len(frame), "filename": version.filename}))
    try: db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Customer IDs conflict with an existing tenant dataset") from exc
    return {"dataset_version": version_name, "row_count": len(frame), "validation": report,
            "model_retrained": False, "note": "Customer data was imported; the active model was not changed."}


@app.get("/api/datasets")
def list_datasets(context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    items = db.scalars(select(DatasetVersion).where(DatasetVersion.tenant_id == context.tenant.id)
                       .order_by(DatasetVersion.created_at.desc())).all()
    return {"items": [{"version": row.version, "filename": row.filename, "row_count": row.row_count,
                        "validation_report": row.validation_report, "created_at": row.created_at.isoformat()} for row in items]}


@app.post("/api/predictions")
def predict(request: PredictionRequest, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    global _predictor
    try:
        if _predictor is None: _predictor = ChurnPredictor()
        customer_row = _customer_record(db, context.tenant.id, request.customer_id)
        if request.features != customer_row.features:
            raise HTTPException(status_code=422, detail="Prediction features must match this tenant's stored customer record")
        result = _predictor.predict(request.customer_id, customer_row.features)
        db.add(PredictionRecord(id=str(uuid4()), tenant_id=context.tenant.id, customer_pk=customer_row.id,
            customer_id_snapshot=request.customer_id, probability=result["churn_probability"],
            decision_threshold=result["decision_threshold"], predicted_churn=result["predicted_churn"],
            risk_level=result["risk_level"], model_version=result["model_version"]))
        db.commit()
        return result
    except FileNotFoundError as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/customers")
def customers(limit: int = 100, offset: int = 0, search: str | None = None,
              risk_level: str | None = None, segment: str | None = None,
              sort_by: str = "customer_id", descending: bool = False,
              context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    if limit < 1 or limit > 200 or offset < 0: raise HTTPException(status_code=422, detail="Invalid pagination")
    if risk_level and risk_level.upper() not in {"CRITICAL","HIGH","MEDIUM","LOW","NOT_ANALYZED"}:
        raise HTTPException(422, "Unknown risk level")
    if sort_by not in {"customer_id","segment","risk_level","churn_probability"}:
        raise HTTPException(422, "Unsupported sort field")
    latest_prediction = (select(PredictionRecord.customer_pk.label("customer_pk"),
        func.max(PredictionRecord.created_at).label("latest_at")).where(
        PredictionRecord.tenant_id == context.tenant.id, PredictionRecord.customer_pk.is_not(None))
        .group_by(PredictionRecord.customer_pk).subquery())
    latest = PredictionRecord
    query = (select(Customer, latest).where(Customer.tenant_id == context.tenant.id)
        .outerjoin(latest_prediction, latest_prediction.c.customer_pk == Customer.id)
        .outerjoin(latest, (latest.customer_pk == Customer.id) & (latest.created_at == latest_prediction.c.latest_at)
                   & (latest.tenant_id == context.tenant.id)))
    if search: query = query.where(Customer.customer_id.ilike(f"%{search[:100]}%"))
    if segment: query = query.where(Customer.segment == segment)
    if risk_level:
        query = query.where(latest.risk_level == risk_level.upper()) if risk_level.upper() != "NOT_ANALYZED" else query.where(latest.id.is_(None))
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    sort_column = {"customer_id":Customer.customer_id,"segment":Customer.segment,
        "risk_level":latest.risk_level,"churn_probability":latest.probability}[sort_by]
    sort_expr = sort_column.desc().nullslast() if descending else sort_column.asc().nullslast()
    records = db.execute(query.order_by(sort_expr,Customer.customer_id).offset(offset).limit(limit)).all()
    items = []
    for record, prediction in records:
        item = _customer_payload(record)
        item["churn_probability"] = prediction.probability if prediction else None
        item["risk_level"] = prediction.risk_level if prediction else "NOT_ANALYZED"
        item["prediction_timestamp"] = prediction.created_at.isoformat() if prediction else None
        items.append(item)
    return {"total": total, "items": items, "limit": limit, "offset": offset,
            "sort_by": sort_by, "descending": descending}


@app.get("/api/dashboard/overview")
def overview(context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    total_customers = db.scalar(select(func.count(Customer.id)).where(Customer.tenant_id == context.tenant.id)) or 0
    predictions = db.scalars(select(PredictionRecord).where(PredictionRecord.tenant_id == context.tenant.id)
                             .order_by(PredictionRecord.created_at.desc())).all()
    latest = {}
    for prediction in predictions: latest.setdefault(prediction.customer_id_snapshot, prediction)
    distribution = {level: sum(1 for prediction in latest.values() if prediction.risk_level == level)
                    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    labels = db.scalar(select(DatasetVersion).where(DatasetVersion.tenant_id == context.tenant.id)
                       .order_by(DatasetVersion.created_at.desc()).limit(1))
    label_summary = labels.validation_report if labels else {}
    pending = db.scalar(select(func.count(Recommendation.id)).where(Recommendation.tenant_id == context.tenant.id,
                            Recommendation.status == "pending_approval")) or 0
    approved = db.scalar(select(func.count(Approval.id)).where(Approval.tenant_id == context.tenant.id,
        Approval.decision=="approved")) or 0
    rec_statuses = db.execute(select(Recommendation.status,func.count(Recommendation.id)).where(
        Recommendation.tenant_id==context.tenant.id).group_by(Recommendation.status)).all()
    segments = db.execute(select(Customer.segment,func.count(Customer.id)).where(
        Customer.tenant_id==context.tenant.id).group_by(Customer.segment).order_by(func.count(Customer.id).desc()).limit(12)).all()
    recent_analyses = db.scalars(select(AnalysisRun).where(AnalysisRun.tenant_id==context.tenant.id)
        .order_by(AnalysisRun.created_at.desc()).limit(5)).all()
    recent_executions = db.scalar(select(func.count(ActionExecution.id)).where(
        ActionExecution.tenant_id==context.tenant.id,
        ActionExecution.executed_at>=datetime.now(timezone.utc)-__import__("datetime").timedelta(days=7))) or 0
    return {"total_customers": int(total_customers), "analyzed_customers": len(latest),
            "prediction_risk_distribution": distribution, "pending_recommendations": pending,
            "approved_interventions":approved,"recommendation_status_distribution":dict(rec_statuses),
            "segment_distribution":[{"segment":segment or "Unspecified","customers":count} for segment,count in segments],
            "recent_analysis_runs":[{"run_id":row.id,"customer_id":(db.get(Customer,row.customer_pk).customer_id if db.get(Customer,row.customer_pk) else None),
                "status":row.status,"created_at":row.created_at.isoformat()} for row in recent_analyses],
            "recent_simulated_executions_7d":recent_executions,
            "dataset_label_summary": label_summary,
            "source": "tenant-scoped stored records; dataset labels shown only when a dataset version is imported"}


@app.get("/api/reports/models")
def model_metrics(context: AuthContext = Depends(current_context)):
    path = ROOT / "ml/reports/model_comparison.json"
    if not path.exists(): raise HTTPException(status_code=503, detail="Model evaluation report is not available")
    return json.loads(path.read_text())


@app.get("/api/reports/threshold")
def threshold_report(context: AuthContext = Depends(current_context)):
    path = ROOT / "ml/reports/threshold_evaluation.json"
    if not path.exists(): raise HTTPException(status_code=503, detail="Threshold report is not available; run model training")
    data = json.loads(path.read_text())
    configured = _predictor.threshold if _predictor is not None else None
    if configured is None:
        try: configured = ChurnPredictor().threshold
        except FileNotFoundError: configured = data["decision_threshold"]
    data["backend_active_threshold"] = configured
    data["active_threshold_source"] = "AURA_PREDICTION_THRESHOLD environment override" if configured != data["decision_threshold"] else "trained model metadata"
    return data


@app.get("/api/customers/{customer_id}")
def customer(customer_id: str, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    record = _customer_record(db, context.tenant.id, customer_id)
    latest = db.scalar(select(PredictionRecord).where(PredictionRecord.tenant_id==context.tenant.id,
        PredictionRecord.customer_pk==record.id).order_by(PredictionRecord.created_at.desc()).limit(1))
    return {**_customer_payload(record),"prediction":({"churn_probability":latest.probability,
        "decision_threshold":latest.decision_threshold,"predicted_churn":latest.predicted_churn,
        "risk_level":latest.risk_level,"model_version":latest.model_version,
        "prediction_timestamp":latest.created_at.isoformat()} if latest else None)}


@app.get("/api/customers/{customer_id}/events")
def customer_events(customer_id: str, seed: int = 42, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    record = _customer_record(db, context.tenant.id, customer_id)
    features = record.features
    events = simulate_events(customer_id, features, seed)
    return {"items": events, "count": len(events), "provenance": "simulated from aggregate features; original CSV contains no event timestamps"}


@app.get("/api/customers/{customer_id}/graph")
def customer_graph(customer_id: str, seed: int = 42, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    record = _customer_record(db, context.tenant.id, customer_id)
    features = record.features
    events = simulate_events(customer_id, features, seed)
    return build_customer_graph(customer_id, events)


@app.get("/api/actions")
def actions(customer_id: str | None = None, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    weights = context.tenant.settings_json.get("mcia_weights")
    if customer_id:
        record = _customer_record(db, context.tenant.id, customer_id)
        return rank_actions(record.features, weights)
    return rank_actions({}, weights)


def _start_workflow_for_customer(customer_id: str, context: AuthContext, db: Session):
    from backend.orchestrator.workflow import start_analysis
    record = _customer_record(db, context.tenant.id, customer_id)
    events = simulate_events(customer_id, record.features, seed=42)[:100]
    run_id = str(uuid4())
    run = AnalysisRun(id=run_id, tenant_id=context.tenant.id, customer_pk=record.id, status="running",
                      created_by=context.user.id, result={"stage":"starting"})
    db.add(run)
    for event in events:
        event_pk = str(uuid5(NAMESPACE_URL, f"{context.tenant.id}:{event['event_id']}"))
        if db.get(CustomerEvent, event_pk) is None:
            db.add(CustomerEvent(id=event_pk, tenant_id=context.tenant.id, customer_pk=record.id,
                event_type=event["event_type"], timestamp=datetime.fromisoformat(event["timestamp"]),
                payload=event["payload"], severity=event["severity"], provenance="simulated"))
    db.add(AuditLog(id=str(uuid4()), tenant_id=context.tenant.id, user_id=context.user.id,
        action="customer.analysis.started", resource_type="analysis_run", resource_id=run_id,
        details={"customer_id":customer_id,"event_provenance":"simulated"}))
    db.commit()
    session_factory = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    payload = AnalysisStart(analysis_run_id=run_id, tenant_id=context.tenant.id, user_id=context.user.id,
        customer_pk=record.id, customer_id=record.customer_id, features=record.features, events=events,
        tenant_settings=dict(context.tenant.settings_json or {}),
        mcia_weights=context.tenant.settings_json.get("mcia_weights"))
    try:
        state = start_analysis(payload, session_factory=session_factory)
    except Exception as exc:
        db.expire_all()
        failed = db.get(AnalysisRun, run_id)
        failed.status = "failed"; failed.error = type(exc).__name__; failed.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(500, "Analysis workflow failed; the run was recorded for review") from exc
    interrupted = bool(state.get("__interrupt__"))
    db.expire_all()
    stored_run = db.get(AnalysisRun, run_id)
    recs = db.scalars(select(Recommendation).where(Recommendation.tenant_id == context.tenant.id,
        Recommendation.analysis_run_id == run_id).order_by(Recommendation.created_at.desc())).all()
    return {"run_id":run_id,"status":"pending_approval" if interrupted else stored_run.status,
        "detect":state.get("detect"),"diagnose":state.get("diagnose"),"prediction":state.get("prediction"),
        "recommendations":[_recommendation_payload(item,db) for item in recs],
        "event_count":len(events),"event_provenance":"simulated",
        "interrupt":state.get("__interrupt__")[0].value if interrupted else None,
        "limitations":["No causal effect was estimated.","Event timestamps are simulated.","MCIA is relative utility, not retention probability or ROI."]}


def _recommendation_payload(row: Recommendation, db: Session):
    customer = db.get(Customer,row.customer_pk)
    prediction = db.scalar(select(PredictionRecord).where(PredictionRecord.tenant_id==row.tenant_id,
        PredictionRecord.customer_pk==row.customer_pk).order_by(PredictionRecord.created_at.desc()).limit(1))
    return {"id":row.id,"customer_id":customer.customer_id if customer else None,
        "risk_level":prediction.risk_level if prediction else "NOT_ANALYZED",
        "churn_probability":prediction.probability if prediction else None,
        "analysis_run_id":row.analysis_run_id,"action_id":row.action_id,"action_name":row.action_name,
        "rationale":row.rationale,"mcia":row.mcia,"eligibility":row.eligibility,"status":row.status,
        "created_at":row.created_at.isoformat()}


@app.post("/api/customers/{customer_id}/analyze", status_code=202)
def analyze_customer(customer_id: str, context: AuthContext = Depends(require_roles("ANALYST","APPROVER")),
                     db: Session = Depends(get_db)):
    return _start_workflow_for_customer(customer_id, context, db)


@app.post("/api/analysis/batch", status_code=200)
def batch_analysis(payload: BatchAnalysisRequest, context: AuthContext = Depends(require_roles("ANALYST","APPROVER")),
                   db: Session = Depends(get_db)):
    results = []
    for customer_id in payload.customer_ids:
        try: results.append({"customer_id":customer_id,"ok":True,**_start_workflow_for_customer(customer_id,context,db)})
        except HTTPException as exc: results.append({"customer_id":customer_id,"ok":False,"status_code":exc.status_code,"error":exc.detail})
        except Exception as exc: results.append({"customer_id":customer_id,"ok":False,"status_code":500,"error":type(exc).__name__})
    return {"results":results,"succeeded":sum(item["ok"] for item in results),"failed":sum(not item["ok"] for item in results)}


@app.get("/api/analysis/{run_id}")
def get_analysis(run_id: str, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    run = db.scalar(select(AnalysisRun).where(AnalysisRun.id == run_id, AnalysisRun.tenant_id == context.tenant.id))
    if run is None: raise HTTPException(404,"Analysis run not found")
    agents = db.scalars(select(AgentExecution).where(AgentExecution.tenant_id == context.tenant.id,
        AgentExecution.analysis_run_id == run.id).order_by(AgentExecution.started_at)).all()
    customer = db.get(Customer,run.customer_pk)
    return {"run_id":run.id,"customer_id":customer.customer_id if customer else None,"status":run.status,"result":run.result,
        "error":run.error,"created_at":run.created_at.isoformat(),"completed_at":run.completed_at.isoformat() if run.completed_at else None,
        "agent_executions":[{"agent":a.agent_name,"status":a.status,"output":a.output_json,"error":a.error,
            "started_at":a.started_at.isoformat() if a.started_at else None,
            "completed_at":a.completed_at.isoformat() if a.completed_at else None} for a in agents]}


@app.get("/api/analysis")
def list_analyses(limit: int = 50, offset: int = 0, context: AuthContext = Depends(current_context),
                  db: Session = Depends(get_db)):
    if not 1 <= limit <= 200 or offset < 0: raise HTTPException(422,"Invalid pagination")
    query=select(AnalysisRun).where(AnalysisRun.tenant_id == context.tenant.id)
    total=db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows=db.scalars(query.order_by(AnalysisRun.created_at.desc()).offset(offset).limit(limit)).all()
    return {"total":total,"items":[{"run_id":row.id,"customer_id":db.get(Customer,row.customer_pk).customer_id,
        "status":row.status,"created_at":row.created_at.isoformat(),"completed_at":row.completed_at.isoformat() if row.completed_at else None}
        for row in rows]}


@app.get("/api/agents/status")
def agent_status(context: AuthContext = Depends(current_context)):
    llm=LLMGateway().status(context.tenant.settings_json)
    return {"agents":[{"name":name,"mode":"model" if name=="Predict" else "deterministic" if name in ("Detect","Execute") or not llm["available"] else "llm-assisted",
        "available":True} for name in ("Detect","Diagnose","Predict","Strategy","Execute")],"llm_gateway":llm}


@app.get("/api/agents/runs/{run_id}")
def agent_run_status(run_id: str, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    run=db.scalar(select(AnalysisRun).where(AnalysisRun.id==run_id,AnalysisRun.tenant_id==context.tenant.id))
    if run is None: raise HTTPException(404,"Analysis run not found")
    records=db.scalars(select(AgentExecution).where(AgentExecution.tenant_id==context.tenant.id,
        AgentExecution.analysis_run_id==run_id).order_by(AgentExecution.started_at)).all()
    return {"run_id":run_id,"run_status":run.status,"agents":[{"name":r.agent_name,"status":r.status,
        "started_at":r.started_at.isoformat(),"completed_at":r.completed_at.isoformat() if r.completed_at else None,
        "output":r.output_json,"error":r.error} for r in records]}


@app.get("/api/recommendations")
def list_recommendations(limit: int = 50, offset: int = 0, status: str | None = None,
                         customer_id: str | None = None,
                         context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    if not 1 <= limit <= 200 or offset < 0: raise HTTPException(422,"Invalid pagination")
    query = select(Recommendation).where(Recommendation.tenant_id == context.tenant.id)
    if status: query = query.where(Recommendation.status == status)
    if customer_id:
        customer_row=db.scalar(select(Customer).where(Customer.tenant_id==context.tenant.id,
            Customer.customer_id==customer_id))
        if customer_row is None: raise HTTPException(404,"Customer not found")
        query=query.where(Recommendation.customer_pk==customer_row.id)
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(query.order_by(Recommendation.created_at.desc()).offset(offset).limit(limit)).all()
    return {"total":total,"items":[_recommendation_payload(row,db) for row in rows]}


@app.get("/api/recommendations/{recommendation_id}")
def get_recommendation(recommendation_id: str, context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    row = db.scalar(select(Recommendation).where(Recommendation.id == recommendation_id, Recommendation.tenant_id == context.tenant.id))
    if row is None: raise HTTPException(404,"Recommendation not found")
    approval = db.scalar(select(Approval).where(Approval.recommendation_id == row.id, Approval.tenant_id == context.tenant.id))
    execution = db.scalar(select(ActionExecution).where(ActionExecution.recommendation_id == row.id, ActionExecution.tenant_id == context.tenant.id))
    return {**_recommendation_payload(row,db),"approval":({"decision":approval.decision,"reviewer_id":approval.reviewer_id,
        "reason":approval.reason,"decided_at":approval.decided_at.isoformat()} if approval else None),
        "execution":({"status":execution.status,"simulated":execution.simulated,"result":execution.result,
        "executed_at":execution.executed_at.isoformat()} if execution else None)}


def _review_recommendation(recommendation_id: str, decision: str, note: str | None, context: AuthContext, db: Session):
    from backend.orchestrator.workflow import resume_analysis
    row = db.scalar(select(Recommendation).where(Recommendation.id == recommendation_id,
        Recommendation.tenant_id == context.tenant.id).with_for_update())
    if row is None: raise HTTPException(404,"Recommendation not found")
    if row.status != "pending_approval": raise HTTPException(409,f"Recommendation status is {row.status}")
    if not row.analysis_run_id: raise HTTPException(409,"Recommendation is not attached to a resumable analysis")
    approval = Approval(id=str(uuid4()), tenant_id=context.tenant.id, recommendation_id=row.id,
        reviewer_id=context.user.id, decision=decision, reason=note)
    row.status = "approved" if decision == "approved" else "rejected"
    db.add(approval)
    db.add(AuditLog(id=str(uuid4()),tenant_id=context.tenant.id,user_id=context.user.id,
        action=f"recommendation.{decision}",resource_type="recommendation",resource_id=row.id,
        details={"note":note,"analysis_run_id":row.analysis_run_id}))
    db.commit()
    session_factory = sessionmaker(bind=db.get_bind(),autoflush=False,expire_on_commit=False)
    try:
        state = resume_analysis(row.analysis_run_id, ApprovalResume(decision=decision,recommendation_id=row.id,
            approval_id=approval.id,reviewer_id=context.user.id,note=note), session_factory=session_factory)
    except Exception as exc:
        db.expire_all(); row=db.get(Recommendation,recommendation_id)
        if decision == "approved": row.status="execution_failed"
        run=db.get(AnalysisRun,row.analysis_run_id); run.status="failed"; run.error=type(exc).__name__
        db.commit()
        raise HTTPException(500,"Approval was recorded but workflow resume failed; inspect the run history") from exc
    db.expire_all()
    row = db.get(Recommendation,recommendation_id)
    return {"recommendation_id":row.id,"decision":decision,"status":row.status,
        "execution":state.get("execution"),"workflow_status":state.get("status"),
        "awaiting_more_reviews":bool(state.get("__interrupt__")),
        "message":"Execution is simulated; customer retention outcome remains unknown." if decision=="approved" else "Recommendation rejected; no execution occurred."}


@app.post("/api/recommendations/{recommendation_id}/approve")
def approve_recommendation(recommendation_id: str, payload: ReviewRequest,
    context: AuthContext = Depends(require_roles("APPROVER","OWNER","ADMIN")), db: Session = Depends(get_db)):
    return _review_recommendation(recommendation_id,"approved",payload.note,context,db)


@app.post("/api/recommendations/{recommendation_id}/reject")
def reject_recommendation(recommendation_id: str, payload: ReviewRequest,
    context: AuthContext = Depends(require_roles("APPROVER","OWNER","ADMIN")), db: Session = Depends(get_db)):
    return _review_recommendation(recommendation_id,"rejected",payload.note,context,db)


@app.post("/api/recommendations/{recommendation_id}/execute")
def execute_recommendation(recommendation_id: str, context: AuthContext = Depends(require_roles("APPROVER","OWNER","ADMIN")),
                           db: Session = Depends(get_db)):
    row = db.scalar(select(Recommendation).where(Recommendation.id == recommendation_id, Recommendation.tenant_id == context.tenant.id))
    if row is None: raise HTTPException(404,"Recommendation not found")
    execution = db.scalar(select(ActionExecution).where(ActionExecution.recommendation_id == row.id,
        ActionExecution.tenant_id == context.tenant.id))
    if execution: return {"execution_id":execution.id,"status":execution.status,"simulated":execution.simulated,
                         "idempotent_replay":True,"result":execution.result}
    raise HTTPException(409,"Execution is only resumed by the approval checkpoint; no approved execution record exists")


@app.post("/api/interventions/{intervention_id}/outcome", status_code=201)
def record_outcome(intervention_id: str, payload: OutcomeRequest,
    context: AuthContext = Depends(require_roles("ANALYST","APPROVER","OWNER","ADMIN")), db: Session = Depends(get_db)):
    if payload.outcome_status not in {"retained","churned","unknown","no_response"}:
        raise HTTPException(422,"outcome_status must be retained, churned, unknown, or no_response")
    execution = db.scalar(select(ActionExecution).where(ActionExecution.tenant_id == context.tenant.id,
        (ActionExecution.id == intervention_id) | (ActionExecution.recommendation_id == intervention_id)))
    if execution is None: raise HTTPException(409,"An intervention outcome requires a recorded execution")
    recommendation = db.scalar(select(Recommendation).where(Recommendation.id == execution.recommendation_id,
        Recommendation.tenant_id == context.tenant.id))
    if recommendation is None: raise HTTPException(404,"Recommendation not found")
    outcome = OutcomeFeedback(id=str(uuid4()),tenant_id=context.tenant.id,recommendation_id=recommendation.id,
        customer_pk=recommendation.customer_pk,outcome_status=payload.outcome_status,customer_response=payload.customer_response,
        notes=payload.notes,follow_up_status=payload.follow_up_status,recorded_by=context.user.id)
    db.add(outcome)
    db.add(AuditLog(id=str(uuid4()),tenant_id=context.tenant.id,user_id=context.user.id,
        action="intervention.outcome_recorded",resource_type="recommendation",resource_id=recommendation.id,
        details={"outcome_status":payload.outcome_status}))
    db.commit()
    return {"outcome_id":outcome.id,"outcome_status":outcome.outcome_status,"recorded_at":outcome.feedback_at.isoformat(),
            "note":"This is manually recorded follow-up, not a model-verified causal intervention effect."}


@app.get("/api/interventions")
def list_interventions(limit: int = 50, offset: int = 0, context: AuthContext = Depends(current_context),
                       db: Session = Depends(get_db)):
    if not 1 <= limit <= 200 or offset < 0: raise HTTPException(422,"Invalid pagination")
    query=select(ActionExecution).where(ActionExecution.tenant_id==context.tenant.id)
    total=db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows=db.scalars(query.order_by(ActionExecution.executed_at.desc()).offset(offset).limit(limit)).all()
    items=[]
    for execution in rows:
        rec=db.get(Recommendation,execution.recommendation_id)
        customer=db.get(Customer,rec.customer_pk) if rec else None
        feedback=db.scalars(select(OutcomeFeedback).where(OutcomeFeedback.tenant_id==context.tenant.id,
            OutcomeFeedback.recommendation_id==rec.id).order_by(OutcomeFeedback.feedback_at.desc())).all() if rec else []
        items.append({"execution_id":execution.id,"recommendation_id":execution.recommendation_id,
            "customer_id":customer.customer_id if customer else None,"action_name":rec.action_name if rec else None,
            "status":execution.status,"simulated":execution.simulated,"result":execution.result,
            "executed_at":execution.executed_at.isoformat(),"outcomes":[{"status":item.outcome_status,
                "feedback_at":item.feedback_at.isoformat(),"notes":item.notes} for item in feedback]})
    return {"total":total,"items":items}


@app.get("/api/reports/overview")
def report_overview(context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    customer_count=db.scalar(select(func.count(Customer.id)).where(Customer.tenant_id==context.tenant.id)) or 0
    run_count=db.scalar(select(func.count(AnalysisRun.id)).where(AnalysisRun.tenant_id==context.tenant.id)) or 0
    recommendation_count=db.scalar(select(func.count(Recommendation.id)).where(Recommendation.tenant_id==context.tenant.id)) or 0
    execution_count=db.scalar(select(func.count(ActionExecution.id)).where(ActionExecution.tenant_id==context.tenant.id)) or 0
    outcomes=db.scalars(select(OutcomeFeedback.outcome_status).where(OutcomeFeedback.tenant_id==context.tenant.id)).all()
    outcome_counts={key:outcomes.count(key) for key in sorted(set(outcomes))}
    rec_statuses=db.execute(select(Recommendation.status,func.count(Recommendation.id)).where(
        Recommendation.tenant_id==context.tenant.id).group_by(Recommendation.status)).all()
    return {"customers":customer_count,"analysis_runs":run_count,"recommendations":recommendation_count,
        "simulated_executions":execution_count,"recorded_outcomes":outcome_counts,
        "recommendation_status_distribution":dict(rec_statuses),
        "retention_claim":"No retention or ROI inference is made from execution counts."}


@app.get("/api/audit")
def list_audit(limit: int = 50, offset: int = 0, action: str | None = None,
               context: AuthContext = Depends(current_context), db: Session = Depends(get_db)):
    if not 1 <= limit <= 200 or offset < 0: raise HTTPException(422,"Invalid pagination")
    query=select(AuditLog).where(AuditLog.tenant_id==context.tenant.id)
    if action: query=query.where(AuditLog.action==action)
    total=db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows=db.scalars(query.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit)).all()
    return {"total":total,"items":[{"id":row.id,"action":row.action,"resource_type":row.resource_type,
        "resource_id":row.resource_id,"details":row.details,"created_at":row.created_at.isoformat()} for row in rows]}
