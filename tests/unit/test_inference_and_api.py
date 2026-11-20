import json
from uuid import uuid4
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import create_engine

from backend.main import app
from backend.database.models import Base, Customer, AuditLog
from backend.database.session import get_db
from ml.inference.predict import ChurnPredictor, ROOT
from backend.services.event_simulator import simulate_events
from backend.graph_engine.customer_graph import build_customer_graph
from backend.mcia.optimizer import rank_actions
from ml.evaluation.thresholds import select_operational_threshold, threshold_metrics

TEST_FACTORY = None


@pytest.fixture
def client():
    global TEST_FACTORY
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        cursor = connection.cursor(); cursor.execute("PRAGMA foreign_keys=ON"); cursor.close()
    Base.metadata.create_all(engine)
    TEST_FACTORY = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    def override_db():
        session = TEST_FACTORY()
        try: yield session
        finally: session.close()
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client: yield test_client
    app.dependency_overrides.clear(); Base.metadata.drop_all(engine); engine.dispose(); TEST_FACTORY = None


def signup(client, email=None):
    response = client.post("/api/auth/signup", json={"email": email or f"user-{uuid4()}@example.test",
        "password": "correct horse battery", "full_name": "Demo Owner", "organization_name": "Demo Org"})
    assert response.status_code == 201, response.text
    token_data = response.json()
    client.headers.update({"Authorization": f"Bearer {token_data['access_token']}"})
    return token_data


def seed_customer(client, customer_id="CUST_00001"):
    token_data = signup(client)
    row = pd.read_csv(ROOT / "data/raw/AURA_Synthetic_Customer_Retention_Dataset.csv").set_index("customer_id").loc[customer_id]
    features = json.loads(row.drop(labels=["churn"]).to_json())
    with TEST_FACTORY() as db:
        db.add(Customer(id=str(uuid4()), tenant_id=token_data["tenant"]["id"], customer_id=customer_id,
                        segment=str(features["customer_segment"]), features=features))
        db.commit()
    return token_data, features


def test_saved_model_predicts_and_rejects_bad_schema():
    predictor = ChurnPredictor()
    sample = pd.read_csv(ROOT / "data/raw/AURA_Synthetic_Test.csv").iloc[0]
    features = json.loads(sample.drop(labels=["customer_id", "churn"]).to_json())
    result = predictor.predict(sample.customer_id, features)
    assert 0 <= result["churn_probability"] <= 1
    assert result["model_version"] == "churn-v1"
    with pytest.raises(ValueError): predictor.predict(sample.customer_id, {"age": 33})
    with pytest.raises(ValueError): predictor.predict(sample.customer_id, {**features, "churn": 1})


def test_blank_threshold_env_falls_back_and_invalid_values_are_actionable(monkeypatch):
    monkeypatch.setenv("AURA_PREDICTION_THRESHOLD", "   ")
    predictor = ChurnPredictor()
    assert predictor.threshold == predictor.metadata["threshold"]
    monkeypatch.setenv("AURA_PREDICTION_THRESHOLD", "not-a-number")
    with pytest.raises(ValueError, match="AURA_PREDICTION_THRESHOLD must be a number"):
        ChurnPredictor()
    monkeypatch.setenv("AURA_PREDICTION_THRESHOLD", "nan")
    with pytest.raises(ValueError, match="finite number"):
        ChurnPredictor()


def test_prediction_rejects_empty_numeric_feature_with_feature_name():
    sample = pd.read_csv(ROOT / "data/raw/AURA_Synthetic_Test.csv").iloc[0]
    features = json.loads(sample.drop(labels=["customer_id", "churn"]).to_json())
    features["age"] = ""
    with pytest.raises(ValueError, match="Invalid numeric feature: age"):
        ChurnPredictor().predict(sample.customer_id, features)


def test_prediction_api_uses_saved_model_and_tenant_record(client):
    _, features = seed_customer(client)
    response = client.post("/api/predictions", json={"customer_id": "CUST_00001", "features": features})
    assert response.status_code == 200, response.text
    assert 0 <= response.json()["churn_probability"] <= 1
    assert response.json()["model_version"] == "churn-v1"
    assert response.json()["decision_threshold"] == ChurnPredictor().threshold


def test_prediction_api_rejects_incomplete_features(client):
    seed_customer(client)
    response = client.post("/api/predictions", json={"customer_id": "CUST_00001", "features": {"age": 22}})
    assert response.status_code == 422


def test_dashboard_metrics_are_tenant_scoped_and_source_calculated(client):
    seed_customer(client)
    response = client.get("/api/dashboard/overview")
    assert response.status_code == 200
    assert response.json()["total_customers"] == 1
    assert response.json()["analyzed_customers"] == 0


def test_seeded_event_simulator_and_graph_are_explicitly_simulated():
    customer_id = "CUST_TEST"
    features = {"payment_failures_90d": 2, "support_tickets_30d": 1, "api_error_rate": 7,
                "login_frequency_30d": 16}
    events_a = simulate_events(customer_id, features, seed=11)
    events_b = simulate_events(customer_id, features, seed=11)
    assert events_a == events_b
    assert all(event["provenance"].startswith("simulated") for event in events_a)
    graph = build_customer_graph(customer_id, events_a)
    assert graph["nodes"][0]["kind"] == "customer"
    assert graph["edges"]
    assert "simulated" in graph["provenance"]


def test_mcia_scores_normalize_and_rank_without_claiming_probability():
    ranked = rank_actions({})
    assert ranked == sorted(ranked, key=lambda item: item["mcia_score"], reverse=True)
    assert all("not retention probability" in item["score_interpretation"] for item in ranked)
    with pytest.raises(ValueError): rank_actions({}, {"money": 0, "time": 0, "relation": 0, "feasibility": 0})


def test_analysis_api_returns_prediction_graph_and_candidate_actions(client):
    seed_customer(client)
    response = client.post("/api/customers/CUST_00001/analyze")
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["prediction"]["model_version"] == "churn-v1"
    assert result["event_provenance"] == "simulated"
    assert result["recommendations"]
    assert result["run_id"]
    assert "No causal effect was estimated." in result["limitations"]
    profile=client.get("/api/customers/CUST_00001").json()
    assert profile["prediction"]["model_version"] == "churn-v1"


def test_customer_risk_filter_sort_and_audit_are_tenant_scoped(client):
    first, _=seed_customer(client)
    client.post("/api/customers/CUST_00001/analyze")
    filtered=client.get("/api/customers",params={"risk_level":"NOT_ANALYZED"})
    assert filtered.status_code == 200 and filtered.json()["total"] == 0
    risk=client.get("/api/customers",params={"sort_by":"churn_probability","descending":"true"})
    assert risk.status_code == 200 and risk.json()["items"][0]["risk_level"] in {"CRITICAL","HIGH","MEDIUM","LOW"}
    first_audit=client.get("/api/audit")
    assert first_audit.status_code == 200 and first_audit.json()["total"] > 0
    second=signup(client,email=f"other-{uuid4()}@example.test")
    assert client.get("/api/audit",headers={"Authorization":f"Bearer {second['access_token']}"}).json()["total"] == 0


def test_threshold_selection_respects_capacity_and_reports_balanced_metrics():
    y = [0, 0, 0, 1, 1, 1]
    oof = [0.05, 0.20, 0.35, 0.40, 0.65, 0.90]
    threshold, table, rationale = select_operational_threshold(y, oof, capacity_rate=0.5,
                                                                thresholds=[0.3, 0.5, 0.7])
    metrics = threshold_metrics(y, oof, threshold)
    assert table.loc[table.threshold.eq(threshold), "flagged_rate"].iloc[0] <= 0.5
    assert "training-only out-of-fold" in rationale
    assert set(("precision", "recall", "f1", "specificity", "false_positive_rate", "false_negative_rate", "confusion_matrix")) <= set(metrics)


def test_threshold_report_and_configured_prediction_threshold(client):
    signup(client)
    report = client.get("/api/reports/threshold")
    assert report.status_code == 200
    data = report.json()
    assert data["selection_data"] == "training-only out-of-fold predictions"
    assert data["decision_threshold"] == ChurnPredictor().threshold
    assert data["held_out_test_metrics"]["confusion_matrix"]


def test_auth_password_login_logout_and_tenant_isolation(client):
    first, _ = seed_customer(client)
    first_token = first["access_token"]
    login_response = client.post("/api/auth/login", json={"email": first["user"]["email"], "password": "correct horse battery"})
    assert login_response.status_code == 200
    assert client.post("/api/auth/login", json={"email": first["user"]["email"], "password": "wrong password"}).status_code == 401
    second = signup(client, email=f"other-{uuid4()}@example.test")
    response = client.get("/api/customers/CUST_00001", headers={"Authorization": f"Bearer {second['access_token']}"})
    assert response.status_code == 404
    logout_response = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {first_token}"})
    assert logout_response.status_code == 200
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {first_token}"}).status_code == 401


def test_protected_customer_api_requires_authentication(client):
    assert client.get("/api/customers").status_code == 401
    assert client.get("/api/reports/models").status_code == 401


def test_api_returns_request_id_header_and_unhandled_error_body(client, monkeypatch):
    _, features = seed_customer(client)
    monkeypatch.setattr("backend.main.ChurnPredictor", lambda: (_ for _ in ()).throw(RuntimeError("diagnostic probe")))
    monkeypatch.setattr("backend.main._predictor", None)
    response = client.get("/api/reports/threshold", headers={"X-Request-ID": "audit-probe-123"})
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "audit-probe-123"
    assert response.json() == {"detail": "Internal server error", "request_id": "audit-probe-123"}


def test_pdf_report_is_authenticated_and_tenant_scoped(client):
    signup(client)
    report=client.get("/api/reports/tenant.pdf")
    assert report.status_code == 200
    assert report.headers["content-type"] == "application/pdf"
    assert report.content.startswith(b"%PDF")
    assert client.get("/api/reports/tenant.pdf",headers={"Authorization":""}).status_code == 401


def test_websocket_event_feed_rejects_invalid_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/events?token=not-a-token"):
            pass


def test_websocket_stream_is_tenant_isolated(client, monkeypatch):
    first,_=seed_customer(client)
    second=signup(client,email=f"other-{uuid4()}@example.test")
    monkeypatch.setattr("backend.main.SessionLocal",TEST_FACTORY)
    with client.websocket_connect(f"/api/events?token={first['access_token']}") as socket:
        with TEST_FACTORY() as db:
            db.add_all([
                AuditLog(id=str(uuid4()),tenant_id=first["tenant"]["id"],action="tenant.one",
                    resource_type="test",resource_id="private-one",details={}),
                AuditLog(id=str(uuid4()),tenant_id=second["tenant"]["id"],action="tenant.two",
                    resource_type="test",resource_id="private-two",details={})])
            db.commit()
        event=socket.receive_json()
        assert event["resource_id"] == "private-one"


def test_analyst_cannot_approve_recommendation(client):
    owner,_=seed_customer(client)
    started=client.post("/api/customers/CUST_00001/analyze").json()
    recommendation=started["recommendations"][0]
    invited=client.post("/api/tenant/users",json={"email":"review-analyst@example.test","full_name":"Analyst",
        "password":"secure analyst password","roles":["ANALYST"]})
    assert invited.status_code == 201
    analyst=client.post("/api/auth/login",json={"email":"review-analyst@example.test","password":"secure analyst password"}).json()
    denied=client.post(f"/api/recommendations/{recommendation['id']}/approve",
        headers={"Authorization":f"Bearer {analyst['access_token']}"},json={"note":"must be denied"})
    assert denied.status_code == 403
    assert client.get(f"/api/recommendations/{recommendation['id']}",headers={"Authorization":f"Bearer {owner['access_token']}"}).json()["status"] == "pending_approval"
    other=signup(client,email=f"isolated-{uuid4()}@example.test")
    assert client.get(f"/api/recommendations/{recommendation['id']}",headers={"Authorization":f"Bearer {other['access_token']}"}).status_code == 404
    assert client.post(f"/api/recommendations/{recommendation['id']}/approve",
        headers={"Authorization":f"Bearer {other['access_token']}"},json={}).status_code == 404


def test_agent_failure_is_persisted_on_run_and_execution(client,monkeypatch):
    seed_customer(client)
    monkeypatch.setattr("backend.agents.predict.PredictAgent.run",
        lambda *_args,**_kwargs: (_ for _ in ()).throw(RuntimeError("predictor unavailable")))
    response=client.post("/api/customers/CUST_00001/analyze")
    assert response.status_code == 500
    with TEST_FACTORY() as db:
        from backend.database.models import AnalysisRun,AgentExecution
        run=db.query(AnalysisRun).order_by(AnalysisRun.created_at.desc()).first()
        records=db.query(AgentExecution).filter(AgentExecution.analysis_run_id==run.id).all()
        assert run.status == "failed"
        predict=next(record for record in records if record.agent_name=="Predict")
        assert predict.status == "failed" and "RuntimeError" in predict.error


def test_dataset_upload_validates_persists_and_does_not_retrain_model(client):
    signup(client)
    frame = pd.read_csv(ROOT / "data/raw/AURA_Synthetic_Customer_Retention_Dataset.csv").head(3)
    payload = frame.to_csv(index=False).encode()
    validated = client.post("/api/datasets/validate", files={"file": ("customers.csv", payload, "text/csv")})
    assert validated.status_code == 200, validated.text
    assert validated.json()["validation"]["valid"] is True
    assert len(validated.json()["preview"]) == 3
    uploaded = client.post("/api/datasets/upload", files={"file": ("customers.csv", payload, "text/csv")})
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["row_count"] == 3
    assert uploaded.json()["model_retrained"] is False
    customers = client.get("/api/customers", params={"limit": 10}).json()
    assert customers["total"] == 3
    assert "churn" not in customers["items"][0]


def test_supplied_demo_import_persists_all_12000_rows_once(client):
    signup(client)
    imported = client.post("/api/datasets/demo")
    assert imported.status_code == 201, imported.text
    assert imported.json()["row_count"] == 12000
    assert client.get("/api/customers", params={"limit": 2}).json()["total"] == 12000
    second = client.post("/api/datasets/demo")
    assert second.status_code == 201
    assert second.json()["already_imported"] is True


def test_owner_can_invite_role_scoped_user_and_update_tenant_mcia(client):
    signup(client)
    invited = client.post("/api/tenant/users", json={"email":"analyst@example.test", "full_name":"Analyst",
        "password":"secure analyst password", "roles":["ANALYST"]})
    assert invited.status_code == 201, invited.text
    listing = client.get("/api/tenant/users")
    assert listing.status_code == 200
    assert any(item["email"] == "analyst@example.test" for item in listing.json()["items"])
    updated = client.put("/api/settings/mcia", json={"money":3,"time":2,"relation":4,"feasibility":1})
    assert updated.status_code == 200
    assert updated.json()["mcia_weights"]["money"] == 3
    analyst = client.post("/api/auth/login", json={"email":"analyst@example.test", "password":"secure analyst password"}).json()
    analyst_headers={"Authorization":f"Bearer {analyst['access_token']}"}
    assert client.get("/api/tenant/users",headers=analyst_headers).status_code == 403
    assert client.put("/api/settings/mcia",headers=analyst_headers,json={"money":1,"time":1,"relation":1,"feasibility":1}).status_code == 403


def test_langgraph_approval_checkpoint_simulated_execution_and_outcome(client):
    seed_customer(client)
    started = client.post("/api/customers/CUST_00001/analyze")
    assert started.status_code == 202, started.text
    run = started.json()
    assert run["status"] == "pending_approval"
    assert run["interrupt"]["recommendation_ids"]
    recommendation = run["recommendations"][0]
    pending_run = client.get(f"/api/analysis/{run['run_id']}")
    assert pending_run.status_code == 200
    assert {x["agent"] for x in pending_run.json()["agent_executions"]} >= {"Detect","Diagnose","Predict","Strategy"}
    gate=next(x for x in pending_run.json()["agent_executions"] if x["agent"]=="HumanApproval")
    assert gate["status"] == "awaiting_approval"
    approved = client.post(f"/api/recommendations/{recommendation['id']}/approve",json={"note":"Approved for simulation"})
    assert approved.status_code == 200, approved.text
    assert approved.json()["execution"]["simulated"] is True
    assert "no CRM" in approved.json()["execution"]["result"]["message"]
    detail=client.get(f"/api/recommendations/{recommendation['id']}")
    assert detail.json()["status"] == "executed"
    assert detail.json()["execution"]["result"]["customer_retention_outcome"].startswith("unknown")
    replay=client.post(f"/api/recommendations/{recommendation['id']}/execute")
    assert replay.status_code == 200 and replay.json()["idempotent_replay"] is True
    outcome=client.post(f"/api/interventions/{recommendation['id']}/outcome",json={"outcome_status":"unknown","notes":"No response yet"})
    assert outcome.status_code == 201
    assert "not a model-verified" in outcome.json()["note"]


def test_pending_or_rejected_recommendation_cannot_execute(client):
    seed_customer(client)
    start_response=client.post("/api/customers/CUST_00001/analyze")
    assert start_response.status_code == 202, start_response.text
    started=start_response.json()
    recommendation=started["recommendations"][0]
    assert client.post(f"/api/recommendations/{recommendation['id']}/execute").status_code == 409
    rejected=client.post(f"/api/recommendations/{recommendation['id']}/reject",json={"note":"Not suitable"})
    run_detail = client.get(f"/api/analysis/{started['run_id']}").json()
    assert rejected.status_code == 200, f"{rejected.text} run_error={run_detail['error']}"
    assert rejected.json()["decision"] == "rejected"
    assert client.post(f"/api/recommendations/{recommendation['id']}/execute").status_code == 409


def test_token_expiration_is_enforced(client):
    from datetime import datetime, timedelta, timezone
    from jose import jwt
    from backend.security.auth import JWT_SECRET, ALGORITHM
    expired=jwt.encode({"sub":"missing","tid":"missing","exp":datetime.now(timezone.utc)-timedelta(minutes=5)},
        JWT_SECRET,algorithm=ALGORITHM)
    response=client.get("/api/auth/me",headers={"Authorization":f"Bearer {expired}"})
    assert response.status_code == 401
