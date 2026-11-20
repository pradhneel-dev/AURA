from streamlit.testing.v1 import AppTest
from pathlib import Path
import json
from uuid import uuid4
from types import SimpleNamespace
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from backend.main import app as fastapi_app
from backend.database.models import Base, Customer
from backend.database.session import get_db
from ml.inference.predict import ROOT


def test_streamlit_login_view_renders_without_backend():
    app=AppTest.from_file(Path(__file__).resolve().parents[2]/"frontend"/"app.py").run(timeout=10)
    assert not app.exception
    assert any(element.value == "AURA" for element in app.title)
    assert any(element.label == "Account" for element in app.radio)


def test_authenticated_dashboard_and_customer_search_render_with_live_api(client, monkeypatch):
    import frontend.api as frontend_api
    owner=client.test_client.post("/api/auth/signup",json={"email":"ui@example.test","password":"correct horse battery",
        "full_name":"UI Tester","organization_name":"UI Workspace"}).json()
    row=pd.read_csv(ROOT/"data/raw/AURA_Synthetic_Customer_Retention_Dataset.csv").iloc[0]
    features=json.loads(row.drop(labels=["customer_id","churn"]).to_json())
    with client.factory() as db:
        db.add(Customer(id=str(uuid4()),tenant_id=owner["tenant"]["id"],customer_id=str(row.customer_id),
            segment=str(features["customer_segment"]),features=features)); db.commit()
    def bridge(method,url,**kwargs):
        kwargs.pop("timeout",None)
        response=client.test_client.request(method,url.replace("http://127.0.0.1:8000",""),**kwargs)
        response.ok=response.is_success
        return response
    monkeypatch.setattr(frontend_api.requests,"request",bridge)
    path=Path(__file__).resolve().parents[2]/"frontend"/"app.py"
    app_test=AppTest.from_file(path)
    app_test.session_state["access_token"]=owner["access_token"]
    app_test.run(timeout=15)
    assert not app_test.exception
    assert any("Total customers" in element.label for element in app_test.metric)
    app_test.session_state["active_page"]="Customers"
    app_test.run(timeout=15)
    assert not app_test.exception
    assert any(str(row.customer_id) in str(item.value) for item in app_test.dataframe)
    open_profile=next(button for button in app_test.button if button.label=="Open customer profile")
    open_profile.click().run(timeout=15)
    assert not app_test.exception
    assert any(element.value=="Customer profile" for element in app_test.title)
    run_button=next(button for button in app_test.button if button.label=="Run five-agent analysis")
    run_button.click().run(timeout=20)
    assert not app_test.exception
    assert any("Workflow status" in str(markdown.value) for markdown in app_test.markdown)
    app_test.session_state["active_page"]="Retention center"
    app_test.run(timeout=15)
    assert not app_test.exception
    review=next(box for box in app_test.checkbox if box.label=="I reviewed this simulated recommendation")
    approve=next(button for button in app_test.button if button.label=="Approve and simulate")
    assert not approve.disabled, "Form buttons must remain submit-capable; confirmation is enforced after submit."
    approve.click().run(timeout=15)
    assert any("Confirm that you reviewed" in str(item.value) for item in app_test.warning)
    assert not app_test.exception
    review=next(box for box in app_test.checkbox if box.label=="I reviewed this simulated recommendation")
    review.check().run(timeout=15)
    approve=next(button for button in app_test.button if button.label=="Approve and simulate")
    approve.click().run(timeout=20)
    assert not app_test.exception
    for page in ("Overview","Customers","Customer profile","Model report","Causal explorer",
                 "Retention center","History & activity","Datasets","Reports","Settings"):
        app_test.session_state["active_page"]=page
        app_test.run(timeout=20)
        assert not app_test.exception, f"Streamlit page failed: {page}; {app_test.exception}"


@pytest.fixture
def client():
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    @event.listens_for(engine,"connect")
    def fk(connection,_record):
        cursor=connection.cursor(); cursor.execute("PRAGMA foreign_keys=ON"); cursor.close()
    Base.metadata.create_all(engine)
    factory=sessionmaker(bind=engine,autoflush=False,expire_on_commit=False)
    def override_db():
        session=factory()
        try: yield session
        finally: session.close()
    fastapi_app.dependency_overrides[get_db]=override_db
    with TestClient(fastapi_app) as test_client:
        yield SimpleNamespace(factory=factory,test_client=test_client)
    fastapi_app.dependency_overrides.clear(); Base.metadata.drop_all(engine); engine.dispose()
