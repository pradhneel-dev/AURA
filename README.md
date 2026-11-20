# AURA — AI-Powered User Retention Analytics

AURA is a local-first academic customer-retention intelligence prototype. It combines a trained churn-risk model, customer feature inspection, explicitly simulated event timelines, NetworkX graphs, and transparent MCIA decision support. The supplied data is synthetic and does not establish real-world model validity or causal effects.

## Current implementation

Implemented: supplied data validation; three trained and compared models; training-only OOF threshold selection; FastAPI APIs; SQLAlchemy persistence and Alembic migrations; password-hashed signup/login/logout and role controls; tenant-scoped records; five-agent persistent LangGraph workflow with actual human interrupt; optional centralized LLM gateway and deterministic fallback; recommendation approval/rejection and safe simulated execution; dataset upload; authenticated event WebSocket; redesigned Streamlit workspace with ten pages; tenant CSV/PDF reports; Docker Compose configuration. The frontend includes customer search, server-side filtering/sorting, profiles, persisted workflow traces, activity history, reports, and tenant settings. See [PROGRESS.md](PROGRESS.md) for limitations and verification.

## Dataset

The four supplied CSVs in `data/raw/` remain unchanged. The primary dataset has 12,000 rows, 36 columns, 34 dictionary-defined input features, identifier `customer_id`, and target `churn`. No other churn dataset is included. See [dataset documentation](docs/DATASET_DOCUMENTATION.md) and run `python scripts/validate_data.py`.

## ML methodology and results

Logistic Regression, Random Forest, and XGBoost are trained with preprocessing inside fitted sklearn pipelines. Three-fold stratified CV on the 9,600-row training split selects hyperparameters and the model. XGBoost was selected by CV ROC-AUC (0.6278). The 2,400-row supplied test split is held for final evaluation.

At fixed 0.50 threshold, selected XGBoost test metrics are accuracy 0.7108, precision 0.5108, recall 0.1019, F1 0.1699, ROC-AUC 0.6475, PR-AUC 0.4181. Its recall is low. Separately, an operational threshold of 0.38 was selected from training-only out-of-fold predictions by highest F1 under an explicit 20% contact-capacity constraint. Held-out test results at 0.38: precision 0.4922, recall 0.3156, F1 0.3846, specificity 0.8667, flagged 18.63%. This remains a synthetic-data demonstration, not an operational guarantee. Full reports are in `ml/reports/`.

## Architecture

- **Frontend:** Streamlit, authenticated API calls, Plotly/Streamlit tables.
- **Backend:** FastAPI, Pydantic request validation, bearer tokens, tenant-scoped services.
- **Persistence:** SQLAlchemy 2, SQLite locally; PostgreSQL URL support; Alembic baseline in `alembic/`.
- **ML:** scikit-learn pipelines and XGBoost, saved artifact in `ml/artifacts/`.
- **Events/graph/actions:** seeded simulated events, NetworkX graph, normalized M/T/R/F utility scores.

Database entities include tenants, users, memberships, datasets, customers, events, model metadata, predictions, analysis runs, agent executions, recommendations, approvals, action executions, feedback, audit logs, and LLM usage.

## Install and run locally

On macOS, install OpenMP for XGBoost if needed (`brew install libomp`). Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set a strong `AURA_SECRET_KEY` in `.env` before running if login sessions should survive backend restarts. Apply the database schema and validate the source files:

```bash
alembic upgrade head
python scripts/validate_data.py
```

Train/rebuild model artifacts when desired:

```bash
python -m ml.training.train
```

Run API and Streamlit in separate terminals:

```bash
uvicorn backend.main:app --reload --env-file .env
streamlit run frontend/app.py
```

Create an organization account in the UI, then choose **Load supplied AURA demonstration dataset** on the dashboard. User-uploaded customer CSVs are validated and imported to that tenant; they do not retrain the model.

Set `AURA_DATABASE_URL=sqlite:///./data/aura.db` for local storage. PostgreSQL can use `postgresql+psycopg://user:password@host:5432/aura`. Optional threshold settings: `AURA_CONTACT_CAPACITY_RATE` at training time (default `0.20`) and `AURA_PREDICTION_THRESHOLD` as a server-side inference override.

## API and workflow notes

FastAPI OpenAPI documentation is available at `/docs`. Protected requests use `Authorization: Bearer <token>`. Signup creates a tenant owner. Customer analysis runs Detect → Diagnose → Predict → Strategy and pauses at a persistent LangGraph approval interrupt. Approvers can approve or reject; approval triggers an idempotent simulation after tenant, membership, and approval checks. No external system is contacted. The graph is observational/simulated; MCIA is relative utility, not churn probability, causal effect, or ROI.

The tenant-authenticated WebSocket endpoint is `/api/events?token=<bearer-token>` and sends tenant-scoped status metadata. Streamlit currently uses HTTP refresh/polling. The PDF export is an activity summary; reports make no retention or ROI claim.

Read the [AURA user guide](docs/USER_GUIDE.md) for the in-app navigation and project demonstration workflow.

## Tests

```bash
python -m pytest -q
python -m compileall -q backend frontend ml scripts tests
```

## Documentation

- [Implementation progress](PROGRESS.md)
- [System architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Dataset](docs/DATASET_DOCUMENTATION.md)
- [ML methodology](docs/ML_METHODOLOGY.md)
- [Database schema](docs/DATABASE_SCHEMA.md)
- [API](docs/API_DOCUMENTATION.md)
- [Security status](docs/SECURITY_DOCUMENTATION.md)
- [Agent architecture status](docs/AGENT_ARCHITECTURE.md)
- [Event graph](docs/DYNAMIC_GRAPH_DOCUMENTATION.md)
- [Causal analysis status](docs/CAUSAL_ANALYSIS_DOCUMENTATION.md)
- [MCIA](docs/MCIA_DOCUMENTATION.md)
- [Testing](docs/TESTING_REPORT.md)
- [Deployment](docs/DEPLOYMENT_GUIDE.md)

## Docker deployment

Copy `.env.example` to `.env`, set a strong `AURA_SECRET_KEY` and `POSTGRES_PASSWORD`, then run:

```bash
docker compose up --build
```

The API container applies Alembic migrations at startup. Open Streamlit at `http://localhost:8501` and API docs at `http://localhost:8000/docs`. Docker Compose is configured but was not exercised in local verification; see [deployment notes](docs/DEPLOYMENT_GUIDE.md).
