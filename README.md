# AURA — AI-Powered User Retention Analytics

AURA is a local-first academic customer-retention intelligence prototype. It combines a trained churn-risk model, customer-feature inspection, deterministic demonstration timelines, NetworkX-based signal graphs, and MCIA-based retention decision support in a coordinated five-agent workflow.

AURA is designed for **human-reviewed decision support**. It does not autonomously contact customers or modify external business systems. The supplied customer data is synthetic; the reported metrics do not establish real-world predictive validity, causal effects, churn reduction, or return on investment.

> **Run mode:** The recommended local setup runs FastAPI and Streamlit directly on the host and uses SQLite. Docker is not required. The repository contains Docker deployment files, but Docker Compose and PostgreSQL deployment have not been verified in the recorded project testing.

## Contents

- What AURA does
- Current implementation
- Architecture
- Dataset
- Machine-learning methodology and results
- Getting started
- API and workflow notes
- Tests and validation
- Project structure
- Documentation
- Docker and PostgreSQL
- Limitations and future work
- Security and license

## What AURA does

AURA connects several tasks that are often handled separately in churn-analysis workflows:

1. **Detect** — identifies risk-oriented signals from available customer information.
2. **Diagnose** — organizes plausible churn contributors and supporting evidence. These are investigative signals, not proven causal explanations.
3. **Predict** — obtains a churn probability from the saved machine-learning model and applies the configured risk/threshold logic.
4. **Strategy** — evaluates and ranks candidate retention interventions using the implemented MCIA utility logic.
5. **Execute** — pauses for human approval, rechecks applicable authorization and approval conditions, and records an idempotent *simulated* action.

A persistent LangGraph workflow coordinates the stages and supports interruption and resumption at the approval checkpoint.

The intended decision-support sequence is **Risk → Cause → Action → Outcome → Learning**. The current prototype implements risk analysis, diagnosis, prediction, recommendation, approval and simulated execution. Outcomes from external business systems have not been collected, so the final stages are a future evaluation direction rather than a demonstrated closed-loop optimization result.

## Current implementation

The repository includes:

- Validation and import of the supplied demonstration data.
- Logistic Regression, Random Forest and XGBoost training and comparison.
- Training-only out-of-fold threshold selection under a configured outreach-capacity constraint.
- FastAPI backend and Streamlit frontend.
- SQLAlchemy persistence and Alembic migrations.
- Password-hashed signup/login/logout, role checks and tenant-scoped records.
- Five-agent LangGraph workflow with persistent checkpointing and a human approval interrupt.
- Recommendation approval/rejection and guarded, simulated execution.
- Optional centralized LLM Gateway with deterministic fallback behavior when an external provider is not configured.
- NetworkX customer/signal graph and deterministic demonstration timelines.
- Dataset upload, customer profiles, workflow history, reports and tenant settings.
- Tenant-authenticated WebSocket endpoint for status/audit metadata.
- CSV/PDF export features.

The main Streamlit interaction flow uses HTTP requests and refresh/polling. The presence of a WebSocket endpoint does **not** mean Kafka-based event streaming is implemented.

## Architecture

| Layer | Main components | Responsibility |
|---|---|---|
| Presentation | Streamlit | Dashboard, customer exploration, workflow review, reports and settings |
| API/application | FastAPI, Pydantic | API routes, request validation, authentication and application services |
| Orchestration | LangGraph and five agents | Customer analysis workflow, shared state, checkpointing and human approval |
| ML inference | scikit-learn pipeline, XGBoost | Churn-probability inference using the saved model artifact |
| Graph exploration | NetworkX | Representation and exploration of customer/signal relationships |
| Persistence | SQLAlchemy, Alembic, SQLite | Tenant, customer, dataset, prediction, workflow, recommendation, approval and audit records |
| Optional language-model service | LLM Gateway | Selected language-model operations when configured; deterministic fallback otherwise |

The database schema includes tenant, user and membership records; customers and events; dataset and model metadata; predictions and analysis records; agent runs; recommendations and approvals; simulated action executions; feedback; audit records; and LLM-usage records.

See `docs/DATABASE_SCHEMA.md` for schema details.

## Dataset

Four supplied CSV files are stored in `data/raw/`:

- `AURA_Synthetic_Customer_Retention_Dataset.csv` — full demonstration dataset.
- `AURA_Synthetic_Train.csv` — training split.
- `AURA_Synthetic_Test.csv` — held-out test split.
- `AURA_Data_Dictionary.csv` — feature definitions.

The reported primary dataset contains:

| Attribute | Reported value |
|---|---:|
| Total records | 12,000 |
| Columns | 36 |
| Dictionary-defined predictive inputs | 34 |
| Identifier | `customer_id` |
| Target | `churn` |
| Training records | 9,600 |
| Held-out test records | 2,400 |
| Churned records | 3,485 |
| Non-churned records | 8,515 |

`customer_id` is an identifier and is not used as a predictive feature. `churn` is the supervised-learning target and is excluded from live prediction inputs.

Earlier validation output records zero train/test identifier overlap, reconciliation of the full dataset with the supplied splits, valid schema types and numeric ranges, and category values contained within the full dataset.

Re-run validation against the checked-out files before relying on these values for a new experiment:

```bash
python scripts/validate_data.py
```

See `docs/DATASET_DOCUMENTATION.md`. The data is synthetic and must not be represented as genuine customer records.

## Machine-learning methodology and results

AURA trains and compares Logistic Regression, Random Forest and XGBoost.

Preprocessing is fitted inside scikit-learn pipelines. The documented training procedure uses three-fold stratified cross-validation on the 9,600-row training split for model/hyperparameter selection. The reported selection was XGBoost by cross-validation ROC-AUC of **0.6278**. Verify the exact pipeline configuration and reports before reproducing or citing this result.

### Fixed classification threshold: 0.50

The reported held-out test results for the selected XGBoost model at a fixed classification threshold of 0.50 are:

| Metric | Value |
|---|---:|
| Accuracy | 0.7108 |
| Precision | 0.5108 |
| Recall | 0.1019 |
| F1-score | 0.1699 |
| ROC-AUC | 0.6475 |
| PR-AUC | 0.4181 |

The low recall means this operating point misses a substantial portion of churn-labelled records. Accuracy alone should not be used to characterize model usefulness.

### Operational threshold: 0.38

Separately, an operational threshold of **0.38** was selected from training-only out-of-fold predictions by maximizing F1 under a configured **20% contact-capacity constraint**.

The previously reported held-out test results at this threshold are:

| Metric | Value |
|---|---:|
| Precision | 0.4922 |
| Recall | 0.3156 |
| F1-score | 0.3846 |
| Specificity | 0.8667 |
| Flagged rate | 18.63% |

These two evaluation settings must remain separate. Threshold 0.50 is the fixed-threshold model result; threshold 0.38 is the operational threshold selected using training-only out-of-fold predictions.

The 0.38 threshold is not a claim of real-world campaign performance.

Detailed reports and plots are stored in `ml/reports/`; see `docs/ML_METHODOLOGY.md`.

## Getting started

### Requirements

- Python compatible with the dependencies in `requirements.txt`.
- Git and a modern web browser.
- macOS, Linux or Windows.

The normal local setup uses SQLite. **Docker, PostgreSQL and Kafka are not required** to run the current local demonstration.

### macOS / Linux

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

On macOS, if XGBoost reports that the OpenMP runtime is missing, install it with Homebrew:

```bash
brew install libomp
```

### Windows PowerShell

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, allow it for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Configure the environment

Open `.env` and set a strong, unique `AURA_SECRET_KEY`.

Generate a value locally with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Paste the generated value into `.env`. Do not commit `.env` or share its contents. Keep `.env.example` as the non-secret configuration template.

For local SQLite operation, use:

```env
AURA_DATABASE_URL=sqlite:///./data/aura.db
```

Keep `AURA_PREDICTION_THRESHOLD` blank to use the saved model metadata threshold. Set it only when intentionally overriding inference.

Optional LLM settings may remain blank; deterministic fallback behavior is available where implemented.

### Apply migrations and validate the data

With the virtual environment activated:

```bash
alembic upgrade head
python scripts/validate_data.py
```

### Start the application

Open **Terminal 1** and start FastAPI:

```bash
uvicorn backend.main:app --reload --env-file .env
```

Open **Terminal 2**, activate the same virtual environment, and start Streamlit:

```bash
streamlit run frontend/app.py
```

Open the application:

- Streamlit dashboard: http://localhost:8501
- FastAPI OpenAPI documentation: http://127.0.0.1:8000/docs

### First application run

1. Create an organization account through the application. The signup flow creates the initial tenant owner.
2. Sign in and open the dashboard.
3. Select **Load supplied AURA demonstration dataset** to populate the tenant with the supplied 12,000-record dataset.
4. Open a customer profile and run the analysis workflow.
5. Review the Detect, Diagnose, Predict and Strategy outputs.
6. At the approval checkpoint, approve or reject recommendations as appropriate.
7. Review simulated execution and workflow records in the history area.

User-uploaded customer CSVs are validated and imported into the relevant tenant. They do **not** automatically retrain the saved churn model.

### Optional model retraining

The trained model and associated metadata are included under `ml/artifacts/`. Retraining is not required for a normal application run.

To intentionally rebuild the model artifacts:

```bash
python -m ml.training.train
```

Review the resulting reports and artifacts before committing any retrained model files.

## API and workflow notes

- FastAPI OpenAPI documentation is available at `/docs`.
- Protected HTTP requests use `Authorization: Bearer <token>`.
- Customer analysis runs Detect → Diagnose → Predict → Strategy and pauses at the persistent LangGraph human-approval interrupt.
- Approvers can approve or reject recommendations.
- Before a simulated execution record is created, the executor rechecks the relevant tenant, membership, approval and idempotency conditions.
- No external CRM, email, SMS, billing or customer-engagement system is contacted.
- The graph is an exploratory representation of customer signals and deterministic demonstration timelines; it is not a causal model.
- MCIA produces relative utility scores for recommendation assessment. These scores are not churn probabilities, causal effects or ROI estimates.
- The tenant-authenticated WebSocket endpoint is `/api/events?token=<bearer-token>` and sends tenant-scoped status/audit metadata. The main Streamlit interface uses HTTP refresh/polling.
- The PDF export is an activity summary; it does not claim retention uplift or ROI.

See `docs/USER_GUIDE.md` for the application navigation and demonstration workflow.

## Tests and validation

With the virtual environment activated, run:

```bash
python -m pytest -q
python -m compileall -q backend frontend ml scripts tests
python scripts/validate_data.py
```

Record the output from the exact commit being evaluated. Test success verifies the exercised software paths; it does not establish real-world model validity, absence of security vulnerabilities or production scalability.

## Project structure

```text
AURA/
├── backend/
│   ├── agents/             # Five customer-level agents
│   ├── database/           # SQLAlchemy models and database session
│   ├── graph_engine/       # NetworkX customer/signal graph
│   ├── llm/                # Optional LLM gateway and fallback
│   ├── mcia/               # Retention strategy utility logic
│   ├── orchestrator/       # LangGraph workflow
│   ├── security/           # Authentication and security helpers
│   └── services/           # Supporting services and event simulation
├── frontend/               # Streamlit application and API client
├── ml/
│   ├── artifacts/          # Saved model, feature schema and metadata
│   ├── evaluation/         # Threshold evaluation utilities
│   ├── inference/          # Model inference
│   ├── reports/            # Evaluation reports and plots
│   └── training/           # Model training pipeline
├── data/
│   ├── raw/                # Supplied synthetic CSVs and data dictionary
│   └── reports/            # Dataset validation report
├── alembic/                # Database migrations
├── docs/                   # Architecture, API, ML, security and user docs
├── scripts/                # Validation and utility scripts
├── tests/                  # Unit and frontend tests
├── .env.example            # Non-secret environment template
├── .gitignore
└── requirements.txt
```

## Documentation

- [Implementation progress](PROGRESS.md)
- [System architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Agent architecture](docs/AGENT_ARCHITECTURE.md)
- [Dataset documentation](docs/DATASET_DOCUMENTATION.md)
- [ML methodology](docs/ML_METHODOLOGY.md)
- [Database schema](docs/DATABASE_SCHEMA.md)
- [API documentation](docs/API_DOCUMENTATION.md)
- [Security documentation](docs/SECURITY_DOCUMENTATION.md)
- [Execution lifecycle](docs/EXECUTION_LIFECYCLE.md)
- [Event graph documentation](docs/DYNAMIC_GRAPH_DOCUMENTATION.md)
- [Causal analysis limitations](docs/CAUSAL_ANALYSIS_DOCUMENTATION.md)
- [MCIA documentation](docs/MCIA_DOCUMENTATION.md)
- [Testing report](docs/TESTING_REPORT.md)
- [Deployment guide](docs/DEPLOYMENT_GUIDE.md)
- [User guide](docs/USER_GUIDE.md)

## Docker and PostgreSQL

**Docker is optional and is not part of the recommended local setup.** The local run path above starts FastAPI and Streamlit directly on the host and uses SQLite.

The repository includes a `Dockerfile` and `docker-compose.yml` as deployment configuration. However, the Docker image build, Compose startup, PostgreSQL health checks, database-volume persistence and containerized end-to-end workflow have not been verified in the recorded project testing. Treat these files as experimental deployment scaffolding, not as a guaranteed working deployment path.

A PostgreSQL connection URL can be configured, for example:

```env
AURA_DATABASE_URL=postgresql+psycopg://user:password@host:5432/aura
```

A running PostgreSQL service, valid credentials and the required driver must be configured separately. Test PostgreSQL deployment before describing it as a verified capability.

## Limitations and future work

Current limitations include:

- The demonstration dataset is synthetic; generalization to real customer populations has not been established.
- Reported metrics describe the supplied data split and are not operational guarantees.
- Diagnosis and graph relationships are evidence-oriented/observational, not verified causal explanations.
- Demonstration event timelines are generated deterministically from aggregate information rather than ingested as live customer events.
- MCIA scores are relative decision-support utilities, not estimated treatment effects or financial returns.
- The Execute Agent records simulated actions only.
- External CRM and customer-engagement integrations are not implemented.
- Kafka streaming, federated learning and causal inference are not implemented.
- Docker/Compose and PostgreSQL deployment have not been verified in the recorded environment.

Potential future work includes evaluation on authorized real-world customer data, live event ingestion, Kafka integration, controlled intervention experiments, causal analysis with an appropriate study design, CRM integration, model-drift and fairness monitoring, and tested scalable deployment.

## Security and license

- Never commit `.env`, API keys, database files, private keys, access tokens or Streamlit secrets.
- Use `.env.example` to document variable names without values that grant access.
- The included model artifact is a Python pickle file. Only load trusted model artifacts; loading an untrusted pickle can execute arbitrary code.
- Confirm that the synthetic dataset and its source permit redistribution before making the repository public.
- Do not add real customer data, production exports, personal information or internal reports to this repository.
- Review dependencies, secret handling, TLS, backups, authorization and logging before any production deployment.
- No open-source license is declared here. Public repository visibility does not automatically grant permission to reuse, modify or redistribute the project. Confirm ownership and obtain agreement from all contributors before selecting a license or changing repository visibility.