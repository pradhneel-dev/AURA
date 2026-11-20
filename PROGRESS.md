# AURA implementation status

Updated 2026-11-05.

## Implemented

- Preserved and validated the four supplied data files (12,000 complete records; 9,600 train and 2,400 held-out test; 34 input features). Identifier and target are excluded from model inputs.
- Trained Logistic Regression, Random Forest, and XGBoost with training-only cross-validation. Original fixed-0.50 comparison reports are retained as `model_comparison_initial.*`.
- Selected operational threshold 0.38 using training-only out-of-fold predictions under a configured 20% outreach capacity. Evaluated it once on the held-out test split. Report includes threshold metrics, confusion matrix, precision-recall and threshold plots, top-K ranking metrics, and Brier score diagnostic. Server can override through `AURA_PREDICTION_THRESHOLD`.
- Persisted SQLAlchemy schema and Alembic migrations for tenants, users/memberships, customers/events, datasets, model metadata, predictions, analysis, agent runs, recommendations, approvals, simulated executions, outcome feedback, audit, and LLM usage. SQLite is local default; PostgreSQL is supported.
- Signup, login/logout, password hashing, bearer sessions, role checks, tenant-scoped access, owner/admin user management, tenant MCIA and LLM settings, dataset validation/import, and 12,000-row demo import.
- Five coordinated agents (Detect, Diagnose, Predict, Strategy, Execute) using a persistent SQLite-checkpointed LangGraph. Approval is a real interrupt/resume checkpoint. Recommendation records are tenant-scoped; execution rechecks reviewer role and approval, is idempotent, and only records an explicitly simulated action. Outcomes are manual follow-up and do not imply causal retention.
- Central optional LLM Gateway; customer IDs are omitted from LLM context, tenant authorization and daily usage limits are checked, structured outputs are validated, and deterministic fallbacks keep core operation available without credentials.
- Streamlit authentication, dashboard, paginated customers, analysis results, model/threshold report, observational graph explorer, recommendations and approvals, history, CSV exports, PDF download, MCIA/LLM/user settings, and dataset import/validation.
- Added an in-app user guide and authenticated AppTest journey covering dashboard/customer navigation, profile, analysis, approval, simulated execution, and the remaining major page renders. Fixed mixed-type profile attribute rendering, a review-form button deadlock, and overly wide activity tables/settings readouts.
- Tenant-authenticated WebSocket event feed at `/api/events?token=...`; it emits tenant-local audit/agent status metadata. Streamlit pages retain HTTP polling/refresh as fallback.
- Docker image and Compose stack for PostgreSQL, API, and Streamlit. Migrations run at API container startup; DB and runtime checkpoint data use persistent volumes.
- OpenAPI at `/docs`; architecture, security, database, API, ML, causal limitations, execution, testing, and deployment docs are maintained in `docs/`.

## Measured model results

CV model selection chose XGBoost (training CV ROC-AUC 0.6278). At fixed 0.50 threshold, held-out accuracy 0.7108, precision 0.5108, recall 0.1019, F1 0.1699, ROC-AUC 0.6475, PR-AUC 0.4181. At operational threshold 0.38, held-out precision 0.4922, recall 0.3156, F1 0.3846, specificity 0.8667, FPR 0.1333, FNR 0.6844, flagged rate 18.63%, Brier score 0.1949. This is synthetic data and does not establish real-world validity.

## Known limitations

- The dataset has no defensible randomized intervention, treatment, and confounder design. Causal Explorer is an observational/hypothesis view only; no DoWhy/EconML effect estimate is produced.
- Event timelines are deterministic simulations based on aggregate features, not observed product events.
- The LLM provider is optional and configured server-side; no provider key is included. Fallbacks are deterministic.
- WebSocket pushes minimal status/audit metadata; Streamlit currently uses its HTTP polling/refresh flow rather than maintaining a persistent socket.
- PDF export is an activity summary. CSV exports include model and tenant recommendation records; there is no claim of ROI or retention.
- No external CRM, messaging, billing, or support integrations exist. Execution is simulation only.
- Set a strong stable `AURA_SECRET_KEY` for multi-worker deployments and session persistence. Local generated secret is process-scoped.
- Docker Compose configuration is provided but requires a container runtime and secrets supplied by the operator; it has not been exercised in this workspace.

## Verification

- Root cause confirmed: the local `.env` sets `AURA_PREDICTION_THRESHOLD=` to an empty value. `float('')` failed during predictor initialization, which broke prediction and endpoints that load model threshold details. Blank or whitespace values now use the trained metadata threshold; invalid nonblank values return a clear configuration error. Numeric feature validation names the invalid field.
- API/workflow diagnostics now attach an `X-Request-ID`, include it in safe API errors, and log unhandled request/workflow exceptions with method/path or workflow IDs and tenant/user context. Workflow error rows contain exception type and a log reference, not customer feature values.
- Fresh isolated SQLite database `/private/tmp/aura_audit_fresh.sqlite`: `alembic upgrade head` passed through `0002_llm_usage (head)`. The application ran with an explicitly blank threshold environment value.
- Fresh-browser workflow passed on a new isolated owner tenant: sign-up, dashboard, import all 12,000 demo records, open `CUST_00001`, run Detect/Diagnose/Predict/Strategy, and reach the approval interrupt. An owner approved one recommendation; the resumed workflow recorded one simulated Execute action visible in History. Two other recommendations were rejected and show rejected status; no execution is associated with them. After both decisions the run reached `completed`.
- Browser review covered sign-in and all ten workspace pages, including model report and Settings sections. At the available default 840×792 browser viewport, the reviewed layouts, charts, tables, status labels, empty states, and navigation rendered without a blocking visual defect. This was not a separate 1280×720 viewport run.
- `python3 -m pytest -q` — **29 passed**, 14 third-party deprecation warnings.
- `python3 -m compileall -q backend frontend ml scripts tests` and `git diff --check` — passed.
- Git reports the project files as untracked in this workspace. `git diff --check` exited successfully, but Git has no tracked source diff to inspect.
- Docker, Podman, and Nerdctl are unavailable; PostgreSQL/Compose build, container health checks, and persistent-volume behavior remain unverified.
