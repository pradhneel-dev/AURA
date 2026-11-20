# AURA user guide

## Start and sign in

Use Python 3.11 or newer (macOS may need `brew install libomp` for XGBoost). From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set a long, stable `AURA_SECRET_KEY` in `.env`. Then apply migrations and run these commands in separate terminals:

```bash
alembic upgrade head
uvicorn backend.main:app --reload --env-file .env
```

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

Open `http://localhost:8501`. Create an organization account or sign in. If your account belongs to multiple tenants, choose the organization shown after the first sign-in attempt. An Owner/Admin can load **Load supplied 12,000-row demo dataset** on Overview or import a validated customer CSV from **Datasets**.

`AURA_PREDICTION_THRESHOLD` is an optional inference override. Leave it unset or blank to use the threshold stored in the trained model metadata. A nonblank value must be a finite number between 0 and 1. API errors include a request ID that can be matched to backend logs.

## Explore the workspace

- **Overview:** tenant customer totals, stored prediction categories, pending/approved recommendations, segment counts, historical labels, recent analyses, and agent activity. Prediction counts only include records that have been scored; target labels are shown separately.
- **Customers:** filter by ID, risk, and segment; sort and paginate through server-side results. Select a result and open its profile.
- **Customer profile:** inspect source attributes and previous prediction, request a prediction if one is absent, review simulated event history, and start the five-agent workflow. Analyze requests persist a new prediction and analysis run.
- **Workflow:** review Detect evidence, Diagnose hypotheses and their provenance, Predict output, eligible Strategy proposals, and actual persisted agent statuses. A paused run is waiting for approval.
- **Retention center:** open a recommendation to view evidence and M/T/R/F values. Only Owner, Admin, or Approver roles can review. Confirm the review before approving or rejecting. Approval records and resumes a simulation; rejection never executes.
- **History & activity:** inspect runs, agent executions, recommendation decisions, simulated executions, recorded follow-up, and audit activity. Follow-up is a manual observation, not a causal effect estimate.
- **Model report:** compare candidate models and inspect the training-OOF-selected threshold, held-out metrics, confusion matrix, ranking charts, and feature importance.
- **Causal explorer:** browse a descriptive customer-event graph. Event nodes are simulated; graph edges do not establish causation.
- **Datasets:** Owners/Admins validate, preview, and import CSV customer records. Customer imports never retrain the model.
- **Reports:** download model comparison CSV, tenant recommendation CSV, and tenant activity PDF.
- **Settings:** view account and organization details, manage MCIA weights, tenant LLM authorization, and members. Provider keys stay on the backend.

## Recommendation lifecycle

An analysis creates eligible proposals that wait for human review. Approving a proposal triggers the persisted LangGraph approval resume and a safe simulation. No external email, CRM, billing, or support integration is called. The execution result says customer retention is unknown. Use History to record observed follow-up later.

## Interpret model results carefully

The supplied data is synthetic. The model's 0.38 threshold was selected using training-only out-of-fold probabilities under a 20% contact-capacity assumption. Test recall is about 31.6%, so the model misses many positive labels in this evaluation. Prediction probabilities do not guarantee churn or retention.

## Release verification (2026-11-05)

The fresh-browser check used a new owner tenant, an isolated SQLite database, and an explicitly blank threshold environment value. The UI loaded all 12,000 supplied demo rows. From Customers, a profile was opened and a five-agent run paused for approval. An authorized owner approved one recommendation; the LangGraph run resumed and History showed one simulated execution. Two other recommendations were rejected; neither was executed. The run completed after all three were reviewed.

The visual pass covered sign-in, all ten workspace pages, and Settings sections at the available default 840×792 browser viewport. Automated verification: `python3 -m pytest -q` — 29 passed with 14 third-party deprecation warnings; Python compilation passed. `git diff --check` exited successfully, though all project files are untracked in this workspace, so there is no tracked diff for it to inspect. Isolated SQLite migrations reached `0002_llm_usage (head)`.

Docker-compatible runtimes were not present, so PostgreSQL/Compose build, health checks, and persistent-volume behavior were not verified. See [deployment notes](DEPLOYMENT_GUIDE.md) before running Compose.
