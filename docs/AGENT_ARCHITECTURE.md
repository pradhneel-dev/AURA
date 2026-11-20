# Agent architecture

The FastAPI analysis entry point starts a typed LangGraph state machine in `backend/orchestrator/workflow.py`:

`Detect → Diagnose → Predict → Strategy → Human Approval interrupt → Execute`

- **Detect** inspects the stored customer features and generated simulated events. It returns structured signals with evidence references.
- **Diagnose** returns evidence-linked possible explanations. It can use the centralized LLM gateway only when provider configuration and tenant authorization are both present; otherwise it uses deterministic templates. LLM claims that are not supported by supplied evidence are rejected.
- **Predict** validates the 34-feature schema and loads the saved XGBoost inference pipeline. It returns probability, threshold decision, risk category, and model version.
- **Strategy** applies eligibility/policy filters and the normalized M/T/R/F MCIA ranking. It persists tenant-scoped proposals and pauses before any action.
- **Execute** resumes only after an approval record and currently authorized reviewer membership are verified. It records an idempotent simulation only; no external service is contacted.

LangGraph `interrupt()` is checkpointed with a persistent SQLite checkpointer and resumed with a recommendation ID and reviewer identity. Checkpoint thread ID is the analysis run ID. Database agent execution rows provide history; recommendation, approval, execution, audit, and optional LLM usage are persisted separately. Batch analysis isolates each customer's errors.

No API key is required. LLM credentials are server-side, tenant authorization is explicit, and a daily call limit is enforced. Customer IDs are not sent to the LLM. Event text/features are untrusted evidence and cannot grant permissions or bypass human approval.

Events are synthetic and aggregate feature relationships are observational. Explanations and MCIA ranks are not causal effects, guaranteed retention, or ROI. Read [execution lifecycle](EXECUTION_LIFECYCLE.md) and [security](SECURITY_DOCUMENTATION.md).
