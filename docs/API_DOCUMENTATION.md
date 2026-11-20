# API reference

Run `uvicorn backend.main:app --reload`; OpenAPI UI is at `/docs`. Protected HTTP routes require `Authorization: Bearer <access_token>`. Tenant identity is derived from the signed token and current membership, never from a caller-supplied tenant ID.

## Authentication and tenant settings

- `POST /api/auth/signup`, `/api/auth/login`, `/api/auth/logout`; `GET /api/auth/me`
- `GET/POST /api/tenant/users` (Owner/Admin)
- `GET /api/settings`; `PUT /api/settings/mcia`, `GET/PUT /api/settings/llm` (Owner/Admin)

## Customers, data, and model

- `GET /api/dashboard/overview`
- `GET /api/customers?limit=&offset=&search=&risk_level=&segment=&sort_by=&descending=`; `GET /api/customers/{customer_id}`. Supported sort fields are `customer_id`, `segment`, `risk_level`, and `churn_probability`.
- `GET /api/customers/{customer_id}/events`; `GET /api/customers/{customer_id}/graph`
- `POST /api/datasets/validate`, `/api/datasets/upload`, `/api/datasets/demo`; `GET /api/datasets`
- `POST /api/predictions` validates the saved model's 34-feature schema and persists prediction.
- `GET /api/reports/models`; `GET /api/reports/threshold`

Uploads are tenant-scoped CSV customer records. They do not retrain the active model. The supplied synthetic dataset can be imported through `/api/datasets/demo`.

## Agents and recommendations

- `POST /api/customers/{customer_id}/analyze` starts an individual workflow and returns 202 with recommendation IDs and approval interrupt details.
- `POST /api/analysis/batch` analyzes up to 100 IDs independently.
- `GET /api/analysis`, `GET /api/analysis/{run_id}`; `GET /api/agents/status`; `GET /api/agents/runs/{run_id}`
- `GET /api/recommendations?status=&customer_id=&limit=&offset=`; `GET /api/recommendations/{id}`
- `POST /api/recommendations/{id}/approve` or `/reject` with optional `{ "note": "..." }`.
- `POST /api/recommendations/{id}/execute` returns existing idempotent result or 409 if no execution was recorded via approval.
- `GET /api/interventions`; `POST /api/interventions/{id}/outcome`

Approval resumes the persisted LangGraph interrupt; execution is simulated only. See [execution lifecycle](EXECUTION_LIFECYCLE.md).

## Reports and events

- `GET /api/reports/overview`; `GET /api/reports/tenant.pdf`
- `GET /api/audit?limit=&offset=&action=` returns tenant-scoped audit events.
- `WS /api/events?token=<access_token>` authenticates at connection time and streams tenant-local audit/agent status metadata. Streamlit uses HTTP refresh polling as fallback.

All event histories are simulated, graph relations are observational, and manually recorded outcomes are not causal estimates. OpenAPI is authoritative for exact request/response schemas and status codes.
