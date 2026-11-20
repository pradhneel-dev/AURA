# Database schema

SQLAlchemy models are in `backend/database/models.py`; the initial Alembic revision is `alembic/versions/0001_initial_schema.py`. Apply it with `alembic upgrade head`.

| Table | Purpose and tenant key |
|---|---|
| `tenants` | Organization identity and settings |
| `users` | Global login identity and token version |
| `memberships` | Tenant membership and comma-separated roles; unique per tenant/user |
| `dataset_versions` | Tenant dataset metadata and validation report |
| `customers` | Tenant customer features (the target label is not copied into customer records); unique tenant/customer ID |
| `customer_events` | Tenant-scoped source or simulated customer events |
| `model_metadata` | Versioned model metadata and threshold |
| `predictions` | Tenant/customer probability, threshold, classification and model version |
| `analysis_runs` | Tenant analysis result/status/error and actor |
| `agent_executions` | Per-run structured agent inputs/outputs and status |
| `recommendations` | Tenant/customer action, MCIA breakdown, status and idempotency key |
| `approvals` | Unique review record, reviewer, decision and reason |
| `action_executions` | Unique simulated execution per recommendation |
| `outcome_feedback` | Later human-recorded intervention outcome; not assumed from execution |
| `audit_logs` | Tenant/user action history |

SQLite is the local default. `AURA_DATABASE_URL` can select PostgreSQL using `postgresql+psycopg://...`. Foreign-key enforcement is enabled for SQLite. JSON columns use SQLAlchemy's portable JSON type. Sensitive credentials are not stored in tenant settings.

Note: the initial migration records the project schema baseline. Use explicit Alembic operations for subsequent schema changes.
