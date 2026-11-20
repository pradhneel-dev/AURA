# Verification record

Latest run in the development workspace:

- `AURA_DATABASE_URL=sqlite:///./data/aura.db alembic upgrade head` — passed; `alembic current` reports `0002_llm_usage (head)`.
- `python3 -m pytest -q` — **29 passed**, 14 third-party deprecation warnings.
- `python3 -m compileall -q backend frontend ml scripts tests` — passed.
- Test coverage includes saved model inference and schema rejection, threshold selection constraints, prediction API, tenant scope, auth and role checks, dataset upload and 12,000-row demo import, MCIA, graph/event simulation, real LangGraph interruption/resume, simulated execution/idempotency, feedback, PDF auth, and invalid WebSocket token rejection.

The suite uses an isolated in-memory SQLite database for HTTP tests and covers blank/invalid threshold settings, feature validation, request IDs, tenant scope, and workflow failure recording in addition to the established API and LangGraph flows. Streamlit AppTest exercises authentication, dashboard/customer views, profile navigation, analysis, approval, simulated execution, page rendering, and the review confirmation guard. A fresh-browser journey used a new tenant/database and an explicitly blank threshold. One recommendation was approved and executed in simulation; two were rejected, and History showed no execution for them. The run reached `completed`. Browser review covered sign-in, all ten workspace pages, and Settings sections at the available 840×792 viewport; a distinct 1280×720 run was not performed. Fresh isolated SQLite Alembic upgrade passed at `0002_llm_usage (head)`. PostgreSQL/Compose were not run because Docker, Podman, and Nerdctl were unavailable. The 14 warnings originate from installed LangSmith/AnyIO dependencies. `git diff --check` exited successfully, but the workspace reports all project files as untracked, so there is no tracked diff for Git to inspect.
