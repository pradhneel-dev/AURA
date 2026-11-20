# Deployment

## Local SQLite development

Use Python 3.11+, install `requirements.txt` (macOS may need `brew install libomp` for XGBoost), copy `.env.example` to `.env`, set a stable high-entropy `AURA_SECRET_KEY`, then run:

```bash
alembic upgrade head
uvicorn backend.main:app --reload --env-file .env
streamlit run frontend/app.py
```

The default database is `sqlite:///./data/aura.db`. The LangGraph checkpoint is stored at `data/aura_graph_checkpoints.sqlite`. Back up both if retaining local workflow state.

## Docker Compose with PostgreSQL

Docker Compose runs PostgreSQL, the FastAPI API, and Streamlit. Copy `.env.example` to `.env` and set nonempty unique values for `POSTGRES_PASSWORD` and `AURA_SECRET_KEY`; then:

```bash
docker compose up --build
```

The API service applies Alembic migrations before starting. PostgreSQL and runtime/checkpoint data use named persistent volumes. UI: `http://localhost:8501`; API docs: `http://localhost:8000/docs`.

The Docker configuration is present but was not run in local verification. Before production, review TLS termination, network exposure, secret rotation, backups, rate limiting, multi-worker checkpoint behavior, and operational monitoring. Keep provider credentials in server-side deployment secrets. The application does not need an LLM key for its deterministic fallback.
