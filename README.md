# Mise backend

FastAPI service for Mise, an AI-assisted meal-planning application. It owns
authentication, user preferences, recipe data, weekly-plan state, progress, and
the planner API consumed by the Next.js client.

## Architecture

- **API and persistence:** FastAPI, Pydantic, SQLAlchemy, Alembic, and PostgreSQL
- **Planning:** LangGraph tools retrieve recipes through pgvector similarity search
  plus SQL preference filters; the agent can generate and embed a recipe when
  retrieval falls short
- **Plan integrity:** ordered schedules, progress records, swap limits, and
  cooldown exclusions keep plan and recommendation state consistent
- **Reliability:** JWT auth, explicit CORS origins, per-IP and per-user AI rate
  limits, moderation, Railway health checks, and pytest coverage

## Run locally

```bash
cd mise-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Set `DATABASE_URL` and `SECRET_KEY` before starting. Adaptive planning requires
PostgreSQL with the `pgvector` extension and `OPENAI_API_KEY`; SQLite is suitable
only for lightweight development and tests. Set `CORS_ORIGINS` to the exact
frontend origin outside local development.

## Test

```bash
DATABASE_URL=sqlite:///:memory: OPENAI_API_KEY=test-key pytest -q
```

## Deployment

Railway runs `uvicorn main:app` and verifies `/health`, which also checks database
connectivity. Configure production secrets and database URLs in the deployment
environment; never commit them.
