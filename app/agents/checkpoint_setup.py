from langgraph.checkpoint.postgres import PostgresSaver
from app.database import engine

try:
    # This tells LangGraph to use the PostgreSQL connection pool for state persistence.
    CHECKPOINT_SAVER = PostgresSaver.from_engine(engine)
    print("✅ SQLAlchemySaver initialized successfully.")

except Exception as e:
    print(f"❌ ERROR: Failed to initialize SQLAlchemySaver. Check DB connection: {e}")
    # In a real app, you'd fail gracefully or use a non-persistent saver for dev.