import os
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

CHECKPOINT_SAVER = None
try:
    # This tells LangGraph to use the PostgreSQL connection pool for state persistence.
    CHECKPOINT_SAVER = PostgresSaver(DATABASE_URL)
    print("✅ SQLAlchemySaver initialized successfully.")

except Exception as e:
    print(f"❌ ERROR: Failed to initialize SQLAlchemySaver. Check DB connection: {e}")
    # In a real app, you'd fail gracefully or use a non-persistent saver for dev.
