import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv

load_dotenv()


def initialize_postres_saver():
    """Creates the AsyncPostgresSaver instance (but does NOT call setup())."""

    DB_URI = os.environ.get("DATABASE_URL")
    # --- SANITIZE URI (Remove SQLAlchemy dialect prefix) ---
    # PostgresSaver expects postgresql:// or postgres://, NOT postgresql+psycopg2://
    if DB_URI and DB_URI.startswith("postgresql+psycopg://"):
        DB_URI = DB_URI.replace("postgresql+psycopg://", "postgresql://")

    return AsyncPostgresSaver.from_conn_string(DB_URI)
