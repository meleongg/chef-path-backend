import os
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()


def initialize_postgres_saver():
    """Creates and sets up the PostgresSaver instance for synchronous checkpointing."""

    DB_URI = os.environ.get("DATABASE_URL")
    # --- SANITIZE URI (Remove SQLAlchemy dialect prefix) ---
    # PostgresSaver expects postgresql:// or postgres://, NOT postgresql+psycopg2://
    if DB_URI and DB_URI.startswith("postgresql+psycopg://"):
        DB_URI = DB_URI.replace("postgresql+psycopg://", "postgresql://")

    # Create persistent connection with required parameters
    # autocommit=True: Required for .setup() to properly commit checkpoint tables
    # row_factory=dict_row: Required for dictionary-style row access (e.g., row["column_name"])
    conn = psycopg.connect(DB_URI, autocommit=True, row_factory=dict_row)

    # Create PostgresSaver with the connection
    checkpointer = PostgresSaver(conn)

    # Setup tables (creates checkpoints, checkpoint_writes, etc. if they don't exist)
    checkpointer.setup()
    print("✅ PostgresSaver initialized and tables created/verified.")

    return checkpointer
