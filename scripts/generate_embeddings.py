import os
import sys
import time
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import select, update
import uuid

# --- Ensure pathing works for local imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from app.database import engine
from app.models import Recipe
from langchain_openai import OpenAIEmbeddings

# Load environment variables (OPENAI_API_KEY and DATABASE_URL)
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50


def generate_embeddings_for_recipes():
    """Generates vectors for recipes missing an embedding and saves them to the DB."""
    try:
        embeddings_client = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(
            f"❌ Error initializing OpenAI client. Check OPENAI_API_KEY environment variable: {e}"
        )
        return

    db_session = Session(bind=engine)
    try:
        # --- Query to find unembedded recipes with content_text ---
        recipes_to_process: list[Recipe] = db_session.scalars(
            select(Recipe).filter(
                Recipe.content_text.is_not(None), Recipe.embedding.is_(None)
            )
        ).all()
        if not recipes_to_process:
            print(
                "✅ All recipes already have embeddings. Database is fully vectorized."
            )
            return
        print(
            f"🧠 Found {len(recipes_to_process)} recipes to process. Starting batch vectorization..."
        )
        total_processed = 0
        for i in range(0, len(recipes_to_process), BATCH_SIZE):
            batch: list[Recipe] = recipes_to_process[i : i + BATCH_SIZE]
            texts: list[str] = [r.content_text for r in batch if r.content_text]
            recipe_ids: list[uuid.UUID] = [r.id for r in batch if r.content_text]
            if not texts:
                continue
            print(
                f"  -> Processing batch {i // BATCH_SIZE + 1} ({len(texts)} items)..."
            )
            batch_embeddings: list[list[float]] = embeddings_client.embed_documents(
                texts
            )
            for recipe_id, vector in zip(recipe_ids, batch_embeddings):
                db_session.execute(
                    update(Recipe)
                    .where(Recipe.id == recipe_id)
                    .values(embedding=vector)
                )
                total_processed += 1
            db_session.commit()
            print(f"  -> Batch {i // BATCH_SIZE + 1} saved successfully.")
            time.sleep(0.5)
        print(f"\n🎉 Vectorization complete! Total recipes processed: {total_processed}")
    except Exception as e:
        db_session.rollback()
        print(f"❌ Critical error during vectorization. Rolling back batch: {e}")
    finally:
        db_session.close()


if __name__ == "__main__":
    generate_embeddings_for_recipes()
