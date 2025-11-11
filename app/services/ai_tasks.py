import uuid
from sqlalchemy.orm import Session
from sqlalchemy import select
from langchain_openai import OpenAIEmbeddings
from app.models import Recipe
from scripts.constants import EMBEDDING_MODEL

def process_single_recipe_embedding_sync(recipe_id: uuid.UUID, db: Session):
    """
    Generates and saves the vector embedding for a single recipe synchronously.
    (Used temporarily for MVP E2E testing).
    """

    embeddings_client = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # fetch recipe data
    recipe = db.scalars(
        select(Recipe).filter(Recipe.id == recipe_id)
    ).first()

    if not recipe or not recipe.content_text:
        print(f"Vectorization skipped: Recipe {recipe_id} content missing.")
        return

    # generate the vector
    vector_list = embeddings_client.embed_documents([recipe.content_text])
    vector = vector_list[0]

    # update db record with new vector
    recipe.embedding = vector
    db.add(recipe)
    db.commit()
    print(f"✅ SYNCHRONOUS VECTORIZATION COMPLETE for {recipe.name}")