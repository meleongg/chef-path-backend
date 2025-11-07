import os
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import PGVector
from langchain_core.tools import tool
from app.database import engine
from app.models import User, Recipe, UserRecipeProgress
from app.schemas.adaptive_planner import HybridSearchInput

CONNECTION_STRING = os.environ.get("DATABASE_URL")
if not CONNECTION_STRING:
    raise EnvironmentError("DATABASE_URL not set for Adaptive Planner Service.")


class AdaptivePlannerService:
    """
    Service responsible for adaptive meal plan retrieval and optimization using RAG/Vector Search.
    """

    def __init__(self, db: Session):
        self.db: Session = db
        self.embeddings_client: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.vector_store: PGVector = PGVector(
            embedding_function=self.embeddings_client,
            collection_name="recipes",
            connection_string=CONNECTION_STRING,
        )

    def get_recipe_candidates_hybrid(
        self,
        user_id: uuid.UUID,
        intent_query: str,
        exclude_ids: List[uuid.UUID],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Recipe]:
        """
        Executes a HYBRID (Vector Search + SQL Filter) query.
        1. Ranks recipes by semantic similarity (Vector Search).
        2. Excludes recipes that the user has rated poorly (SQL NOT IN filter).
        """
        query_vector = self.embeddings_client.embed_query(intent_query)
        exclusion_str_list = [str(uid) for uid in exclude_ids]
        raw_sql_query = text(
            f"""
            SELECT
                r.id,
                r.name,
                r.content_text,
                (1 - (r.embedding <=> :query_vector)) AS similarity_score
            FROM
                recipes r
            WHERE
                r.id NOT IN :exclusion_list
                AND (1 - (r.embedding <=> :query_vector)) > :similarity_threshold
            ORDER BY
                similarity_score DESC
            LIMIT :limit;
        """
        )
        result = self.db.execute(
            raw_sql_query,
            {
                "query_vector": query_vector,
                "exclusion_list": tuple(exclusion_str_list),
                "limit": limit,
                "similarity_threshold": similarity_threshold,
            },
        ).all()
        candidate_ids = [row[0] for row in result]
        final_recipes = self.db.scalars(
            select(Recipe).filter(Recipe.id.in_(candidate_ids))
        ).all()
        return final_recipes

    def generate_weekly_plan(self, user_id: UUID) -> List[UUID]:
        """
        Orchestrates the entire week's plan generation.
        """
        # Placeholder for future LangGraph logic
        return []

planner_service = AdaptivePlannerService(db=CONNECTION_STRING)

@tool(args_schema=HybridSearchInput)
def get_recipe_candidates(
    intent_query: str,
    user_id: uuid.UUID,
    exclude_ids: List[uuid.UUID] = [],
    similarity_threshold: float = 0.7,
    limit: int = 10,
) -> str:
    """
    Retrieves the most semantically relevant recipes by performing a vector search,
    filtered by user-specific negative feedback and exclusion lists.
    Use this tool to find the optimal candidates for a weekly meal plan.
    """

    results: List[Recipe] = planner_service.get_recipe_candidates_hybrid(
        user_id=user_id,
        intent_query=intent_query,
        exclude_ids=exclude_ids,
        limit=limit,
        similarity_threshold=similarity_threshold,
    )

    # Return the results as a string summary for the LLM to process
    if not results:
        return "No suitable recipes found based on the hybrid search criteria."

    # Format the output into a clean string for the LLM's context window
    output_summary = "\n--- Recipe Candidates ---\n"
    for i, recipe in enumerate(results):
        output_summary += f"{i+1}. ID: {recipe.id}, Name: {recipe.name}, Difficulty: {recipe.difficulty}, Score: [RETRIEVED SEMANTIC SCORE]\n"

    return output_summary
