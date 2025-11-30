import os
import uuid
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import select, text
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.database import engine
from app.models import Recipe
from app.schemas.adaptive_planner import (
    HybridSearchInput,
    FinalPlanOutput,
    NewRecipeSchema,
    RecipeSelectionInput,
)
from app.constants import EMBEDDING_MODEL, GENERATIVE_MODEL
from app.services.ai_tasks import process_single_recipe_embedding_sync
from app.database import SessionLocal

CONNECTION_STRING = os.environ.get("DATABASE_URL")
if not CONNECTION_STRING:
    raise EnvironmentError("DATABASE_URL not set for Adaptive Planner Service.")

GENERATIVE_LLM = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0.7)


class AdaptivePlannerService:
    """
    Service responsible for adaptive meal plan retrieval and optimization using RAG/Vector Search.
    """

    def __init__(self, db: Session):
        self.db: Session = db
        self.embeddings_client: OpenAIEmbeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )
        self.vector_store: PGVector = PGVector(
            embeddings=self.embeddings_client,
            collection_name="recipes",
            connection=CONNECTION_STRING,
        )

    def get_recipe_candidates_hybrid(
        self,
        user_id: uuid.UUID,
        intent_query: str,
        exclude_ids: List[uuid.UUID],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[tuple]:
        """
        Executes a HYBRID (Vector Search + SQL Filter) query.
        1. Ranks recipes by semantic similarity (Vector Search).
        2. Excludes recipes that the user has rated poorly (SQL NOT IN filter).
        Returns a list of (Recipe, similarity_score) tuples.
        """
        query_vector = self.embeddings_client.embed_query(intent_query)
        exclusion_str_list = [str(uid) for uid in exclude_ids]

        # Convert list to string format for pgvector - embed directly in SQL
        vector_str = f"'[{','.join(map(str, query_vector))}]'::vector"

        # Build SQL query dynamically based on whether there are exclusions
        if exclusion_str_list:
            raw_sql_query = text(
                f"""
                SELECT
                    r.id,
                    r.name,
                    r.content_text,
                    (1 - (r.embedding <=> {vector_str})) AS similarity_score
                FROM
                    recipes r
                WHERE
                    r.id NOT IN :exclusion_list
                    AND (1 - (r.embedding <=> {vector_str})) > :similarity_threshold
                ORDER BY
                    similarity_score DESC
                LIMIT :limit;
            """
            )
            result = self.db.execute(
                raw_sql_query,
                {
                    "exclusion_list": tuple(exclusion_str_list),
                    "limit": limit,
                    "similarity_threshold": similarity_threshold,
                },
            ).all()
        else:
            # No exclusions - simpler query
            raw_sql_query = text(
                f"""
                SELECT
                    r.id,
                    r.name,
                    r.content_text,
                    (1 - (r.embedding <=> {vector_str})) AS similarity_score
                FROM
                    recipes r
                WHERE
                    (1 - (r.embedding <=> {vector_str})) > :similarity_threshold
                ORDER BY
                    similarity_score DESC
                LIMIT :limit;
            """
            )
            result = self.db.execute(
                raw_sql_query,
                {
                    "limit": limit,
                    "similarity_threshold": similarity_threshold,
                },
            ).all()
        candidate_ids = [row[0] for row in result]
        recipes_by_id = {
            r.id: r
            for r in self.db.scalars(
                select(Recipe).filter(Recipe.id.in_(candidate_ids))
            ).all()
        }
        # Return list of (Recipe, similarity_score) tuples, preserving order
        return [
            (recipes_by_id[row[0]], row[3]) for row in result if row[0] in recipes_by_id
        ]


@tool(args_schema=RecipeSelectionInput)
def finalize_recipe_selection(recipe_ids: List[str]) -> str:
    """
    Call this tool to finalize your recipe selection for the weekly plan.
    Provide a list of recipe IDs (as strings) that you've chosen.
    """
    print(f"\n[TOOL: finalize_recipe_selection] === CALLED ===")
    print(f"[TOOL] recipe_ids: {recipe_ids}")
    print(f"[TOOL] Number of recipes: {len(recipe_ids)}")
    return f"Successfully selected {len(recipe_ids)} recipes for the weekly plan."


@tool(args_schema=HybridSearchInput)
def get_recipe_candidates(
    intent_query: str,
    user_id: uuid.UUID,
    exclude_ids: List[uuid.UUID] = None,
    similarity_threshold: float = 0.3,
    limit: int = 10,
) -> str:
    """
    Retrieves the most semantically relevant recipes by performing a vector search,
    filtered by user-specific negative feedback and exclusion lists.
    Use this tool to find the optimal candidates for a weekly meal plan.
    """
    print(f"\n[TOOL: get_recipe_candidates] === CALLED ===")
    print(f"[TOOL] intent_query: {intent_query}")
    print(f"[TOOL] user_id: {user_id}")
    print(f"[TOOL] exclude_ids: {exclude_ids}")
    print(f"[TOOL] similarity_threshold: {similarity_threshold}")
    print(f"[TOOL] limit: {limit}")

    if exclude_ids is None:
        exclude_ids = []

    db = SessionLocal()
    try:
        service = AdaptivePlannerService(db=db)
        results: List[tuple] = service.get_recipe_candidates_hybrid(
            user_id=user_id,
            intent_query=intent_query,
            exclude_ids=exclude_ids,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

        # Return the results as a string summary for the LLM to process
        if not results:
            print(
                f"[TOOL: get_recipe_candidates] ❌ No recipes found with similarity > {similarity_threshold}"
            )
            return "No suitable recipes found based on the hybrid search criteria."

        # Format the output into a clean string for the LLM's context window
        output_summary = "\n--- Recipe Candidates ---\n"
        for i, (recipe, score) in enumerate(results):
            output_summary += f"{i+1}. ID: {recipe.id}, Name: {recipe.name}, Difficulty: {getattr(recipe, 'difficulty', 'N/A')}, Score: {score:.3f}\n"

        print(f"[TOOL: get_recipe_candidates] ✅ Found {len(results)} recipes")
        print(
            f"[TOOL: get_recipe_candidates] Similarity scores: {[f'{score:.3f}' for _, score in results]}"
        )
        print(f"[TOOL: get_recipe_candidates] Returning: {output_summary[:200]}...")
        return output_summary
    finally:
        db.close()


@tool(args_schema=FinalPlanOutput)
def submit_weekly_plan_selection(final_recipe_ids: List[uuid.UUID]) -> str:
    """
    Called by the Agent as the final step to submit the definitive list of 7
    Recipe UUIDs that the user will follow for their weekly plan.
    """
    # This tool is purely a structural mechanism. It just returns the list it was given.
    # The LangGraph state machine handles committing this list to the WeeklyPlanService later.
    return f"Selection complete. {len(final_recipe_ids)} recipes submitted."


@tool
def generate_and_save_new_recipe(
    user_prompt: str, user_skill: str, user_goal: str
) -> str:
    """
    Generates a new, custom recipe based on the user's specific request.
    Saves the new recipe to the database and flags it for vectorization.
    """

    # Define the Recipe Generation Prompt
    system_prompt = (
        "You are ChefPath, a professional, meticulous adaptive cooking mentor. "
        "Your primary directives are: 1) Adhere strictly to the provided JSON schema. "
        "2) The recipe MUST be safe, feasible, and use common, accessible ingredients. "
        "3) Your output MUST contain only the final JSON object, with no introductory text, commentary, or Markdown fences (```json)."
    )
    user_request_prompt = f"""
      Generate a new, unique recipe. Ensure the following constraints are met:

      CRITERIA:
      - Difficulty Level MUST be suitable for a '{user_skill}' user.
      - Primary Cooking Goal MUST align with: {user_goal}.
      - Cuisine/Style MUST satisfy the user's specific request.

      RECIPE DETAILS:
      - Name: Must be creative and descriptive.
      - Ingredients: The list in the JSON string must be precise (e.g., '2 tbsp soy sauce').
      - Instructions: Must be step-by-step and clear.

      USER SPECIFIC REQUEST: {user_prompt}
    """

    # Bind the structured schema to the LLM call
    structured_llm = GENERATIVE_LLM.with_structured_output(NewRecipeSchema)

    try:
        # Invoke the LLM to generate the structured recipe object
        generated_recipe_data: NewRecipeSchema = structured_llm.invoke(
            [
                HumanMessage(content=system_prompt),
                HumanMessage(content=user_request_prompt),
            ]
        )

        # Save to Database and Trigger Vectorization
        with Session(
            bind=engine
        ) as db:  # Use direct engine bind for a transactional script
            # Ensure the ingredients field is converted correctly
            ingredients_text = generated_recipe_data.ingredients

            # Create the final recipe object
            new_recipe = Recipe(
                name=generated_recipe_data.name,
                cuisine=generated_recipe_data.cuisine,
                ingredients=ingredients_text,
                instructions=generated_recipe_data.instructions,
                difficulty=generated_recipe_data.difficulty,
                is_ai_generated=True,  # Flag this as an LLM creation
                # content_text is left NULL, triggering the background vectorization
            )

            db.add(new_recipe)
            db.commit()
            db.refresh(new_recipe)

            # --- SYNCHRONOUS VECTORIZATION (Blocking the API thread) ---
            # TODO: use BackGround Tasks for optimization
            process_single_recipe_embedding_sync(new_recipe.id, db)

        return f"Successfully generated and saved recipe '{new_recipe.name}' with ID {new_recipe.id}. It is now available for the planner."

    except Exception as e:
        print(f"Error during generative recipe creation: {e}")
        return f"Failed to generate and save recipe. Error: {type(e).__name__}"
