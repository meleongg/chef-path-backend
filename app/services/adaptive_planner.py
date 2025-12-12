import os
import uuid
import json
from typing import List
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import select, text
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
import openai
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
from app.utils.uuid_helpers import str_to_uuid, strs_to_uuids
from app.utils.recipe_formatters import instructions_json_to_text

load_dotenv()

CONNECTION_STRING = os.environ.get("DATABASE_URL")
if not CONNECTION_STRING:
    raise EnvironmentError("DATABASE_URL not set for Adaptive Planner Service.")

# Initialize LangSmith client
TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
langsmith_client = None

if TRACING_ENABLED:
    langsmith_client = Client(
        api_key=os.getenv("LANGCHAIN_API_KEY"),
        api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )
    print("[LangSmith] Tracing enabled")
else:
    print("[LangSmith] Tracing disabled")

# Wrap OpenAI client for tracing
openai_client = wrap_openai(openai.Client())
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

    @traceable(name="hybrid_recipe_search")
    def get_recipe_candidates_hybrid(
        self,
        user_id: uuid.UUID,
        intent_query: str,
        exclude_ids: List[uuid.UUID],
        limit: int = 10,
        similarity_threshold: float = 0.3,
    ) -> List[tuple]:
        """
        Executes a HYBRID (Vector Search + SQL Filter) query.
        1. Ranks recipes by semantic similarity (Vector Search).
        2. Excludes recipes that the user has rated poorly (SQL NOT IN filter).
        Returns a list of (Recipe, similarity_score) tuples.
        """
        query_vector = self.embeddings_client.embed_query(intent_query)
        exclusion_str_list = [str(uid) for uid in exclude_ids]

        print(
            f"[get_recipe_candidates_hybrid] Excluding {len(exclusion_str_list)} recipes: {exclusion_str_list[:5]}..."
        )

        # Convert list to string format for pgvector - embed directly in SQL
        vector_str = f"'[{','.join(map(str, query_vector))}]'::vector"

        # Build SQL query dynamically based on whether there are exclusions
        if exclusion_str_list:
            # Create a comma-separated list of UUIDs for NOT IN clause
            exclusion_placeholders = ", ".join(
                [f"'{uid}'" for uid in exclusion_str_list]
            )
            print(
                f"[get_recipe_candidates_hybrid] SQL exclusion: NOT IN ({exclusion_placeholders[:200]}...)"
            )
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
                    r.id NOT IN ({exclusion_placeholders})
                    AND (1 - (r.embedding <=> {vector_str})) > :similarity_threshold
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
@traceable(name="finalize_recipe_selection_tool")
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
@traceable(name="get_recipe_candidates_tool")
def get_recipe_candidates(
    intent_query: str,
    user_id: str,
    exclude_ids: List[str] = None,
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

    # Convert string IDs to UUIDs for database operations
    user_uuid = str_to_uuid(user_id)
    exclude_uuids = strs_to_uuids(exclude_ids)

    db = SessionLocal()
    try:
        service = AdaptivePlannerService(db=db)
        results: List[tuple] = service.get_recipe_candidates_hybrid(
            user_id=user_uuid,
            intent_query=intent_query,
            exclude_ids=exclude_uuids,
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

        # Add summary line about shortfall
        shortfall = limit - len(results)
        if shortfall > 0:
            output_summary += f"\n⚠️ Found {len(results)}/{limit} recipes. Need {shortfall} more recipe(s) to meet the requested {limit} meals.\n"
        else:
            output_summary += f"\n✓ Found all {len(results)} requested recipes.\n"

        print(f"[TOOL: get_recipe_candidates] ✅ Found {len(results)} recipes")
        print(
            f"[TOOL: get_recipe_candidates] Similarity scores: {[f'{score:.3f}' for _, score in results]}"
        )
        print(f"[TOOL: get_recipe_candidates] Returning: {output_summary[:200]}...")
        return output_summary
    finally:
        db.close()


@tool(args_schema=FinalPlanOutput)
@traceable(name="submit_weekly_plan_tool")
def submit_weekly_plan_selection(final_recipe_ids: List[str]) -> str:
    """
    Called by the Agent as the final step to submit the list of recipes
    Recipe UUID strings that the user will follow for their weekly plan.
    """
    # This tool is purely a structural mechanism. It just returns the list it was given.
    # The LangGraph state machine handles committing this list to the WeeklyPlanService later.
    return f"Selection complete. {len(final_recipe_ids)} recipes submitted."


@tool
@traceable(name="generate_and_save_recipe_tool")
def generate_and_save_new_recipe(recipe_description: str, user_id: str = None) -> str:
    """
    Generates a new, custom recipe based on the recipe description.
    Saves the new recipe to the database with embeddings for vector search.

    Args:
        recipe_description: Detailed description of the desired recipe (include cuisine, difficulty, goal)
        user_id: Optional UUID string of the user requesting the recipe

    Returns:
        A success message with the newly created recipe ID in format: "Successfully generated recipe: <uuid>"
    """
    print("\n[TOOL: generate_and_save_new_recipe] === CALLED ===")
    print(f"[TOOL] recipe_description: {recipe_description}")
    print(f"[TOOL] user_id: {user_id}")

    # Extract skill level and goal from description or use defaults
    # Simple heuristic: look for keywords
    user_skill = "medium"
    if "beginner" in recipe_description.lower() or "easy" in recipe_description.lower():
        user_skill = "beginner"
    elif (
        "advanced" in recipe_description.lower()
        or "expert" in recipe_description.lower()
        or "hard" in recipe_description.lower()
    ):
        user_skill = "advanced"

    # Extract goal from description
    user_goal = "general"
    if "technique" in recipe_description.lower():
        user_goal = "techniques"
    elif (
        "health" in recipe_description.lower()
        or "nutrition" in recipe_description.lower()
    ):
        user_goal = "health"
    elif "budget" in recipe_description.lower() or "cost" in recipe_description.lower():
        user_goal = "budget"

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
      - Ingredients: Provide a list where EACH ingredient has TWO fields:
        * "name": The ingredient name (e.g., "Cashew nuts", "Onions", "Cumin seeds")
        * "measure": The quantity/measurement (e.g., "12", "½ tbsp", "3 sliced thinly")
      - Instructions: Must be step-by-step and clear.

      USER SPECIFIC REQUEST: {recipe_description}
    """

    # Bind the structured schema to the LLM call
    structured_llm = GENERATIVE_LLM.with_structured_output(NewRecipeSchema)

    try:
        print("[TOOL] Invoking LLM to generate recipe...")
        # Invoke the LLM to generate the structured recipe object
        generated_recipe_data: NewRecipeSchema = structured_llm.invoke(
            [
                HumanMessage(content=system_prompt),
                HumanMessage(content=user_request_prompt),
            ]
        )

        print(f"[TOOL] LLM generated recipe: {generated_recipe_data.name}")

        # Save to Database and Trigger Vectorization
        with Session(
            bind=engine
        ) as db:  # Use direct engine bind for a transactional script
            # Convert ingredients list to JSON string in the correct format
            ingredients_list = [
                {"name": item.name, "measure": item.measure}
                for item in generated_recipe_data.ingredients
            ]
            ingredients_json = json.dumps(ingredients_list)

            # Create content_text for embedding generation (flatten ingredients for text)
            ingredients_text = " ".join(
                [
                    f"{item.name} {item.measure}"
                    for item in generated_recipe_data.ingredients
                ]
            )

            # Convert structured instructions to JSON and text
            instructions_list = [
                {"step": item.step, "text": item.text}
                for item in generated_recipe_data.instructions
            ]
            instructions_json = json.dumps(instructions_list)
            instructions_text = instructions_json_to_text(instructions_list)

            content_text = f"{generated_recipe_data.name} {generated_recipe_data.cuisine} {ingredients_text} {instructions_text}"

            # Create the final recipe object
            new_recipe = Recipe(
                name=generated_recipe_data.name,
                cuisine=generated_recipe_data.cuisine,
                ingredients=ingredients_json,  # Store as JSON string
                instructions=instructions_json,  # Store structured instructions as JSON
                difficulty=generated_recipe_data.difficulty,
                is_ai_generated=True,  # Flag this as an LLM creation
                external_id=f"ai-generated-{uuid.uuid4()}",  # Unique external_id per recipe
                content_text=content_text,  # Add content_text for embeddings
            )

            db.add(new_recipe)
            db.commit()
            db.refresh(new_recipe)

            print(f"[TOOL] Recipe saved to database with ID: {new_recipe.id}")

            # --- SYNCHRONOUS VECTORIZATION (Blocking the API thread) ---
            print("[TOOL] Generating embeddings for vector search...")
            process_single_recipe_embedding_sync(new_recipe.id, db)

            print(f"[TOOL: generate_and_save_new_recipe] ✅ Recipe created successfully")

            # Return in format that execute_tool can parse
            return f"Successfully generated recipe: {new_recipe.id}\nName: {new_recipe.name}\nCuisine: {new_recipe.cuisine}\nDifficulty: {new_recipe.difficulty}"

    except Exception as e:
        print(f"[TOOL: generate_and_save_new_recipe] ❌ Error: {type(e).__name__}: {e}")
        return f"Failed to generate recipe: {type(e).__name__}: {str(e)}"

    except Exception as e:
        print(f"Error during generative recipe creation: {e}")
        return f"Failed to generate and save recipe. Error: {type(e).__name__}"
