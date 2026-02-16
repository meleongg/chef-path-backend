import json
import uuid
import psycopg2
import os
from dotenv import load_dotenv
from app.constants import GENERATIVE_MODEL
from typing import Annotated, List, Dict, Any
from fastapi import APIRouter, Depends, Body, HTTPException, status, Request
from app.agents.runtime_context import (
    set_runtime_context,
    clear_runtime_context,
    PlannerContext,
    PlannerRuntimeState,
)
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from app.database import get_db
from app.models import User, WeeklyPlan, UserRecipeProgress, Recipe
from app.services.weekly_plan import WeeklyPlanService
from app.agents.planner_agent import get_agent_with_checkpointer, PlanState
from app.schemas import (
    WeeklyPlanResponse,
    PlanGenerationInput,
    GeneralChatInput,
    AdaptiveChatResponse,
    SwapRecipeRequest,
    SwapRecipeResponse,
    RecipeResponse,
)
from app.services.intent_classifier import classify_message_intent
from app.utils.uuid_helpers import uuids_to_strs, strs_to_uuids
from app.utils.prompt_helpers import get_goal_description, get_skill_description
from app.services.weekly_plan import (
    validate_recipe_can_be_swapped,
    cleanup_swapped_recipe_progress,
)

# Load environment variables
load_dotenv()

# Check if tracing is enabled
TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

router = APIRouter()


def get_weekly_plan_service() -> WeeklyPlanService:
    return WeeklyPlanService()


def _get_general_knowledge_response(user_message: str) -> str:
    """
    Helper function to get a general knowledge response from Mise.
    Reused across multiple endpoints.
    """
    llm = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0.5)
    prompt = (
        "You are Mise, a friendly and experienced cooking mentor for the ChefPath app. "
        "Your goal is to help users find their 'mise en place'—getting organized and confident. "
        "Your primary directives are: "
        "1. Be helpful, concise, and professional. "
        "2. Stick strictly to the topic of cooking, ingredients, techniques, or kitchen facts. "
        "3. Limit your response to a maximum of 150 words. Do not provide disclaimers or commentary. "
        "Answer the user's question directly and clearly. "
        f"Question: {user_message}"
    )
    response = llm.invoke(prompt)
    return response.content


def cleanup_user_checkpoints(user_id: uuid.UUID):
    """
    Cleans up LangGraph checkpoint tables for a specific user.
    Removes entries from checkpoints, checkpoint_writes, and checkpoint_blobs tables.
    """
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print(f"[CheckpointCleanup] ⚠️ DATABASE_URL not found, skipping cleanup")
            return

        # psycopg2 needs plain postgresql:// not postgresql+psycopg://
        if database_url.startswith("postgresql+psycopg://"):
            database_url = database_url.replace(
                "postgresql+psycopg://", "postgresql://"
            )

        thread_id = str(user_id)

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        # Delete from checkpoint_writes first (foreign key constraint)
        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        writes_deleted = cur.rowcount

        # Delete from checkpoint_blobs (foreign key constraint)
        cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
        blobs_deleted = cur.rowcount

        # Delete from checkpoints
        cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        checkpoints_deleted = cur.rowcount

        conn.commit()
        cur.close()
        conn.close()

        print(f"[CheckpointCleanup] ✅ Cleaned up checkpoints for user {user_id}")
        print(
            f"[CheckpointCleanup] Deleted: {checkpoints_deleted} checkpoint(s), {writes_deleted} write(s), {blobs_deleted} blob(s)"
        )

    except Exception as e:
        print(f"[CheckpointCleanup] ⚠️ Cleanup failed: {e}")
        # Non-critical error, don't raise


@router.post("/general/{user_id}", response_model=Dict[str, str])
async def casual_chat_endpoint(
    user_id: uuid.UUID, chat_input: GeneralChatInput = Body(...)
):
    """
    Handles general knowledge questions using a low-cost LLM (Stateless).
    Bypasses the expensive LangGraph Agent and RAG tools.
    """
    try:
        response_content = _get_general_knowledge_response(chat_input.user_message)
        return {"response": response_content}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Error communicating with the general knowledge AI.",
        )


@router.post("/adaptive_chat/{user_id}", response_model=AdaptiveChatResponse)
async def adaptive_chat_endpoint(
    user_id: uuid.UUID,
    chat_input: GeneralChatInput = Body(...),
    db: Session = Depends(get_db),
):
    """
    Adaptive chat endpoint that classifies intent and routes accordingly.

    Routes:
    - general_knowledge → Fast, cheap Q&A (stateless)
    - analytics → Medium-cost query endpoint (future implementation)

    Note: Recipe modifications are now handled via dedicated /swap-recipe endpoint.

    Returns:
        AdaptiveChatResponse with response text and intent
    """
    try:
        # Step 1: Classify the user's intent (cheap, fast operation)
        print(f"[AdaptiveChat] Classifying message: {chat_input.user_message[:50]}...")
        intent = classify_message_intent(chat_input.user_message)
        print(f"[AdaptiveChat] Classified as: {intent}")

        # Step 2: Route based on intent
        if intent == "general_knowledge":
            print(f"[AdaptiveChat] Routing to: stateless Q&A")
            response_content = _get_general_knowledge_response(chat_input.user_message)
            return AdaptiveChatResponse(
                response=response_content,
                intent=intent,
            )

        elif intent == "analytics":
            print(f"[AdaptiveChat] Routing to: analytics")
            # Future: Query user's progress/stats
            return AdaptiveChatResponse(
                response="Analytics feature coming soon! You'll be able to track your progress here.",
                intent=intent,
            )

        else:
            # Fallback
            print(f"[AdaptiveChat] Routing to: fallback (general knowledge)")
            response_content = _get_general_knowledge_response(chat_input.user_message)
            return AdaptiveChatResponse(
                response=response_content,
                intent="general_knowledge",
            )

    except Exception as e:
        print(f"[AdaptiveChat] Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error processing chat message: {str(e)}",
        )


@router.post("/swap-recipe/{user_id}")
async def swap_recipe_endpoint(
    user_id: uuid.UUID,
    swap_request: SwapRecipeRequest = Body(...),
    request: Request = None,
    plan_service: Annotated[WeeklyPlanService, Depends(get_weekly_plan_service)] = None,
    db: Session = Depends(get_db),
):
    """
    Swap a single recipe in the user's weekly plan with an AI-powered replacement.

    Args:
        user_id: The user's UUID
        swap_request: Contains recipe_id_to_replace, optional week_number, and swap_context

    Returns:
        SwapRecipeResponse with old and new recipe details

    Raises:
        400: Recipe is already completed or not found in plan
        404: User or plan not found
        500: Swap operation failed
    """
    print(f"\n[SwapRecipe] Starting swap for user: {user_id}")
    print(f"[SwapRecipe] Recipe to replace: {swap_request.recipe_id_to_replace}")
    print(f"[SwapRecipe] Swap context: {swap_request.swap_context}")

    try:
        # 1. Validate user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 2. Get target week (provided or most recent)
        if swap_request.week_number is not None:
            target_plan = (
                db.query(WeeklyPlan)
                .filter(
                    WeeklyPlan.user_id == user_id,
                    WeeklyPlan.week_number == swap_request.week_number,
                )
                .first()
            )
        else:
            target_plan = (
                db.query(WeeklyPlan)
                .filter(WeeklyPlan.user_id == user_id)
                .order_by(WeeklyPlan.week_number.desc())
                .first()
            )

        if not target_plan:
            raise HTTPException(status_code=404, detail="Weekly plan not found")

        print(f"[SwapRecipe] Target week: {target_plan.week_number}")

        # 3. Load current plan and parse recipe_ids
        try:
            current_recipe_ids_str = json.loads(target_plan.recipe_ids)
            print(
                f"[SwapRecipe] Current plan has {len(current_recipe_ids_str)} recipes"
            )
        except Exception as e:
            print(f"[SwapRecipe] Error parsing recipe IDs: {e}")
            raise HTTPException(status_code=400, detail="Invalid plan data")

        # 4. Validate recipe_id_to_replace is in the plan
        recipe_id_str = str(swap_request.recipe_id_to_replace)
        if recipe_id_str not in current_recipe_ids_str:
            raise HTTPException(
                status_code=404,
                detail="Recipe not found in weekly plan",
            )

        # Get old recipe details before swap
        old_recipe = (
            db.query(Recipe)
            .filter(Recipe.id == swap_request.recipe_id_to_replace)
            .first()
        )
        if not old_recipe:
            raise HTTPException(status_code=404, detail="Recipe not found in database")

        print(f"[SwapRecipe] Swapping recipe: {old_recipe.name}")

        # 5. Validate recipe is NOT completed
        validate_recipe_can_be_swapped(
            user_id=user_id,
            recipe_id=swap_request.recipe_id_to_replace,
            week_number=target_plan.week_number,
            db=db,
        )

        # 6. Get exclusion IDs
        exclusion_ids: List[uuid.UUID] = plan_service.get_user_exclusion_ids(
            user, db, current_week=target_plan.week_number
        )

        # Add the recipe being swapped to exclusions (prevent recommending same recipe)
        if swap_request.recipe_id_to_replace not in exclusion_ids:
            exclusion_ids.append(swap_request.recipe_id_to_replace)
            print(
                f"[SwapRecipe] Added recipe {swap_request.recipe_id_to_replace} to exclusions"
            )

        exclude_ids_str = uuids_to_strs(exclusion_ids)

        # 7. Initialize runtime context with full user preferences
        runtime_context = PlannerContext(
            user_id=user.id,
            user_goal=user.user_goal,
            frequency=user.frequency,
            exclude_ids=exclusion_ids,
            skill_level=getattr(user, "skill_level", None),
            cuisine=getattr(user, "cuisine", None),
            dietary_restrictions=json.loads(
                getattr(user, "dietary_restrictions", "[]") or "[]"
            ),
            allergens=json.loads(getattr(user, "allergens", "[]") or "[]"),
            preferred_portion_size=getattr(user, "preferred_portion_size", None),
            max_prep_time_minutes=getattr(user, "max_prep_time_minutes", None),
            max_cook_time_minutes=getattr(user, "max_cook_time_minutes", None),
        )

        runtime_state = PlannerRuntimeState(
            candidate_recipes=current_recipe_ids_str.copy(),
            search_attempts=0,
            generation_attempts=0,
            is_swap_mode=True,  # Enable swap mode behavior
        )

        set_runtime_context(runtime_context, runtime_state)

        print("=" * 80)
        print("[SwapRecipe: RUNTIME CONTEXT INITIALIZED]")
        print(f"  user_id: {runtime_context.user_id}")
        print(f"  frequency: {runtime_context.frequency}")
        print(f"  cuisine: {runtime_context.cuisine}")
        print(f"  dietary_restrictions: {runtime_context.dietary_restrictions}")
        print(f"  allergens: {runtime_context.allergens}")
        print(f"  current recipes: {len(runtime_state.candidate_recipes)}")
        print("=" * 80)

        # 8. Build focused agent message with explicit swap instructions
        swap_message = f"""RECIPE SWAP REQUEST:

OLD RECIPE TO REPLACE:
- ID: {recipe_id_str}
- Name: {old_recipe.name}

CURRENT PLAN (candidate_recipes):
{current_recipe_ids_str}

USER'S SWAP REASON: {swap_request.swap_context}

YOUR TASK:
1. Search for ONE replacement recipe matching the user's cuisine and swap reason
2. Extract the new recipe ID from the search results
3. Replace {recipe_id_str} in the current plan with the new recipe ID
4. Call finalize_recipe_selection with the updated list (only ONE recipe changed)

Keep all other recipes unchanged."""

        # 9. Invoke LangGraph agent
        checkpointer = request.app.state.checkpoint_saver
        agent = get_agent_with_checkpointer(checkpointer)
        thread_id_str = f"{user_id}_swap_{target_plan.week_number}"

        agent_state: PlanState = {
            "messages": [HumanMessage(content=swap_message)],
            "user_id": str(user_id),
            "user_goal": user.user_goal,
            "candidate_recipes": current_recipe_ids_str,
            "frequency": user.frequency,
            "exclude_ids": exclude_ids_str,
        }

        print(f"[SwapRecipe] Invoking agent for swap...")

        with tracing_context(enabled=TRACING_ENABLED):
            updated_state: PlanState = agent.invoke(
                agent_state,
                config={
                    "configurable": {"thread_id": thread_id_str},
                    "recursion_limit": 25,
                },
            )

        # 10. Extract updated recipe list
        updated_recipe_ids_str: List[str] = (
            runtime_state.candidate_recipes
            or updated_state.get("candidate_recipes", [])
        )

        print(f"[SwapRecipe] Agent returned {len(updated_recipe_ids_str)} recipes")
        print(f"[SwapRecipe] Current recipes: {current_recipe_ids_str}")
        print(f"[SwapRecipe] Updated recipes: {updated_recipe_ids_str}")

        if not updated_recipe_ids_str:
            raise ValueError("No recipes selected after swap")

        # 11. Validate exactly ONE recipe changed
        changed_recipes = set(updated_recipe_ids_str) - set(current_recipe_ids_str)
        removed_recipes = set(current_recipe_ids_str) - set(updated_recipe_ids_str)

        print(f"[SwapRecipe] Added: {changed_recipes}")
        print(f"[SwapRecipe] Removed: {removed_recipes}")

        if len(changed_recipes) != 1 or len(removed_recipes) != 1:
            print(
                f"[SwapRecipe] ⚠️ Unexpected swap: {len(changed_recipes)} added, {len(removed_recipes)} removed"
            )
            print(
                f"[SwapRecipe] Swap candidates found: {runtime_state.swap_candidates}"
            )

            # If agent didn't swap correctly, try to recover
            if len(changed_recipes) >= 1:
                new_recipe_id_str = list(changed_recipes)[0]
                print(
                    f"[SwapRecipe] Recovering: Using first new recipe {new_recipe_id_str}"
                )
            elif len(runtime_state.swap_candidates) > 0:
                # Agent found candidates but didn't use them - use the first one
                new_recipe_id_str = runtime_state.swap_candidates[0]
                print(
                    f"[SwapRecipe] Recovering: Using first swap candidate {new_recipe_id_str}"
                )
                # Manually construct the correct swap
                updated_recipe_ids_str = [
                    new_recipe_id_str if rid == recipe_id_str else rid
                    for rid in current_recipe_ids_str
                ]
                print(
                    f"[SwapRecipe] Manually constructed swap: {updated_recipe_ids_str}"
                )
            else:
                error_msg = f"Agent did not provide a replacement recipe. Added: {len(changed_recipes)}, Removed: {len(removed_recipes)}"
                raise ValueError(error_msg)
        else:
            new_recipe_id_str = list(changed_recipes)[0]

        print(f"[SwapRecipe] New recipe ID: {new_recipe_id_str}")

        # Get new recipe details
        new_recipe = (
            db.query(Recipe).filter(Recipe.id == uuid.UUID(new_recipe_id_str)).first()
        )
        if not new_recipe:
            raise HTTPException(
                status_code=500, detail="New recipe not found in database"
            )

        # 12. Update WeeklyPlan.recipe_ids
        target_plan.recipe_ids = json.dumps(updated_recipe_ids_str)
        db.commit()
        db.refresh(target_plan)

        print(f"[SwapRecipe] ✅ Plan updated successfully")
        print(f"[SwapRecipe] Swapped: {old_recipe.name} → {new_recipe.name}")

        # 13. Delete old UserRecipeProgress entry
        cleanup_swapped_recipe_progress(
            user_id=user_id,
            old_recipe_id=swap_request.recipe_id_to_replace,
            week_number=target_plan.week_number,
            db=db,
        )
        db.commit()

        # 14. Return both old and new recipe details
        return SwapRecipeResponse(
            success=True,
            old_recipe=RecipeResponse.model_validate(old_recipe),
            new_recipe=RecipeResponse.model_validate(new_recipe),
            message=f"Successfully swapped {old_recipe.name} with {new_recipe.name}",
        )

    except HTTPException:
        raise
    except ValueError as e:
        print(f"[SwapRecipe] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[SwapRecipe] Exception: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Recipe swap failed: {str(e)}",
        )
    finally:
        clear_runtime_context()
        print("[SwapRecipe: RUNTIME CONTEXT CLEARED]")


@router.post("/generate/{user_id}", response_model=WeeklyPlanResponse)
async def generate_user_plan_endpoint(
    user_id: uuid.UUID,
    input: PlanGenerationInput = Body(...),
    request: Request = None,
    plan_service: Annotated[WeeklyPlanService, Depends(get_weekly_plan_service)] = None,
    db: Session = Depends(get_db),
):
    """
    Triggers the LangGraph Adaptive Agent to generate a new weekly plan
    based on user history and goals.

    Uses runtime context for efficient token management.
    """

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Always generate/regenerate week 1 for initial plan generation
    week_number = 1

    exclusion_ids: List[uuid.UUID] = plan_service.get_user_exclusion_ids(
        user, db, current_week=week_number
    )

    exclude_ids_str = uuids_to_strs(exclusion_ids)

    # Initialize runtime context (NOT passed through LLM!)
    runtime_context = PlannerContext(
        user_id=user.id,
        user_goal=user.user_goal,
        frequency=user.frequency,
        exclude_ids=exclusion_ids,
        skill_level=getattr(user, "skill_level", None),
        cuisine=getattr(user, "cuisine", None),
        dietary_restrictions=json.loads(
            getattr(user, "dietary_restrictions", "[]") or "[]"
        ),
        allergens=json.loads(getattr(user, "allergens", "[]") or "[]"),
        preferred_portion_size=getattr(user, "preferred_portion_size", None),
        max_prep_time_minutes=getattr(user, "max_prep_time_minutes", None),
        max_cook_time_minutes=getattr(user, "max_cook_time_minutes", None),
    )

    runtime_state = PlannerRuntimeState(
        candidate_recipes=[],
        search_attempts=0,
        generation_attempts=0,
        is_swap_mode=False,
    )

    # Set runtime context for tools to access
    set_runtime_context(runtime_context, runtime_state)

    # Simplified initial state (context is in runtime, not LLM messages)
    initial_state: PlanState = {
        "messages": [HumanMessage(content=input.initial_intent)],
        "user_id": str(user.id),
        "user_goal": user.user_goal,
        "candidate_recipes": [],
        "frequency": user.frequency,
        "exclude_ids": exclude_ids_str,
    }

    thread_id_str = str(user.id)

    try:
        print("=" * 80)
        print("[PHASE 1: RUNTIME CONTEXT INITIALIZED]")
        print(f"  user_id: {runtime_context.user_id}")
        print(f"  frequency: {runtime_context.frequency}")
        print(f"  exclude_ids: {len(runtime_context.exclude_ids)} recipes")
        print("=" * 80)
        print("Initial state:", initial_state)
        print("Thread ID:", thread_id_str)

        # Get checkpointer from app.state and compile agent with it
        checkpointer = request.app.state.checkpoint_saver
        agent = get_agent_with_checkpointer(checkpointer)

        # the agent runs its entire cycle (retrieve, reason, generate)
        with tracing_context(enabled=TRACING_ENABLED):
            final_state: PlanState = agent.invoke(
                initial_state,
                config={
                    "configurable": {"thread_id": thread_id_str},
                    "recursion_limit": 25,
                },
            )

        print("Final state:", final_state)

        # Sync final state from runtime (runtime state is source of truth)
        final_recipe_ids_str: List[str] = (
            runtime_state.candidate_recipes or final_state.get("candidate_recipes", [])
        )

        print("=" * 80)
        print("[PHASE 1: RUNTIME STATE SUMMARY]")
        print(f"  Search attempts: {runtime_state.search_attempts}")
        print(f"  Generation attempts: {runtime_state.generation_attempts}")
        print(f"  Final recipes: {len(final_recipe_ids_str)}")
        print("=" * 80)
        print("Final recipe IDs (strings):", final_recipe_ids_str)

        if not final_recipe_ids_str:
            print("No recipe IDs returned by agent.")
            raise ValueError(
                "The Adaptive Planner Agent failed to select final recipe IDs."
            )

        # Convert string IDs back to UUIDs for database storage
        final_recipe_ids: List[uuid.UUID] = strs_to_uuids(final_recipe_ids_str)

        new_plan = await plan_service.generate_weekly_plan(
            user=user,
            week_number=week_number,
            recipe_ids_from_agent=final_recipe_ids,
            db=db,
        )
        print("New plan:", new_plan)

        return new_plan

    except ValueError as e:
        print("ValueError:", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print("Exception:", e)
        print("DB type at error:", type(db))
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Planning Critical Error: {e}",
        )
    finally:
        # Always clean up runtime context
        clear_runtime_context()
        print("[PHASE 1: RUNTIME CONTEXT CLEARED]")


@router.get("/can_generate_next_week/{user_id}", response_model=Dict[str, Any])
async def check_next_week_eligibility(
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """
    Check if user is eligible to generate next week's plan.

    Returns:
        can_generate: boolean indicating if user can generate next week
        current_week: the current week number
        next_week: the next week number to be generated
        completion_status: progress details (completed/total recipes)
        message: human-readable status message
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Get most recent plan
    last_plan = (
        db.query(WeeklyPlan)
        .filter(WeeklyPlan.user_id == user_id)
        .order_by(WeeklyPlan.week_number.desc())
        .first()
    )

    if not last_plan:
        return {
            "can_generate": False,
            "current_week": None,
            "next_week": None,
            "completion_status": "0/0",
            "message": "No plans exist yet. Generate your first week's plan.",
        }

    current_week = last_plan.week_number

    # Check progress for current week
    week_progress = (
        db.query(UserRecipeProgress)
        .filter(
            UserRecipeProgress.user_id == user_id,
            UserRecipeProgress.week_number == current_week,
        )
        .all()
    )

    if not week_progress:
        return {
            "can_generate": False,
            "current_week": current_week,
            "next_week": current_week + 1,
            "completion_status": "0/0",
            "message": f"No progress entries found for week {current_week}.",
        }

    completed_count = sum(
        1 for p in week_progress if getattr(p, "status") == "completed"
    )
    total_count = len(week_progress)

    can_generate = completed_count == total_count

    return {
        "can_generate": can_generate,
        "current_week": current_week,
        "next_week": current_week + 1,
        "completion_status": f"{completed_count}/{total_count}",
        "message": (
            f"✅ Week {current_week} completed! Ready to generate week {current_week + 1}."
            if can_generate
            else f"Complete week {current_week} first. Progress: {completed_count}/{total_count} recipes."
        ),
    }


@router.post("/generate_next_week/{user_id}", response_model=WeeklyPlanResponse)
async def generate_next_week_plan(
    user_id: uuid.UUID,
    request: Request = None,
    plan_service: Annotated[WeeklyPlanService, Depends(get_weekly_plan_service)] = None,
    db: Session = Depends(get_db),
):
    """
    Generate the next week's meal plan.

    Requirements:
    - Current week must be completed (all recipes marked as completed)
    - Creates a new plan for week_number + 1
    - Uses AI agent to generate recipes based on user feedback and history
    """
    print(f"\n[GenerateNextWeek] Starting for user: {user_id}")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Get the most recent plan
    last_plan = (
        db.query(WeeklyPlan)
        .filter(WeeklyPlan.user_id == user_id)
        .order_by(WeeklyPlan.week_number.desc())
        .first()
    )

    if not last_plan:
        raise HTTPException(
            status_code=404,
            detail="No existing plan found. Please generate your first week's plan first.",
        )

    current_week = last_plan.week_number
    print(f"[GenerateNextWeek] Current week: {current_week}")

    # Check if current week is completed
    week_progress = (
        db.query(UserRecipeProgress)
        .filter(
            UserRecipeProgress.user_id == user_id,
            UserRecipeProgress.week_number == current_week,
        )
        .all()
    )

    if not week_progress:
        raise HTTPException(
            status_code=400,
            detail="No progress entries found for current week. Cannot generate next week.",
        )

    completed_count = sum(
        1 for p in week_progress if getattr(p, "status") == "completed"
    )

    if completed_count < len(week_progress):
        raise HTTPException(
            status_code=400,
            detail=f"Current week not completed. {completed_count}/{len(week_progress)} recipes done. Complete all recipes before generating next week.",
        )

    print(
        f"[GenerateNextWeek] ✅ Week {current_week} completed. Generating week {current_week + 1}"
    )

    # Check if next week already exists
    next_week_number = current_week + 1
    existing_next_plan = (
        db.query(WeeklyPlan)
        .filter(
            WeeklyPlan.user_id == user_id,
            WeeklyPlan.week_number == next_week_number,
        )
        .first()
    )

    if existing_next_plan:
        print(
            f"[GenerateNextWeek] ⚠️ Week {next_week_number} already exists, will regenerate"
        )

    # Get exclusion IDs (difficult recipes + recipes from last 2 weeks)
    exclusion_ids: List[uuid.UUID] = plan_service.get_user_exclusion_ids(
        user, db, current_week=next_week_number
    )

    exclude_ids_str = uuids_to_strs(exclusion_ids)

    print(
        f"[GenerateNextWeek] Excluding {len(exclusion_ids)} recipes (difficult + recent)"
    )

    # Create initial state for agent
    # Use adaptive skill level based on user feedback
    adapted_skill = plan_service.adapt_skill_level(user, next_week_number, db)

    initial_intent = (
        f"Create a Week {next_week_number} weekly meal plan for a user who prefers {user.cuisine} cuisine, "
        f"who wants {user.frequency} meals per week, has this cooking skill level: {get_skill_description(adapted_skill)}, "
        f"and whose goal is: {get_goal_description(user.user_goal)}."
    )

    initial_state: PlanState = {
        "messages": [HumanMessage(content=initial_intent)],
        "user_id": str(user.id),
        "user_goal": user.user_goal,
        "candidate_recipes": [],
        "frequency": user.frequency,
        "exclude_ids": exclude_ids_str,
    }

    thread_id_str = str(user.id)

    print(f"[GenerateNextWeek] Cleaning up old checkpoints before generation...")
    cleanup_user_checkpoints(user_id)

    try:
        print(f"[GenerateNextWeek] Initial state: {initial_state}")
        print(
            f"[GenerateNextWeek] Excluding {len(exclusion_ids)} recipe IDs: {exclude_ids_str}"
        )

        # Get checkpointer and compile agent
        checkpointer = request.app.state.checkpoint_saver
        agent = get_agent_with_checkpointer(checkpointer)

        # Initialize runtime context for the agent
        print("[PHASE 1: RUNTIME CONTEXT INITIALIZED FOR NEXT WEEK GENERATION]")
        runtime_context = PlannerContext(
            user_id=str(user.id),
            user_goal=user.user_goal,
            frequency=user.frequency,
            exclude_ids=exclude_ids_str,
            skill_level=adapted_skill,
        )
        runtime_state = PlannerRuntimeState(
            candidate_recipes=[],
            search_attempts=0,
            generation_attempts=0,
        )
        set_runtime_context(runtime_context, runtime_state)

        # Invoke agent to generate recipes
        with tracing_context(enabled=TRACING_ENABLED):
            final_state: PlanState = agent.invoke(
                initial_state,
                config={
                    "configurable": {"thread_id": thread_id_str},
                    "recursion_limit": 25,
                },
            )

        # Get results from runtime state (tools updated it directly)
        final_recipe_ids_str: List[str] = runtime_state.candidate_recipes
        print(f"[GenerateNextWeek] Agent returned {len(final_recipe_ids_str)} recipes")

        # Log runtime state statistics
        print(
            f"[PHASE 1: NEXT WEEK GENERATION RUNTIME STATE SUMMARY]\n"
            f"  candidate_recipes: {len(runtime_state.candidate_recipes)} recipes\n"
            f"  search_attempts: {runtime_state.search_attempts}\n"
            f"  generation_attempts: {runtime_state.generation_attempts}"
        )

        if not final_recipe_ids_str:
            raise ValueError("Agent failed to select recipes for next week.")

        final_recipe_ids: List[uuid.UUID] = strs_to_uuids(final_recipe_ids_str)

        # Generate the weekly plan (this will create progress entries too)
        new_plan = await plan_service.generate_weekly_plan(
            user=user,
            week_number=next_week_number,
            recipe_ids_from_agent=final_recipe_ids,
            db=db,
        )

        print(f"[GenerateNextWeek] ✅ Week {next_week_number} generated successfully")

        return new_plan

    except ValueError as e:
        print(f"[GenerateNextWeek] ValueError: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print(f"[GenerateNextWeek] Exception: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Next week generation failed: {str(e)}",
        )
    finally:
        # Clean up runtime context
        clear_runtime_context()
        print("[PHASE 1: RUNTIME CONTEXT CLEARED]")
