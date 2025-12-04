import json
import uuid
import psycopg2
import os
from dotenv import load_dotenv
from app.constants import GENERATIVE_MODEL
from typing import Annotated, List, Dict, Any
from fastapi import APIRouter, Depends, Body, HTTPException, status, Request
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from app.database import get_db
from app.models import User, WeeklyPlan, UserRecipeProgress
from app.services.weekly_plan import WeeklyPlanService
from app.agents.planner_agent import get_agent_with_checkpointer, PlanState
from app.schemas import WeeklyPlanResponse, PlanGenerationInput, GeneralChatInput
from app.services.intent_classifier import classify_message_intent

# Load environment variables
load_dotenv()

# Check if tracing is enabled
TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

router = APIRouter()


def get_weekly_plan_service() -> WeeklyPlanService:
    return WeeklyPlanService()


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
        llm = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0.5)

        prompt = (
            "You are ChefPath, a friendly and experienced cooking assistant. "
            "Your primary directives are: "
            "1. Be helpful, concise, and professional. "
            "2. Stick strictly to the topic of cooking, ingredients, techniques, or kitchen facts. "
            "3. Limit your response to a maximum of 150 words. Do not provide disclaimers or commentary. "
            "Answer the user's question directly and clearly."
            f"Question: {chat_input.user_message}"
        )

        response = llm.invoke(prompt)

        return {"response": response.content}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Error communicating with the general knowledge AI.",
        )


@router.post("/adaptive_chat/{user_id}", response_model=Dict[str, str])
async def adaptive_chat_endpoint(
    user_id: uuid.UUID,
    chat_input: GeneralChatInput = Body(...),
    db: Session = Depends(get_db),
):
    """
    Adaptive chat endpoint that classifies intent and routes accordingly.

    Routes:
    - general_knowledge → Fast, cheap Q&A (stateless)
    - plan_modification → Expensive LangGraph agent (stateful)
    - analytics → Medium-cost query endpoint (future implementation)

    Returns:
        response: The AI's response to the user
        intent: The classified intent (for frontend routing/UX)
    """
    try:
        # Step 1: Classify the user's intent (cheap, fast operation)
        print(f"[AdaptiveChat] Classifying message: {chat_input.user_message[:50]}...")
        intent = classify_message_intent(chat_input.user_message)
        print(f"[AdaptiveChat] Classified as: {intent}")

        # Step 2: Route based on intent
        if intent == "general_knowledge":
            print(f"[AdaptiveChat] Routing to: stateless Q&A")
            # Use cheap, stateless Q&A
            llm = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0.5)
            prompt = (
                "You are ChefPath, a friendly and experienced cooking assistant. "
                "Your primary directives are: "
                "1. Be helpful, concise, and professional. "
                "2. Stick strictly to the topic of cooking, ingredients, techniques, or kitchen facts. "
                "3. Limit your response to a maximum of 150 words. Do not provide disclaimers or commentary. "
                "Answer the user's question directly and clearly. "
                f"Question: {chat_input.user_message}"
            )
            response = llm.invoke(prompt)
            return {
                "response": response.content,
                "intent": intent,
            }

        elif intent == "plan_modification":
            print(
                f"[AdaptiveChat] Routing to: plan modification (requires confirmation)"
            )

            # Check if user has an existing plan
            last_plan = (
                db.query(WeeklyPlan)
                .filter(WeeklyPlan.user_id == user_id)
                .order_by(WeeklyPlan.week_number.desc())
                .first()
            )

            if not last_plan:
                return {
                    "response": "You don't have an active meal plan yet. Would you like me to create one?",
                    "intent": "plan_modification",
                    "requires_confirmation": "false",
                }

            # Return a summary of what will be modified for user confirmation
            return {
                "response": f"I understand you want to modify your Week {last_plan.week_number} meal plan. I'll help you with that change. Please confirm to proceed.",
                "intent": intent,
                "requires_confirmation": "true",
                "modification_request": chat_input.user_message,  # Pass through for confirmation
            }

        elif intent == "analytics":
            print(f"[AdaptiveChat] Routing to: analytics")
            # Future: Query user's progress/stats
            return {
                "response": "Analytics feature coming soon! You'll be able to track your progress here.",
                "intent": intent,
            }

        else:
            print(f"[AdaptiveChat] Routing to: fallback (general knowledge)")
            # Fallback to general knowledge
            llm = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0.5)
            prompt = (
                "You are ChefPath, a friendly and experienced cooking assistant. "
                f"Question: {chat_input.user_message}"
            )
            response = llm.invoke(prompt)
            return {
                "response": response.content,
                "intent": "general_knowledge",
            }

    except Exception as e:
        print(f"[AdaptiveChat] Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error processing chat message: {str(e)}",
        )


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
    """

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    exclusion_ids: List[uuid.UUID] = plan_service.get_user_exclusion_ids(user, db)

    # Always generate/regenerate week 1 for initial plan generation
    week_number = 1

    initial_state: PlanState = {
        "messages": [HumanMessage(content=input.initial_intent)],
        "user_id": user.id,
        "user_goal": user.user_goal,
        "candidate_recipes": [],
        "frequency": user.frequency,
        "exclude_ids": exclusion_ids,
    }

    thread_id_str = str(user.id)

    try:
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
        final_recipe_ids: List[uuid.UUID] = final_state.get("candidate_recipes", [])
        print("Final recipe IDs:", final_recipe_ids)

        if not final_recipe_ids:
            print("No recipe IDs returned by agent.")
            raise ValueError(
                "The Adaptive Planner Agent failed to select final recipe IDs."
            )

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

    # Get exclusion IDs (recipes user rated as too difficult)
    exclusion_ids: List[uuid.UUID] = plan_service.get_user_exclusion_ids(user, db)

    # Exclude recipes from the last 2 weeks to prevent repetition
    recent_weeks_to_exclude = 2
    recent_progress = (
        db.query(UserRecipeProgress)
        .filter(
            UserRecipeProgress.user_id == user_id,
            UserRecipeProgress.week_number
            >= (current_week - recent_weeks_to_exclude + 1),
            UserRecipeProgress.status == "completed",
        )
        .all()
    )

    recent_recipe_ids = [p.recipe_id for p in recent_progress]
    exclusion_ids.extend(recent_recipe_ids)
    exclusion_ids = list(set(exclusion_ids))  # Remove duplicates

    print(
        f"[GenerateNextWeek] Excluding {len(exclusion_ids)} recipes (difficult + recent)"
    )

    # Create initial state for agent
    # Use adaptive skill level based on user feedback
    adapted_skill = plan_service.adapt_skill_level(user, next_week_number, db)

    initial_intent = (
        f"Create a weekly meal plan for week {next_week_number} for a user who prefers {user.cuisine} cuisine, "
        f"wants {user.frequency} meals per week, is a {adapted_skill} cook, and whose goal is {user.user_goal}."
    )

    initial_state: PlanState = {
        "messages": [HumanMessage(content=initial_intent)],
        "user_id": user.id,
        "user_goal": user.user_goal,
        "candidate_recipes": [],
        "frequency": user.frequency,
        "exclude_ids": exclusion_ids,
    }

    thread_id_str = str(user.id)

    print(f"[GenerateNextWeek] Cleaning up old checkpoints before generation...")
    cleanup_user_checkpoints(user_id)

    try:
        print(f"[GenerateNextWeek] Initial state: {initial_state}")
        print(
            f"[GenerateNextWeek] Excluding {len(exclusion_ids)} recipe IDs: {exclusion_ids}"
        )

        # Get checkpointer and compile agent
        checkpointer = request.app.state.checkpoint_saver
        agent = get_agent_with_checkpointer(checkpointer)

        # Invoke agent to generate recipes
        with tracing_context(enabled=TRACING_ENABLED):
            final_state: PlanState = agent.invoke(
                initial_state,
                config={
                    "configurable": {"thread_id": thread_id_str},
                    "recursion_limit": 25,
                },
            )

        final_recipe_ids: List[uuid.UUID] = final_state.get("candidate_recipes", [])
        print(f"[GenerateNextWeek] Agent returned {len(final_recipe_ids)} recipes")

        if not final_recipe_ids:
            raise ValueError("Agent failed to select recipes for next week.")

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


@router.post("/chat/confirm_modification/{user_id}", response_model=WeeklyPlanResponse)
async def confirm_plan_modification(
    user_id: uuid.UUID,
    input: GeneralChatInput = Body(...),
    request: Request = None,
    plan_service: Annotated[WeeklyPlanService, Depends(get_weekly_plan_service)] = None,
    db: Session = Depends(get_db),
):
    """
    Executes the plan modification after user confirms.
    Uses LangGraph checkpointing to load previous conversation state.
    """
    print(f"\n[ConfirmModification] Starting for user: {user_id}")
    print(f"[ConfirmModification] Modification request: {input.user_message}")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    last_plan = (
        db.query(WeeklyPlan)
        .filter(WeeklyPlan.user_id == user_id)
        .order_by(WeeklyPlan.week_number.desc())
        .first()
    )
    if not last_plan:
        raise HTTPException(status_code=404, detail="No existing weekly plan found.")

    thread_id_str = str(user_id)

    # Get current recipe IDs from the plan
    try:
        current_recipe_ids = [
            uuid.UUID(rid) for rid in json.loads(last_plan.recipe_ids)
        ]
        print(
            f"[ConfirmModification] Current plan has {len(current_recipe_ids)} recipes"
        )
    except Exception as e:
        print(f"[ConfirmModification] Error parsing recipe IDs: {e}")
        raise HTTPException(status_code=400, detail="Invalid plan data")

    try:
        # Get checkpointer and compile agent
        checkpointer = request.app.state.checkpoint_saver
        agent = get_agent_with_checkpointer(checkpointer)

        print(f"[ConfirmModification] Loading checkpoint for thread: {thread_id_str}")

        # Load the previous checkpoint to get context
        checkpoint_config = {"configurable": {"thread_id": thread_id_str}}
        previous_checkpoint = checkpointer.get(checkpoint_config)

        if previous_checkpoint:
            print(f"[ConfirmModification] ✅ Found existing checkpoint")
            # Extract previous state
            previous_state = previous_checkpoint.get("channel_values", {})
            previous_messages = previous_state.get("messages", [])
            print(
                f"[ConfirmModification] Previous conversation had {len(previous_messages)} messages"
            )
        else:
            print(
                f"[ConfirmModification] ⚠️ No previous checkpoint found, starting fresh"
            )
            previous_messages = []

        # Get exclusion IDs
        exclusion_ids: List[uuid.UUID] = plan_service.get_user_exclusion_ids(user, db)

        # Create new state that includes:
        # 1. Previous conversation context (if any)
        # 2. Current recipe list
        # 3. New modification request
        new_state: PlanState = {
            "messages": previous_messages + [HumanMessage(content=input.user_message)],
            "user_id": user_id,
            "user_goal": user.user_goal,
            "candidate_recipes": [
                str(rid) for rid in current_recipe_ids
            ],  # Start with current recipes
            "frequency": user.frequency,
            "exclude_ids": exclusion_ids,
        }

        print(f"[ConfirmModification] Invoking agent with modification request...")

        # Invoke agent with the modification request
        with tracing_context(enabled=TRACING_ENABLED):
            updated_state: PlanState = agent.invoke(
                new_state,
                config={
                    "configurable": {"thread_id": thread_id_str},
                    "recursion_limit": 20,
                },
            )

        updated_recipe_ids: List[uuid.UUID] = updated_state.get("candidate_recipes", [])
        print(f"[ConfirmModification] Agent returned {len(updated_recipe_ids)} recipes")

        if not updated_recipe_ids:
            raise ValueError("No recipes selected after plan modification.")

        # Check if recipes actually changed
        if set(updated_recipe_ids) == set(current_recipe_ids):
            print(f"[ConfirmModification] ⚠️ No changes detected in recipe selection")
        else:
            print(f"[ConfirmModification] ✅ Recipes changed, updating plan...")

        # Update the plan with new recipe IDs
        recipe_ids_str = json.dumps([str(uid) for uid in updated_recipe_ids])
        last_plan.recipe_ids = recipe_ids_str
        db.commit()
        db.refresh(last_plan)

        print(f"[ConfirmModification] ✅ Plan updated successfully")

        return last_plan

    except ValueError as e:
        print(f"[ConfirmModification] ValueError: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print(f"[ConfirmModification] Exception: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan modification failed: {str(e)}",
        )


@router.post("/chat/{user_id}", response_model=WeeklyPlanResponse)
async def chat_modify_plan_endpoint(
    user_id: uuid.UUID,
    input: GeneralChatInput = Body(...),
    request: Request = None,
    plan_service: Annotated[WeeklyPlanService, Depends(get_weekly_plan_service)] = None,
    db: Session = Depends(get_db),
):
    """
    DEPRECATED: Use /chat/confirm_modification/{user_id} instead.
    Kept for backwards compatibility.
    """
    return await confirm_plan_modification(user_id, input, request, plan_service, db)
