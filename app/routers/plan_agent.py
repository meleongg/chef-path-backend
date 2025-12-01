import json
import uuid
from app.constants import GENERATIVE_MODEL
from typing import Annotated, List, Dict
from fastapi import APIRouter, Depends, Body, HTTPException, status, Request
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from app.database import get_db
from app.models import User, WeeklyPlan, UserRecipeProgress
from app.services.weekly_plan import WeeklyPlanService
from app.agents.planner_agent import get_agent_with_checkpointer, PlanState
from app.schemas import WeeklyPlanResponse, PlanGenerationInput, GeneralChatInput
from app.services.intent_classifier import classify_message_intent

router = APIRouter()


def get_weekly_plan_service() -> WeeklyPlanService:
    return WeeklyPlanService()


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

    last_plan = (
        db.query(WeeklyPlan)
        .filter(WeeklyPlan.user_id == user_id)
        .order_by(WeeklyPlan.week_number.desc())
        .first()
    )

    # Determine if the last plan is fully completed.
    update_week_number = 1
    if last_plan:
        # Check progress for the *last* week number
        week_progress = (
            db.query(UserRecipeProgress)
            .filter(
                UserRecipeProgress.user_id == user_id,
                UserRecipeProgress.week_number == last_plan.week_number,
            )
            .all()
        )
        # This checks if all recipes started for that week are marked 'completed'
        all_completed = (
            all(getattr(p, "status", None) == "completed" for p in week_progress)
            and week_progress
        )

        # If last week is done, generate the next number. Otherwise, overwrite the current week.
        if all_completed:
            update_week_number = last_plan.week_number + 1
        else:
            update_week_number = last_plan.week_number

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
            week_number=update_week_number,
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
