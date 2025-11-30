import json
import uuid
from app.constants import GENERATIVE_MODEL
from typing import Annotated, List, Dict
from fastapi import APIRouter, Depends, Body, HTTPException, status
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from app.database import get_db
from app.models import User, WeeklyPlan, UserRecipeProgress
from app.services.weekly_plan import WeeklyPlanService
from app.agents.planner_agent import AdaptivePlannerAgent, PlanState
from app.schemas import WeeklyPlanResponse, PlanGenerationInput, GeneralChatInput

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


@router.post("/generate/{user_id}", response_model=WeeklyPlanResponse)
async def generate_user_plan_endpoint(
    user_id: uuid.UUID,
    input: PlanGenerationInput = Body(...),
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
        # the agent runs its entire cycle (retrieve, reason, generate)
        final_state: PlanState = AdaptivePlannerAgent.invoke(
            initial_state, config={"configurable": {"thread_id": thread_id_str}}
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


@router.post("/chat/{user_id}", response_model=WeeklyPlanResponse)
async def chat_modify_plan_endpoint(
    user_id: uuid.UUID,
    input: GeneralChatInput = Body(...),
    plan_service: Annotated[WeeklyPlanService, Depends(get_weekly_plan_service)] = None,
    db: Session = Depends(get_db),
):
    """
    Handles mid-cycle plan tweaks via chat. Loads the persistent state
    from the langgraph_checkpoints table, invokes the Agent, and updates the plan.
    """
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

    # for checkpointing
    thread_id_str = str(user_id)

    new_input: PlanState = {
        "messages": [HumanMessage(content=input.user_message)],
        "user_id": user_id,
        "user_goal": user.user_goal,
        "frequency": user.frequency,
        # candidate_recipes, messages loaded from checkpoint
    }

    try:
        # Run the agent for plan modification
        updated_state: PlanState = AdaptivePlannerAgent.invoke(
            new_input,
            config={
                # This tells the PostgresSaver which thread to load/save
                "configurable": {"thread_id": thread_id_str}
            },
        )
        updated_recipe_ids: List[uuid.UUID] = updated_state.get("candidate_recipes", [])

        if not updated_recipe_ids:
            raise ValueError("No recipes selected after plan modification.")

        # serialize the Recipe IDs back to JSON string for database storage
        recipe_ids_str = json.dumps([str(uid) for uid in updated_recipe_ids])

        # update the current plan in the DB
        last_plan.recipe_ids = recipe_ids_str
        db.commit()
        db.refresh(last_plan)

        return last_plan

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Plan Modification Error: {e}",
        )
