import uuid
from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from app.database import get_db
from app.models import User, WeeklyPlan
from app.services.weekly_plan import WeeklyPlanService
from app.agents.planner_agent import AdaptivePlannerAgent, PlanState

router = APIRouter(
    prefix="/plan",
    tags=["Plan Generation"],
)


def get_weekly_plan_service(db: Session = Depends(get_db)) -> WeeklyPlanService:
    return WeeklyPlanService(db=db)


@router.post("/generate/{user_id}", response_model=WeeklyPlan)
async def generate_user_plan_endpoint(
    user_id: uuid.UUID,
    # This input simulates a chat message or a goal trigger from the frontend
    initial_intent: str = "Generate my next week's plan focusing on my goals.",
    plan_service: Annotated[WeeklyPlanService, Depends(get_weekly_plan_service)] = None,
    db: Session = Depends(get_db),
):
    """
    Triggers the LangGraph Adaptive Agent to generate a new weekly plan
    based on user history and goals.
    """

    # 1. Fetch user state (Needed for LangGraph's initial state)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # 2. Define the initial state for the LangGraph agent
    # The agent starts with the user's latest input and profile data
    initial_state: PlanState = {
        "messages": [HumanMessage(content=initial_intent)],
        "user_id": user.id,
        "user_goal": user.user_goal,
        "candidate_recipes": [],  # Start empty
        "next_action": "critique",  # Force the agent to start planning immediately
    }

    try:
        # 3. Invoke the LangGraph Agent
        # The agent runs its entire cycle (retrieve, reason, critique, generate)
        final_state: PlanState = AdaptivePlannerAgent.invoke(initial_state)

        # 4. Extract Final Output from the Agent's State
        # The agent's final node should have returned the definitive list of recipe IDs.
        # We assume the last message contains the JSON structure or a specific output key.

        # NOTE: You MUST update your LangGraph Agent's final node to populate 'candidate_recipes'
        # with the final list of UUIDs before transitioning to END.

        final_recipe_ids: List[uuid.UUID] = final_state.get("candidate_recipes", [])

        if not final_recipe_ids:
            # If the agent didn't successfully find recipes, raise an error
            raise ValueError(
                "The Adaptive Planner Agent failed to select final recipe IDs."
            )

        # 5. Determine the new week number (Find max week + 1)
        last_plan = (
            db.query(WeeklyPlan)
            .filter(WeeklyPlan.user_id == user_id)
            .order_by(WeeklyPlan.week_number.desc())
            .first()
        )
        new_week_number = (last_plan.week_number + 1) if last_plan else 1

        # 6. Commit the Plan to the Database (WeeklyPlanService handles transaction)
        new_plan = plan_service.generate_weekly_plan(
            user=user,
            week_number=new_week_number,
            recipe_ids_from_agent=final_recipe_ids,  # Pass the Agent's decision
            db=db,
        )

        return new_plan

    except ValueError as e:
        # Catch the specific error if the agent fails to find recipes
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Planning Critical Error: {e}",
        )
