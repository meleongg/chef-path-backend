import uuid
from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AnyMessage

from app.database import get_db, SessionLocal
from app.models import User, WeeklyPlan
from app.services.weekly_plan import WeeklyPlanService

# Import the compiled agent object
from app.agents.planner_agent import AdaptivePlannerAgent, PlanState

router = APIRouter(
    prefix="/plan",
    tags=["Plan Generation"],
)


# Dependency to provide the WeeklyPlanService instance
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

        # 4. Extract Final Output (Assumed to be a list of Recipe IDs from the agent)
        # In a fully implemented agent, the final node would commit the plan or return the final IDs.
        # For simplicity here, we assume the final message contains the output:

        # NOTE: A robust agent would return structured output, not just text.
        # Placeholder for extracting the final list of UUIDs:

        # This part requires adjusting the agent to return a structured list of Recipe IDs
        # For testing, let's assume the agent's logic works and returns the IDs to be planned:

        # *** REPLACE THIS LOGIC with actual agent output parser ***
        # For now, we simulate a successful agent run by failing the old service's dependency:

        # If the agent successfully generated the final recipe IDs:
        final_recipe_ids: List[uuid.UUID] = [
            uuid.uuid4() for _ in range(user.frequency)
        ]  # MOCK UUIDs

        # 5. Commit the Plan to the Database (WeeklyPlanService handles transaction)
        # NOTE: The actual WeeklyPlanService.generate_weekly_plan should be updated
        # to accept a pre-calculated list of recipe_ids.

        # Since we cannot modify the WeeklyPlanService to accept a list yet,
        # we still hit the NotImplementedError until we refactor it:

        raise NotImplementedError(
            "Plan generation is complete, but WeeklyPlanService needs refactoring to commit the new recipe IDs list."
        )

        # Once refactored, the logic would look like this:
        # new_plan = plan_service.create_plan(user, user.current_week + 1, final_recipe_ids)
        # return new_plan

    except NotImplementedError as e:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Planning Error: {e}",
        )
