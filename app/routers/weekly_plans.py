from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
import json
from typing import List
from app.database import get_db
from app.utils.auth import get_current_user
from app.models import User, Recipe, WeeklyPlan
from app.schemas import WeeklyPlanResponse, RecipeResponse
from app.services.weekly_plan import WeeklyPlanService

router = APIRouter()
plan_service = WeeklyPlanService()


@router.get("/weekly-plan", response_model=WeeklyPlanResponse)
async def get_weekly_plan(
    user_id: int = Query(..., description="User ID"),
    week_number: int = Query(None, description="Specific week number (optional)"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get current week's recipes for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # If no week specified, get current week
    if week_number is None:
        week_number = plan_service.get_current_week(user, db)

    # Get or generate weekly plan
    plan = await plan_service.generate_weekly_plan(user, week_number, db)

    # Check if week is unlocked
    if not getattr(plan, "is_unlocked"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This week is not yet unlocked. Complete current week first.",
        )

    # Get recipe details
    recipe_ids = json.loads(getattr(plan, "recipe_ids"))
    recipes = db.query(Recipe).filter(Recipe.id.in_(recipe_ids)).all()

    # Create response with recipes
    plan_response = WeeklyPlanResponse.model_validate(plan)
    plan_response.recipes = [RecipeResponse.model_validate(recipe) for recipe in recipes]

    return plan_response


@router.get("/weekly-plan/{user_id}/all", response_model=List[WeeklyPlanResponse])
async def get_all_weekly_plans(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """Get all weekly plans for a user (for progress tracking)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    plans = db.query(WeeklyPlan).filter(WeeklyPlan.user_id == user_id).all()

    # Add recipe details to each plan
    response_plans = []
    for plan in plans:
        recipe_ids = json.loads(getattr(plan, "recipe_ids"))
        recipes = db.query(Recipe).filter(Recipe.id.in_(recipe_ids)).all()

        plan_response = WeeklyPlanResponse.model_validate(plan)
        plan_response.recipes = [RecipeResponse.model_validate(recipe) for recipe in recipes]
        response_plans.append(plan_response)

    return response_plans
