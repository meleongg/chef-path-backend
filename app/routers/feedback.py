from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.utils.auth import get_current_user
from app.schemas import FeedbackCreate, UserRecipeProgressResponse, ProgressSummary
from app.services.weekly_plan import WeeklyPlanService
from app.models import UserRecipeProgress
from uuid import UUID

router = APIRouter()
plan_service = WeeklyPlanService()


@router.post("/feedback/{user_id}", response_model=UserRecipeProgressResponse)
async def submit_feedback(
    user_id: UUID,
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Log recipe completion and feedback (path version)"""
    success = plan_service.process_feedback(
        user_id,
        feedback_data.recipe_id,
        feedback_data.week_number,
        feedback_data.feedback,
        db,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recipe progress not found"
        )
    # Return updated progress
    progress = (
        db.query(UserRecipeProgress)
        .filter(
            UserRecipeProgress.user_id == user_id,
            UserRecipeProgress.recipe_id == feedback_data.recipe_id,
            UserRecipeProgress.week_number == feedback_data.week_number,
        )
        .first()
    )
    return progress


@router.get("/progress/{user_id}", response_model=ProgressSummary)
async def get_progress_path(
    user_id: UUID, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """Get user progress summary (path version)"""
    summary = plan_service.get_progress_summary(user_id, db)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return ProgressSummary(**summary)


@router.get(
    "/progress/{user_id}/week/{week_number}",
    response_model=List[UserRecipeProgressResponse],
)
async def get_weekly_progress(
    user_id: UUID,
    week_number: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get all recipe progress for a user in a specific week"""

    progress_records = (
        db.query(UserRecipeProgress)
        .filter(
            UserRecipeProgress.user_id == user_id,
            UserRecipeProgress.week_number == week_number,
        )
        .all()
    )
    return progress_records
