from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import FeedbackCreate, UserRecipeProgressResponse, ProgressSummary
from app.services.weekly_plan import WeeklyPlanService

router = APIRouter()
plan_service = WeeklyPlanService()


@router.post("/feedback", response_model=UserRecipeProgressResponse)
async def submit_feedback(
    feedback_data: FeedbackCreate,
    user_id: int = Query(..., description="User ID"),
    db: Session = Depends(get_db),
):
    """Log recipe completion and feedback"""
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
    from app.models import UserRecipeProgress

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


@router.get("/progress", response_model=ProgressSummary)
async def get_progress(
    user_id: int = Query(..., description="User ID"), db: Session = Depends(get_db)
):
    """Get user progress summary"""
    summary = plan_service.get_progress_summary(user_id, db)

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return ProgressSummary(**summary)
