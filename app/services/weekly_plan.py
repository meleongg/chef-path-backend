import json
from sqlalchemy.orm import Session
from app.models import User, WeeklyPlan, UserRecipeProgress
from app.services.external_api import MealDBAcquisitionService
from datetime import datetime, timezone


class WeeklyPlanService:
    def __init__(self):
        self.meal_service = MealDBAcquisitionService()

    async def close(self):
        await self.meal_service.close()

    def get_current_week(self, user: User, db: Session) -> int:
        """Get the current week number for the user based on their progress"""
        # Get the latest completed week
        latest_progress = (
            db.query(UserRecipeProgress)
            .filter(
                UserRecipeProgress.user_id == user.id,
                UserRecipeProgress.status == "completed",
            )
            .order_by(UserRecipeProgress.week_number.desc())
            .first()
        )

        if not latest_progress:
            return 1  # First week

        # Check if all recipes in the latest week are completed
        week_recipes = (
            db.query(UserRecipeProgress)
            .filter(
                UserRecipeProgress.user_id == user.id,
                UserRecipeProgress.week_number == latest_progress.week_number,
            )
            .all()
        )

        completed_recipes = [
            p for p in week_recipes if getattr(p, "status") == "completed"
        ]

        if len(completed_recipes) == len(week_recipes) and len(week_recipes) > 0:
            # All recipes completed, move to next week
            next_week = getattr(latest_progress, "week_number") + 1
            return next_week  # One-week-at-a-time, no course duration limit
        else:
            # Still working on current week
            return getattr(latest_progress, "week_number")

    async def generate_weekly_plan(
        self, user: User, week_number: int, db: Session
    ) -> WeeklyPlan:
        """Generate a weekly plan for a user"""
        # Check if plan already exists
        existing_plan = (
            db.query(WeeklyPlan)
            .filter(
                WeeklyPlan.user_id == user.id, WeeklyPlan.week_number == week_number
            )
            .first()
        )

        if existing_plan:
            return existing_plan

        # Get user's feedback history to adapt difficulty
        adapted_skill = self.adapt_skill_level(user, week_number, db)

        # Get recipes from TheMealDB
        recipes = await self.meal_service.get_recipes_for_user(
            getattr(user, "cuisine"), adapted_skill, getattr(user, "frequency"), db
        )

        # Create weekly plan
        recipe_ids = [recipe.id for recipe in recipes]
        plan = WeeklyPlan(
            user_id=user.id,
            week_number=week_number,
            recipe_ids=json.dumps(recipe_ids),
            is_unlocked=(week_number == 1),  # Only first week is unlocked initially
        )

        db.add(plan)
        db.commit()
        db.refresh(plan)

        # Create progress entries for each recipe
        for recipe in recipes:
            progress = UserRecipeProgress(
                user_id=user.id,
                recipe_id=recipe.id,
                week_number=week_number,
                status="not_started",
            )
            db.add(progress)

        db.commit()
        return plan

    def adapt_skill_level(self, user: User, week_number: int, db: Session) -> str:
        """Adapt skill level based on user feedback from previous weeks"""
        if week_number == 1:
            return getattr(user, "skill_level")

        # Get feedback from previous weeks
        previous_feedback = (
            db.query(UserRecipeProgress)
            .filter(
                UserRecipeProgress.user_id == user.id,
                UserRecipeProgress.week_number < week_number,
                UserRecipeProgress.feedback.isnot(None),
            )
            .all()
        )

        if not previous_feedback:
            return getattr(user, "skill_level")

        # Analyze feedback patterns
        too_easy_count = sum(
            1 for f in previous_feedback if getattr(f, "feedback") == "too_easy"
        )
        too_hard_count = sum(
            1 for f in previous_feedback if getattr(f, "feedback") == "too_hard"
        )
        total_feedback = len(previous_feedback)

        # Calculate adaptation thresholds
        easy_threshold = 0.6  # 60% too easy
        hard_threshold = 0.4  # 40% too hard

        current_skill = getattr(user, "skill_level")

        # Adapt skill level based on feedback
        if too_easy_count / total_feedback >= easy_threshold:
            if current_skill == "beginner":
                return "intermediate"
            elif current_skill == "intermediate":
                return "advanced"
        elif too_hard_count / total_feedback >= hard_threshold:
            if current_skill == "advanced":
                return "intermediate"
            elif current_skill == "intermediate":
                return "beginner"

        return current_skill

    def unlock_next_week(self, user: User, current_week: int, db: Session) -> bool:
        """Unlock the next week if current week is completed"""
        # Check if all recipes in current week are completed
        current_week_progress = (
            db.query(UserRecipeProgress)
            .filter(
                UserRecipeProgress.user_id == user.id,
                UserRecipeProgress.week_number == current_week,
            )
            .all()
        )

        if not current_week_progress:
            return False

        completed_count = sum(
            1 for p in current_week_progress if getattr(p, "status") == "completed"
        )

        if completed_count == len(current_week_progress):
            # Unlock next week
            next_week = current_week + 1
            next_plan = (
                db.query(WeeklyPlan)
                .filter(
                    WeeklyPlan.user_id == user.id,
                    WeeklyPlan.week_number == next_week,
                )
                .first()
            )
            if next_plan:
                setattr(next_plan, "is_unlocked", True)
                db.commit()
                return True

        return False

    def process_feedback(
        self, user_id: int, recipe_id: int, week_number: int, feedback: str, db: Session
    ) -> bool:
        """Process user feedback for a recipe"""
        progress = (
            db.query(UserRecipeProgress)
            .filter(
                UserRecipeProgress.user_id == user_id,
                UserRecipeProgress.recipe_id == recipe_id,
                UserRecipeProgress.week_number == week_number,
            )
            .first()
        )

        if not progress:
            return False

        # Update progress
        setattr(progress, "status", "completed")
        setattr(progress, "feedback", feedback)
        setattr(progress, "completed_at", datetime.now(timezone.utc))

        db.commit()

        # Check if we should unlock next week
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            self.unlock_next_week(user, week_number, db)

        return True

    def get_progress_summary(self, user_id: int, db: Session) -> dict:
        """Get user progress summary"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {}

        # Get all progress entries
        all_progress = (
            db.query(UserRecipeProgress)
            .filter(UserRecipeProgress.user_id == user_id)
            .all()
        )

        completed_progress = [
            p for p in all_progress if getattr(p, "status") == "completed"
        ]

        # Calculate current week
        current_week = self.get_current_week(user, db)

        # Calculate skill progression based on feedback
        recent_feedback = [
            getattr(p, "feedback")
            for p in completed_progress[-10:]
            if getattr(p, "feedback")
        ]
        skill_progression = "stable"

        if recent_feedback:
            easy_count = recent_feedback.count("too_easy")
            hard_count = recent_feedback.count("too_hard")

            if easy_count > hard_count and easy_count > len(recent_feedback) * 0.5:
                skill_progression = "advancing"
            elif hard_count > easy_count and hard_count > len(recent_feedback) * 0.5:
                skill_progression = "needs_support"

        return {
            "total_recipes": len(all_progress),
            "completed_recipes": len(completed_progress),
            "current_week": current_week,
            "completion_rate": len(completed_progress) / len(all_progress)
            if all_progress
            else 0,
            "skill_progression": skill_progression,
        }
