from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# User schemas
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    cuisine: str = Field(..., min_length=1, max_length=50)
    frequency: int = Field(..., ge=1, le=7)  # 1-7 meals per week
    skill_level: str = Field(..., pattern="^(beginner|intermediate|advanced)$")
    course_duration: int = Field(..., ge=1, le=52)  # 1-52 weeks


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    cuisine: Optional[str] = Field(None, min_length=1, max_length=50)
    frequency: Optional[int] = Field(None, ge=1, le=7)
    skill_level: Optional[str] = Field(
        None, pattern="^(beginner|intermediate|advanced)$"
    )
    course_duration: Optional[int] = Field(None, ge=1, le=52)


class UserResponse(BaseModel):
    id: int
    name: str
    cuisine: str
    frequency: int
    skill_level: str
    course_duration: int
    created_at: datetime

    class Config:
        from_attributes = True


# Recipe schemas
class RecipeResponse(BaseModel):
    id: int
    external_id: str
    name: str
    cuisine: str
    ingredients: str
    instructions: str
    difficulty: str
    tags: Optional[str]
    image_url: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# Weekly Plan schemas
class WeeklyPlanResponse(BaseModel):
    id: int
    user_id: int
    week_number: int
    recipe_ids: str
    generated_at: datetime
    is_unlocked: bool
    recipes: List[RecipeResponse] = []

    class Config:
        from_attributes = True


# Feedback schemas
class FeedbackCreate(BaseModel):
    recipe_id: int
    week_number: int
    feedback: str = Field(..., pattern="^(too_easy|just_right|too_hard)$")


class UserRecipeProgressResponse(BaseModel):
    id: int
    user_id: int
    recipe_id: int
    week_number: int
    status: str
    feedback: Optional[str]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# Progress summary schema
class ProgressSummary(BaseModel):
    total_recipes: int
    completed_recipes: int
    current_week: int
    completion_rate: float
    skill_progression: str
