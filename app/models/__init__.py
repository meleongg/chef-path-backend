from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    cuisine = Column(String(50), nullable=False)  # preferred cuisine
    frequency = Column(Integer, nullable=False)  # meals per week
    skill_level = Column(String(20), nullable=False)  # beginner, intermediate, advanced
    user_goal = Column(
        String, nullable=False
    )  # e.g., "Learn New Techniques", "Master a Cuisine", etc.
    hashed_password = Column(
        String, nullable=False
    )  # Store hashed password for authentication
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    weekly_plans = relationship("WeeklyPlan", back_populates="user")
    recipe_progress = relationship("UserRecipeProgress", back_populates="user")


class Recipe(Base):
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(50), unique=True, index=True)  # TheMealDB ID
    name = Column(String(200), nullable=False)
    cuisine = Column(String(50), nullable=False)
    ingredients = Column(Text, nullable=False)  # JSON string of ingredients
    instructions = Column(Text, nullable=False)
    difficulty = Column(String(20), nullable=False)  # easy, medium, hard
    tags = Column(Text)  # JSON string of tags
    image_url = Column(String(500))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    recipe_progress = relationship("UserRecipeProgress", back_populates="recipe")


class WeeklyPlan(Base):
    __tablename__ = "weekly_plans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    week_number = Column(Integer, nullable=False)  # week of the course (1-N)
    recipe_ids = Column(Text, nullable=False)  # JSON string of recipe IDs
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_unlocked = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="weekly_plans")


class UserRecipeProgress(Base):
    __tablename__ = "user_recipe_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    recipe_id = Column(Integer, ForeignKey("recipes.id"), nullable=False)
    week_number = Column(Integer, nullable=False)
    status = Column(String(20), default="not_started")  # not_started, completed
    feedback = Column(String(20))  # too_easy, just_right, too_hard
    completed_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="recipe_progress")
    recipe = relationship("Recipe", back_populates="recipe_progress")
