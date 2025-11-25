from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Boolean,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, TEXT
from pgvector.sqlalchemy import Vector
import uuid
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(
        String(255), unique=True, nullable=False, index=True
    )  # unique email for login
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
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

    # AI fields
    content_text = Column(TEXT, nullable=True)  # Full text for embedding
    embedding = Column(Vector(1536), nullable=True)  # Vector embedding
    is_ai_generated = Column(Boolean, default=False)  # Flag for AI origin


class WeeklyPlan(Base):
    __tablename__ = "weekly_plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    week_number = Column(Integer, nullable=False)  # week of the course (1-N)
    recipe_ids = Column(Text, nullable=False)  # JSON string of recipe IDs
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_unlocked = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="weekly_plans")


class UserRecipeProgress(Base):
    __tablename__ = "user_recipe_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    recipe_id = Column(UUID(as_uuid=True), ForeignKey("recipes.id"), nullable=False)
    week_number = Column(Integer, nullable=False)
    status = Column(String(20), default="not_started")  # not_started, completed
    feedback = Column(String(20))  # too_easy, just_right, too_hard
    completed_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="recipe_progress")
    recipe = relationship("Recipe", back_populates="recipe_progress")

    # Enhanced feedback for AI
    satisfaction_rating = Column(Integer, nullable=True)  # 1-5 rating
    difficulty_rating = Column(Integer, nullable=True)  # 1-5 rating


class LangGraphCheckpoint(Base):
    """
    Model for storing LangGraph Agent state checkpoints.
    Required for enabling persistence in the plan modification cycle.
    """

    __tablename__ = "langgraph_checkpoints"

    # The ID of the conversation thread (e.g., your weekly_plans.id UUID)
    thread_id = Column(String(36), primary_key=True, index=True)

    # The version of the checkpoint (Allows multiple checkpoints per thread)
    checkpoint_id = Column(String(36), primary_key=True)

    # The serialized state of the graph (JSON/BLOB)
    state = Column(Text, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Add a composite index for fast lookups by thread ID
    __table_args__ = (Index("idx_langgraph_thread_ts", thread_id, created_at.desc()),)
