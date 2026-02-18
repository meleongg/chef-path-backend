import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import uuid
from datetime import datetime, timezone
import sys
from pathlib import Path
import hashlib

# Add parent directory to path so main and app modules can be imported
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from main import app
from app.database import get_db
from app.models import Base, User, Recipe, WeeklyPlan, UserRecipeProgress
from app.utils.auth import get_current_user
import app.utils.password as password_utils
import app.routers.auth as auth_router
import app.routers.users as users_router
import json

# Use in-memory SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


def _test_hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _test_verify_password(plain_password: str, hashed_password: str) -> bool:
    return _test_hash_password(plain_password) == hashed_password


def override_get_current_user(db: Session = Depends(get_db)) -> User:
    return db.query(User).first()


password_utils.hash_password = _test_hash_password
password_utils.verify_password = _test_verify_password
auth_router.hash_password = _test_hash_password
auth_router.verify_password = _test_verify_password
users_router.hash_password = _test_hash_password
users_router.verify_password = _test_verify_password
app.dependency_overrides[get_current_user] = override_get_current_user


@pytest.fixture
def db() -> Session:
    """Get test database session"""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client():
    """Get test client"""
    return TestClient(app)


@pytest.fixture
def test_user(db: Session):
    """Create a test user"""
    user = User(
        id=uuid.uuid4(),
        email="testuser@example.com",
        first_name="Test",
        last_name="User",
        cuisine="Italian",
        frequency=3,
        skill_level="intermediate",
        user_goal="Learn New Techniques",
        hashed_password=_test_hash_password("test123"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_recipes(db: Session) -> list:
    """Create test recipes"""
    recipes = []
    for i in range(5):
        recipe = Recipe(
            id=uuid.uuid4(),
            external_id=f"test_recipe_{i}",
            name=f"Test Recipe {i}",
            cuisine="Italian",
            ingredients=json.dumps(["ingredient1", "ingredient2"]),
            instructions="Mix and cook",
            difficulty="medium",
            tags=json.dumps(["pasta", "vegetarian"]),
            image_url="https://example.com/image.jpg",
        )
        db.add(recipe)
        recipes.append(recipe)
    db.commit()
    return recipes


@pytest.fixture
def test_plan(db: Session, test_user: User, test_recipes: list):
    """Create a test weekly plan"""
    recipe_schedule = json.dumps([{"recipe_id": str(test_recipes[0].id), "order": 0}])
    plan = WeeklyPlan(
        id=uuid.uuid4(),
        user_id=test_user.id,
        week_number=1,
        recipe_schedule=recipe_schedule,
        swap_count=0,
        generated_at=datetime.now(timezone.utc),
        is_unlocked=True,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan
