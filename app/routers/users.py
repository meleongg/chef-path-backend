from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserUpdate, UserResponse

router = APIRouter()


@router.post("/user", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_or_update_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create or update user profile"""
    # Check if user already exists (for simplicity, we'll use name as identifier)
    existing_user = db.query(User).filter(User.name == user_data.name).first()

    if existing_user:
        # Update existing user
        for field, value in user_data.model_dump(exclude_unset=True).items():
            setattr(existing_user, field, value)

        db.commit()
        db.refresh(existing_user)
        return existing_user
    else:
        # Create new user
        user = User(**user_data.model_dump())
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


@router.get("/user/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user profile by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return user


@router.put("/user/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int, user_data: UserUpdate, db: Session = Depends(get_db)
):
    """Update user profile"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Update user fields
    for field, value in user_data.model_dump(exclude_unset=True).items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)
    return user


@router.get("/users", response_model=List[UserResponse])
async def list_users(db: Session = Depends(get_db)):
    """List all users (for testing purposes)"""
    users = db.query(User).all()
    return users
