from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app.database import get_db
from app.utils.auth import get_current_user
from app.models import User
from app.schemas import UserCreate, UserUpdate, UserResponse

router = APIRouter()


@router.put(
    "/user/profile", response_model=UserResponse, status_code=status.HTTP_200_OK
)
async def update_user_profile(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Update the authenticated user's profile after onboarding"""
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    for field, value in user_data.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


@router.get("/user/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """Get user profile by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return user


@router.put("/user/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Update user profile"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Only update allowed fields (not name)
    for field, value in user_data.model_dump(exclude_unset=True).items():
        if field != "name":
            setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """List all users (for testing purposes)"""
    users = db.query(User).all()
    return users
