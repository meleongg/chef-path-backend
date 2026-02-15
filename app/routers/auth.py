from fastapi import APIRouter, HTTPException, status, Depends, Request, Response
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.utils.password import hash_password, verify_password
from app.utils.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    set_refresh_cookie,
    clear_refresh_cookie,
)
from app.schemas import (
    LoginRequest,
    TokenResponse,
    UserResponse,
    RegisterRequest,
    RegisterResponse,
    AccessTokenResponse,
    MessageResponse,
)
from datetime import datetime, timezone
from app.constants import REFRESH_TOKEN_COOKIE_NAME

router = APIRouter()


@router.post(
    "/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED
)
def register_user(
    data: RegisterRequest, response: Response, db: Session = Depends(get_db)
):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == data.email).first()
    if existing_user:
        return RegisterResponse(success=False, message="Email already registered")
    hashed_pw = hash_password(data.password)
    user = User(
        first_name=data.first_name,
        last_name=data.last_name,
        email=data.email,
        hashed_password=hashed_pw,
        cuisine="",
        frequency=1,
        skill_level="beginner",
        user_goal="",
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    set_refresh_cookie(response, refresh_token)
    user_resp = UserResponse.model_validate(user)
    return RegisterResponse(
        success=True,
        message="User registered successfully",
        access_token=access_token,
        user=user_resp,
    )


@router.post("/login", response_model=TokenResponse)
def login_user(data: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    set_refresh_cookie(response, refresh_token)
    user_resp = UserResponse.model_validate(user)
    return {"access_token": access_token, "token_type": "bearer", "user": user_resp}


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh_access_token(
    request: Request, response: Response, db: Session = Depends(get_db)
):
    refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE_NAME)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    payload = decode_token(refresh_token, expected_type="refresh")
    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    access_token = create_access_token(str(user.id))
    new_refresh_token = create_refresh_token(str(user.id))
    set_refresh_cookie(response, new_refresh_token)
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout", response_model=MessageResponse)
def logout_user(response: Response):
    clear_refresh_cookie(response)
    return MessageResponse(message="Logged out")
