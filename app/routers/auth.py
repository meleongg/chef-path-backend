from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.utils.password import hash_password, verify_password
from app.schemas import (
    LoginRequest,
    TokenResponse,
    UserResponse,
    RegisterRequest,
    RegisterResponse,
)
from jose import jwt
from datetime import datetime, timedelta, timezone
from app.constants import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()


@router.post(
    "/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED
)
def register_user(data: RegisterRequest, db: Session = Depends(get_db)):
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
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"user_id": str(user.id), "exp": expire}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    user_resp = UserResponse.model_validate(user)
    return RegisterResponse(
        success=True,
        message="User registered successfully",
        access_token=token,
        user=user_resp,
    )


@router.post("/login", response_model=TokenResponse)
def login_user(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"user_id": str(user.id), "exp": expire}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    user_resp = UserResponse.model_validate(user)
    return {"access_token": token, "token_type": "bearer", "user": user_resp}
