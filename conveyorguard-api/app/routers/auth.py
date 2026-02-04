"""인증 API (JWT)"""

import os
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.db import get_supabase

router = APIRouter(prefix="/auth", tags=["Auth"])

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "conveyorguard-secret-key-2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    nickname: str


def create_token(user_id: int, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": int(payload["sub"]), "role": payload["role"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/login")
async def login(body: LoginRequest):
    """로그인"""
    sb = get_supabase()
    result = sb.table("users").select("*").eq("email", body.email).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = result.data[0]
    if not pwd_context.verify(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"], user["role"])
    return {
        "success": True,
        "data": {
            "access_token": token,
            "token_type": "Bearer",
            "user_id": user["id"],
            "nickname": user["nickname"],
            "role": user["role"],
        },
    }


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    """회원가입"""
    sb = get_supabase()
    # 중복 확인
    existing = sb.table("users").select("id").eq("email", body.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = pwd_context.hash(body.password)
    result = (
        sb.table("users")
        .insert({"email": body.email, "password_hash": hashed, "nickname": body.nickname})
        .execute()
    )
    return {"success": True, "message": "Registered successfully"}


@router.get("/verify")
async def verify(user=Depends(verify_token)):
    """토큰 검증"""
    return {"success": True, "data": user}
