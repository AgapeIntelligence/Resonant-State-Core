from fastapi import APIRouter
from .auth_middleware import create_jwt

public = APIRouter()

@public.get("/health")
async def health():
    return {"status": "ok"}

@public.post("/auth/issue")
async def issue_token(node_id: str, scope: str = "earth", latency_tier: str = "low"):
    token = await create_jwt(node_id, scope, latency_tier)
    return {"access_token": token, "token_type": "bearer"}
