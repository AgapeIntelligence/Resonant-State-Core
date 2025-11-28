from fastapi import APIRouter
from .auth_middleware import create_jwt

public = APIRouter()

@public.get("/health")
async def health():
    """Check API status"""
    return {"status": "ok"}

@public.post("/auth/issue")
async def issue_token(node_id: str, scope: str = "earth", latency_tier: str = "low"):
    """Issue a JWT token for a node/client"""
    token = await create_jwt(node_id, scope, latency_tier)
    return {"access_token": token, "token_type": "bearer"}
