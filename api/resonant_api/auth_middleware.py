import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
import redis.asyncio as redis

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
JWT_SECRET = "resonant_432_aetheris_2025"
JWT_ALG = "HS256"
TOKEN_TTL = 60 * 60 * 24 * 30  # 30 days default
bearer = HTTPBearer(auto_error=False)

# Redis client
try:
    r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
except Exception:
    r = None

# -------------------------------------------------------------------------
# TOKEN PAYLOAD
# -------------------------------------------------------------------------
class TokenPayload(BaseModel):
    sub: str
    iat: int
    exp: int
    scope: str = "earth"      # "earth" | "mars" | "reef" | "all"
    latency_tier: str = "low" # "low" (<300 ms) | "high" (>8 s RTT)

# -------------------------------------------------------------------------
# REDIS CONTEXT
# -------------------------------------------------------------------------
@asynccontextmanager
async def get_redis():
    if not r:
        yield None
        return
    try:
        await r.ping()
        yield r
    except Exception:
        yield None

# -------------------------------------------------------------------------
# JWT CREATION
# -------------------------------------------------------------------------
async def create_jwt(node_id: str, scope: str = "all", latency_tier: str = "low") -> str:
    now = int(time.time())
    payload = TokenPayload(
        sub=node_id,
        iat=now,
        exp=now + TOKEN_TTL,
        scope=scope,
        latency_tier=latency_tier,
    )
    return jwt.encode(payload.dict(), JWT_SECRET, algorithm=JWT_ALG)

# -------------------------------------------------------------------------
# MIDDLEWARE / DEPENDENCY
# -------------------------------------------------------------------------
async def get_current_node(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    request: Request = None,
):
    if not creds:
        raise HTTPException(401, "Authentication required")

    token = creds.credentials

    # JWT decode
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        token_data = TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")

    # Redis revocation
    async with get_redis() as redis_client:
        if redis_client:
            revoked = await redis_client.get(f"revoked:{token}")
            if revoked:
                raise HTTPException(401, "Token revoked")

    # Latency-tier handling
    now = int(time.time())
    if token_data.latency_tier == "high":
        grace = 15
        if token_data.exp + grace < now:
            raise HTTPException(401, "Token expired (high-latency window exceeded)")

    # Inject node info into request
    request.state.node = token_data
    return token_data
