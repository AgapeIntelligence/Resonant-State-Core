from fastapi import APIRouter, Depends, HTTPException
from .auth_middleware import get_current_node, TokenPayload

protected = APIRouter()

@protected.post("/telemetry/mars/habitat")
async def mars_telemetry(payload: dict, node: TokenPayload = Depends(get_current_node)):
    """
    Mars habitat telemetry ingest (JWT-protected)
    """
    if node.scope not in ("mars", "all"):
        raise HTTPException(403, "Scope denied for Mars operations")

    # Replace with actual telemetry processing
    return {
        "status": "ok",
        "node": node.sub,
        "payload_received": True
    }
