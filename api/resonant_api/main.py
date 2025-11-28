from fastapi import FastAPI, Depends
from .auth_middleware import get_current_node
from .public import public
from .protected import protected

# Global JWT protection; public routes are excluded via separate router
app = FastAPI(
    dependencies=[Depends(get_current_node)]
)

# Public endpoints (health, token issuance)
app.include_router(public)

# Protected endpoints (Mars telemetry, sensitive routes)
app.include_router(protected)
