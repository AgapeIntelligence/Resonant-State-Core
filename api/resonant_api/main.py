from fastapi import FastAPI, Depends
from .auth_middleware import get_current_node
from .public import public
from .protected import protected

app = FastAPI(
    dependencies=[Depends(get_current_node)]  # global JWT protection
)

# Public endpoints (health, token issuance)
app.include_router(public)

# Protected endpoints
app.include_router(protected)
