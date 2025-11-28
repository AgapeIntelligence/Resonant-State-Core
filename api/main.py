```python
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from habitat.simulator import simulate_habitat
from habitat.models import HabitatResponse
from core.ai.rabinet_grok import GrokResonantLayer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager

# Rate limiter setup (5 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])

@asynccontextmanager
async def lifespan(app):
    yield
    # Cleanup if needed
    pass

app = FastAPI(title="Resonant API", description="Quantum-Bio Fusion for Earth-Mars Habitats", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

layer = GrokResonantLayer(d_model=768, d_ff=3072, omega=0.75)

# WebSocket clients
active_connections: set[WebSocket] = set()

@app.get("/health")
@limiter.limit("5/minute")
async def health_check():
    return {"status": "healthy", "timestamp": "2025-11-27T19:27:00Z"}

@app.get("/mars/habitat")
@limiter.limit("5/minute")
async def get_habitat_simulation(environment: str = "earth"):
    try:
        earth_mode = (environment.lower() == "earth")
        sim_result = simulate_habitat(earth_mode=earth_mode)
        x = torch.randn(1, 10, 768) * 0.1
        y = layer(x)
        return HabitatResponse(**sim_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/resonance")
async def websocket_resonance(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()  # Expect environment (e.g., "earth" or "mars")
            earth_mode = data.lower() == "earth"
            sim_result = simulate_habitat(earth_mode=earth_mode)
            await websocket.send_json(sim_result)
    except WebSocketDisconnect:
        active_connections.remove(websocket)

# Broadcast updates (simplified)
async def broadcast_resonance_update():
    while True:
        sim_result_earth = simulate_habitat(earth_mode=True)
        sim_result_mars = simulate_habitat(earth_mode=False)
        for connection in active_connections:
            await connection.send_json({"earth": sim_result_earth, "mars": sim_result_mars})
        await asyncio.sleep(5)  # Update every 5 seconds

import asyncio
asyncio.create_task(broadcast_resonance_update())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- **Rate Limiting**: Added `slowapi` with a limit of 5 requests per minute per IP for `/health` and `/mars/habitat`.
- **WebSocket**: Implemented `/ws/resonance` for real-time updates, broadcasting Earth and Mars simulations every 5 seconds to connected clients.
- **Note**: Requires `asyncio` import (added at the bottom); ensure compatibility with your Python version.
