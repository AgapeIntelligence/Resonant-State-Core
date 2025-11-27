#!/usr/bin/env python3
"""
Resonant-State-Core — Public REST API (FastAPI)
Live at: https://resonant.yourdomain.com/docs (after deploy)

Endpoints:
POST /mars/habitat       → Full Mars habitat resonance bridge
POST /wellness           → Human wellness tracker
GET  /klein/site/{name}  → Klein curvature for any sacred site
GET  /health             → API status

Deploy: uvicorn resonant_api:app --host 0.0.0.0 --port 8000
Cloud: Railway, Fly.io, Render, or Docker + Kubernetes
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import numpy as np
import uvicorn

# Import your core engines
from resonant_mars_habitat_bridge import mars_habitat_resonance_bridge
from resonant_wellness_tracker import resonant_wellness_tracker
from klein_resonance_map_fixed import df_out  # your verified table

app = FastAPI(
    title="Resonant-State-Core API",
    description="Biological + Geometric + Quantum Resonance Engine",
    version="1.0.0",
    contact={"name": "Agape Intelligence", "url": "https://github.com/AgapeIntelligence/Resonant-State-Core"}
)

# ———————————————————————————————
# Request Models
# ———————————————————————————————
class WellnessRequest(BaseModel):
    eeg: List[List[float]] = Field(..., description="EEG epochs: (channels, samples)")
    hrv_rr: List[float] = Field(..., description="RR intervals in seconds")
    biosphere_audio: List[float] = Field(..., description="Habitat or reef audio")

class MarsHabitatRequest(BaseModel):
    eeg: List[float] = Field(..., description="Single-channel EEG (2000 samples)")
    hrv_rr: List[float] = Field(..., description="RR intervals")
    habitat_audio: List[float] = Field(..., description="O2 pump, water, plants sound")
    site: str = Field("Bermuda Triangle", description="Geometric anchor site")


# ———————————————————————————————
# Endpoints
# ———————————————————————————————
@app.get("/health")
async def health_check():
    return {"status": "NOMINAL", "system": "Resonant-State-Core API v1.0"}

@app.get("/klein/site/{site_name}")
async def get_klein_curvature(site_name: str):
    row = df_out[df_out["Site"].str.contains(site_name, case=False)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Site not in resonance map")
    data = row.iloc[0]
    return {
        "site": data["Site"],
        "lat": data["lat_deg"],
        "lon": data["lon_deg"],
        "kappa": float(data["kappa"]),
        "curvature_anomaly": float(1.0 - data["kappa"]),
        "resonance_strength": "HIGH" if data["kappa"] > 0.9998 else "MODERATE"
    }

@app.post("/wellness")
async def wellness_endpoint(req: WellnessRequest):
    try:
        eeg_np = np.array(req.eeg)
        result = resonant_wellness_tracker(
            eeg_epochs=eeg_np,
            hrv_rr=np.array(req.hrv_rr),
            biosphere_audio=np.array(req.biosphere_audio)
        )
        return {"wellness": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mars/habitat")
async def mars_habitat_endpoint(req: MarsHabitatRequest):
    try:
        result = mars_habitat_resonance_bridge(
            eeg_signal=np.array(req.eeg),
            hrv_rr=np.array(req.hrv_rr),
            habitat_audio=np.array(req.habitat_audio),
            site=req.site
        )
        return {"mars_habitat": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ———————————————————————————————
# Run (for local testing)
# ———————————————————————————————
if __name__ == "__main__":
    print("Resonant-State-Core API starting...")
    print("OpenAPI docs: http://127.0.0.1:8000/docs")
    uvicorn.run("resonant_api:app", host="0.0.0.0", port=8000, reload=True)
