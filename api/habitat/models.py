```python
from pydantic import BaseModel

class HabitatResponse(BaseModel):
    curvature_anomaly: float
    bio_sync: float
    flatness: float
    drive_amp: float
    fidelity: float
    wellness: float
    status: str
    environment: str

    class Config:
        schema_extra = {
            "example": {
                "curvature_anomaly": 0.000179,
                "bio_sync": 0.045,
                "flatness": 0.112,
                "drive_amp": 0.1,
                "fidelity": 0.876,
                "wellness": 0.032,
                "status": "MONITOR",
                "environment": "Earth"
            }
        }
```
