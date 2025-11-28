```python
import qutip as qt
import numpy as np
from typing import Optional, Dict

def drive(t: float, args: Optional[dict] = None) -> float:
    amplitude = args.get("amplitude", 0.1) if args else 0.1
    return amplitude * np.cos(0.5 * t)

def simulate_habitat(earth_mode: bool = False) -> Dict:
    H0 = 0.5 * qt.sigmaz()
    H1 = qt.sigmax()
    args = {"amplitude": 0.1 if earth_mode else 0.15}
    H = [H0, [H1, drive]]
    psi0 = qt.basis(2, 0)
    tlist = np.linspace(0, 10, 100)
    result = qt.mesolve(H, psi0, tlist, [], args=args)
    fidelity = np.abs(qt.fidelity(result.states[-1], psi0)) ** 2
    sync = 0.045 if earth_mode else 0.032
    flatness = 0.112 if earth_mode else 0.206
    anomaly = 0.000179
    wellness = 0.032 if earth_mode else 0.015
    return {
        "curvature_anomaly": anomaly,
        "bio_sync": sync,
        "flatness": flatness,
        "drive_amp": args["amplitude"],
        "fidelity": fidelity,
        "wellness": wellness,
        "status": "MONITOR",
        "environment": "Earth" if earth_mode else "Mars"
    }
```
