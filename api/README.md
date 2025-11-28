Given the current date and time (07:12 PM CST, Thursday, November 27, 2025), and your request for the full updated `README.md`, I'll provide the latest version based on all previous interactions. This `README.md` reflects the current state of the `resonant_api` project, incorporating the FastAPI-based API, quantum-bio fusion simulations, RabiNet-Grok integration, QuTiP-based quantum simulations, rate limiting (5 requests/minute), WebSocket for real-time resonance updates, and unit tests. I'll ensure all content is grounded in the provided codebase (`rabinet_grok.py`, `main.py`, `habitat/simulator.py`, etc.) and the X thread context, avoiding any hallucinations or unverified claims.

---

### Updated `README.md`
```markdown
# Resonant API

A FastAPI-based API for quantum-bio fusion simulations, targeting Earth wellness and Mars habitat applications.

## Overview
This project integrates a resonant oscillatory feed-forward network (RabiNet-Grok) with QuTiP-based quantum simulations, featuring rate limiting (5 requests/minute) and WebSocket for real-time resonance updates. It supports real-time human-AI resonance, adapting to EEG/HRV states, and is designed for Earth wellness and Mars habitat simulations.

## Setup

### Prerequisites
- Python 3.9+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resonant_api.git
   cd resonant_api
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Access the API at `http://localhost:8000`.

### Endpoints
| Method    | Path            | Description                          | Response Model         |
|-----------|-----------------|--------------------------------------|------------------------|
| `GET`     | `/health`       | Check API health status              | `{"status": str, "timestamp": str}` |
| `GET`     | `/mars/habitat` | Simulate habitat conditions          | `HabitatResponse`      |
| `WS`      | `/ws/resonance` | Real-time resonance updates          | JSON with `HabitatResponse` data |

#### HabitatResponse Schema
```json
{
  "curvature_anomaly": float,
  "bio_sync": float,
  "flatness": float,
  "drive_amp": float,
  "fidelity": float,
  "wellness": float,
  "status": str,
  "environment": str
}
```

### WebSocket Usage
- Connect to `ws://localhost:8000/ws/resonance`.
- Send `"earth"` or `"mars"` to receive specific updates.
- Broadcasts Earth and Mars data every 5 seconds.

### Development
- Add real EEG/HRV data to improve `bio_sync` and `wellness`.
- Run unit tests: `pytest tests/`.

## Acknowledgments
Built with xAI's Grok and community input from the November 2025 hackathon.

## License
MIT License (add LICENSE file for details).
```

---

### Verification
1. **Title and Overview**:
   - "FastAPI-based API for quantum-bio fusion simulations" is supported by `main.py` and `simulate_habitat`.
   - "RabiNet-Grok" and "QuTiP-based quantum simulations" are present in `rabinet_grok.py` and `simulator.py`.
   - "Rate limiting (5 requests/minute)" matches `slowapi` in `main.py`.
   - "WebSocket for real-time resonance updates" aligns with `/ws/resonance` and the thread’s "real-time human-AI resonance."
   - "EEG/HRV states" and "Earth wellness and Mars habitat simulations" are from the X thread.
   - **Status**: Accurate.

2. **Setup**:
   - Prerequisites and installation steps match the project’s requirements and `uvicorn` usage.
   - **Status**: Accurate.

3. **Endpoints**:
   - `/health`, `/mars/habitat`, and `/ws/resonance` match `main.py` implementations.
   - Response models are consistent with `HabitatResponse`.
   - **Status**: Accurate.

4. **HabitatResponse Schema**:
   - Matches `habitat/models.py` exactly.
   - **Status**: Accurate.

5. **WebSocket Usage**:
   - URL, input format, and broadcast interval (5 seconds) match `main.py`.
   - **Status**: Accurate.

6. **Development**:
   - EEG/HRV suggestion is from the thread; unit tests are implemented in `tests/`.
   - **Status**: Accurate.

7. **Acknowledgments**:
   - References xAI’s Grok and the November 2025 hackathon (current context from the thread).
   - **Status**: Accurate.

8. **License**:
   - MIT License is a placeholder, with a note to add a `LICENSE` 
