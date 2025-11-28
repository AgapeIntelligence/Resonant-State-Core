from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()

def test_mars_habitat_earth():
    response = client.get("/mars/habitat?environment=earth")
    assert response.status_code == 200
    data = response.json()
    assert data["environment"] == "Earth"
    assert data["bio_sync"] == 0.045
    assert data["drive_amp"] == 0.1
    assert 0.0 <= data["fidelity"] <= 1.0

def test_mars_habitat_mars():
    response = client.get("/mars/habitat?environment=mars")
    assert response.status_code == 200
    data = response.json()
    assert data["environment"] == "Mars"
    assert data["bio_sync"] == 0.032
    assert data["drive_amp"] == 0.15
    assert 0.0 <= data["fidelity"] <= 1.0

def test_mars_habitat_invalid_environment():
    response = client.get("/mars/habitat?environment=invalid")
    assert response.status_code == 200  # Should default to Mars
    data = response.json()
    assert data["environment"] == "Mars"

def test_mars_habitat_rate_limit():
    # Simulate multiple requests to trigger rate limit
    for _ in range(6):  # Exceeds 5/minute limit
        response = client.get("/mars/habitat?environment=earth")
    assert response.status_code == 429  # Too Many Requests

# Note: WebSocket testing requires additional setup (e.g., websockets library)
# Placeholder for future implementation
def test_websocket_connection():
    # Requires websockets client; skipped for now
    pass
