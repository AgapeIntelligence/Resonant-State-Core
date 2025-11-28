import pytest
from habitat.simulator import simulate_habitat, drive

def test_drive_function():
    # Test drive function with and without args
    assert drive(0.0) == 0.1  # Default amplitude
    assert drive(0.0, {"amplitude": 0.5}) == 0.5  # Custom amplitude
    assert drive(math.pi / 2) == 0.0  # Cosine at pi/2

def test_simulate_habitat_earth():
    result = simulate_habitat(earth_mode=True)
    assert result["environment"] == "Earth"
    assert result["bio_sync"] == 0.045
    assert result["drive_amp"] == 0.1
    assert 0.0 <= result["fidelity"] <= 1.0
    assert result["curvature_anomaly"] == 0.000179
    assert result["flatness"] == 0.112
    assert result["wellness"] == 0.032

def test_simulate_habitat_mars():
    result = simulate_habitat(earth_mode=False)
    assert result["environment"] == "Mars"
    assert result["bio_sync"] == 0.032
    assert result["drive_amp"] == 0.15
    assert 0.0 <= result["fidelity"] <= 1.0
    assert result["curvature_anomaly"] == 0.000179
    assert result["flatness"] == 0.206
    assert result["wellness"] == 0.015

def test_simulate_habitat_invalid_mode():
    # Test with invalid earth_mode (should default to False)
    result = simulate_habitat(earth_mode="invalid")
    assert result["environment"] == "Mars"  # Default behavior
