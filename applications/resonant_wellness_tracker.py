#!/usr/bin/env python3
"""
resonant_wellness_tracker.py

Practical demo: Mobile wellness tracker using standard signal processing.
- Inputs: EEG (mock, 8 channels) + HRV RR-intervals + biosphere audio
- Outputs: stress, relaxation (alpha), human-biosphere sync, wellness score
- 100% scientific: Welch PSD, HRV variance, coherence, spectral flatness.

Run on iPhone via Replit or Pythonista.
"""

import numpy as np
from scipy.signal import welch, coherence


# -----------------------------------------------------------
# Basic DSP helper functions
# -----------------------------------------------------------

def spectral_flatness(psd):
    """Wiener entropy: geo_mean / arith_mean of PSD."""
    psd = np.asarray(psd) + 1e-20
    g = np.exp(np.mean(np.log(psd)))
    a = np.mean(psd)
    return float(g / a)


# -----------------------------------------------------------
# Human ↔ Biosphere Sync
# -----------------------------------------------------------

def human_biosphere_sync(eeg_signal, hrv_rr, fs):
    """
    EEG–HRV coupling via magnitude-squared coherence.
    Returns: sync_score [0-1], avg_fidelity
    """
    # Upsample HRV to EEG rate
    t_hrv = np.linspace(0, len(eeg_signal)/fs, len(hrv_rr))
    t_eeg = np.linspace(0, len(eeg_signal)/fs, len(eeg_signal))
    hrv_interp = np.interp(t_eeg, t_hrv, hrv_rr)

    f, coh = coherence(eeg_signal, hrv_interp, fs=fs, nperseg=512)

    # Sync emphasized in low-frequency band (0.05–0.15 Hz)
    lf_mask = (f >= 0.05) & (f <= 0.15)
    if not np.any(lf_mask):
        return 0.0, 0.0

    sync_score = float(np.mean(coh[lf_mask]))
    fidelity = float(np.max(coh))

    return sync_score, fidelity


def biosphere_spectral_features(audio, fs):
    """Return basic spectral flatness of biosphere audio."""
    f, Pxx = welch(audio, fs=fs, nperseg=1024)
    flatness = spectral_flatness(Pxx)
    return f, flatness


# -----------------------------------------------------------
# Wellness metrics (standard neurophysiology)
# -----------------------------------------------------------

def compute_stress_from_hrv(hrv_rr):
    """HRV variance as stress proxy: low variance → high stress."""
    diffs = np.diff(hrv_rr)
    var = np.var(diffs)
    stress = 1.0 - np.clip(var / 0.1, 0, 1)
    return float(stress)


def compute_relaxation_from_eeg(eeg_signal, fs=250.0):
    """Alpha band (8–12 Hz) mean PSD as relaxation proxy."""
    f, Pxx = welch(eeg_signal, fs=fs, nperseg=512)
    alpha_mask = (f >= 8) & (f <= 12)
    if not np.any(alpha_mask):
        return 0.0
    alpha_power = float(np.mean(Pxx[alpha_mask]))
    relaxation = np.clip(alpha_power / 1e-12, 0, 1)
    return float(relaxation)


def compute_wellness_score(stress, relaxation, biosphere_sync):
    """Combined score: relaxation × (1 - stress) × biosphere sync factor."""
    return float(relaxation * (1 - stress) * (0.5 + biosphere_sync))


# -----------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------
def resonant_wellness_tracker(eeg_epochs, hrv_rr, biosphere_audio,
                              fs_eeg=250.0, fs_audio=200.0):

    eeg_epochs = np.asarray(eeg_epochs)
    eeg_composite = np.mean(eeg_epochs, axis=0)

    # Human sync (EEG ↔ HRV)
    human_sync, human_fid = human_biosphere_sync(eeg_composite, hrv_rr, fs_eeg)

    # Stress & relaxation
    stress = compute_stress_from_hrv(hrv_rr)
    relaxation = compute_relaxation_from_eeg(eeg_composite, fs_eeg)

    # Biosphere: spectral flatness
    _, flatness = biosphere_spectral_features(biosphere_audio, fs_audio)
    biosphere_sync_factor = 1.0 - flatness

    # Combined
    wellness = compute_wellness_score(stress, relaxation,
                                      human_sync * biosphere_sync_factor)

    return {
        "stress_level": stress,
        "relaxation_level": relaxation,
        "human_biosphere_sync": human_sync,
        "human_fidelity": human_fid,
        "biosphere_flatness": flatness,
        "wellness_score": wellness,
        "recommendation": "Breathe deeply" if wellness < 0.5 else "Balanced state"
    }


# -----------------------------------------------------------
# Demo
# -----------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    n_samples = 2048

    # Mock EEG (8 channels, microvolt range)
    eeg = np.random.randn(8, n_samples) * 1e-6

    # Mock HRV (RR intervals ~0.8 seconds)
    hrv = 0.8 + 0.05 * np.random.randn(n_samples // 4)
    hrv = np.cumsum(hrv)

    # Mock biosphere (reef-like noise)
    t = np.linspace(0, 10, n_samples)
    biosphere = np.sin(2 * np.pi * 432 * t) + 0.3 * np.random.randn(n_samples)

    metrics = resonant_wellness_tracker(eeg, hrv, biosphere)

    print("Resonant Wellness Tracker Demo:")
    print("-" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:22s} → {v:.3f}")
        else:
            print(f"{k:22s} → {v}")

    print("\nInterpretation:")
    if metrics["wellness_score"] < 0.3:
        print("High stress detected — 4-7-8 breathing recommended.")
    elif metrics["wellness_score"] < 0.7:
        print("Moderate balance — nature exposure advised.")
    else:
        print("Optimal state — maintain with mindful movement.")
