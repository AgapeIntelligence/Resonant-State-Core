#!/usr/bin/env python3
"""
resonant_human_biosphere_interface.py

Strictly scientific human–biosphere signal fusion module.
- EEG (multi-channel epochs) + HRV RR-intervals → phase synchrony, coherence, bounded fidelity
- Depth-scaled robustness metric (noise resilience vs system depth)
- Zero esoterica — pure signal processing + statistics

Part of Resonant-State-Core — closes the human–reef–quantum loop.
"""

import numpy as np
from scipy.signal import hilbert, welch


# -----------------------------------------------------------
# Core signal metrics
# -----------------------------------------------------------

def compute_phase_synchrony(x: np.ndarray, y: np.ndarray) -> float:
    """Kuramoto-style phase synchrony via Hilbert transform."""
    if len(x) != len(y):
        L = min(len(x), len(y))
        x, y = x[:L], y[:L]

    phase_x = np.angle(hilbert(x))
    phase_y = np.angle(hilbert(y))
    phase_diff = phase_x - phase_y
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def compute_coherence(x: np.ndarray, y: np.ndarray, fs: float = 250.0) -> float:
    """Approximate magnitude-squared coherence via integrated cross-power."""
    f, Pxx = welch(x, fs=fs, nperseg=min(512, len(x)))
    f, Pyy = welch(y, nperseg=min(512, len(y)))
    f, Pxy = welch(x * y.conj(), fs=fs, nperseg=min(512, len(x)))  # crude proxy

    num = np.trapz(np.abs(Pxy), f)
    den = np.sqrt(np.trapz(Pxx, f) * np.trapz(Pyy, f) + 1e-12)
    return float(num / den)


def compute_state_fidelity(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized correlation mapped to [0,1] — serves as classical fidelity."""
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return float((corr + 1.0) / 2.0)


# -----------------------------------------------------------
# Main analysis function
# -----------------------------------------------------------

def multiverse_fidelity_clean(
    epochs_data: np.ndarray,
    hrv_rr: np.ndarray,
    fs: float = 250.0,
    depths: list = None,
    noise_std: float = 0.1
) -> dict:
    """
    Full human–biosphere resonance analysis.

    Parameters
    ----------
    epochs_data : ndarray
        EEG data as (n_channels, n_samples) or (n_trials, n_channels, n_samples)
    hrv_rr : ndarray
        RR-interval tachogram (1-D)
    fs : float
        EEG sampling rate in Hz
    depths : list of int
        System depths to evaluate (e.g. [16, 64, 128])
    noise_std : float
        Assumed additive noise level for depth scaling

    Returns
    -------
    dict[int, dict] — metrics per depth D
    """
    if depths is None:
        depths = [16, 64, 128]

    # Normalise input shape
    data = np.asarray(epochs_data)
    if data.ndim == 3:           # (trials, channels, samples)
        data = data.mean(axis=0)
    if data.ndim == 1:           # single channel
        data = data[np.newaxis, :]

    # Composite EEG signal (average over channels)
    eeg_composite = data.mean(axis=0)

    # Align lengths
    L = min(len(eeg_composite), len(hrv_rr))
    eeg = eeg_composite[:L]
    hrv = hrv_rr[:L]

    results = {}

    for D in depths:
        sync = compute_phase_synchrony(eeg, hrv)
        coh  = compute_coherence(eeg, hrv, fs)
        fid  = compute_state_fidelity(eeg, hrv)
        scaled = fid * (1.0 - noise_std / max(D, 1))

        results[D] = {
            "phase_synchrony": sync,
            "coherence": coh,
            "fidelity": fid,
            "depth_scaled_score": scaled,
        }

    return results


# -----------------------------------------------------------
# Demo / self-test
# -----------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    n_samples = 4096
    eeg_epochs = np.random.randn(8, n_samples) * 1e-6          # 8 channels, microvolt scale
    hrv_rr = np.cumsum(np.random.exponential(0.8, n_samples//4))  # realistic RR intervals

    metrics = multiverse_fidelity_clean(eeg_epochs, hrv_rr, fs=250.0, noise_std=0.08)

    print("Human–Biosphere Resonance Metrics")
    print("-" * 50)
    for D, m in metrics.items():
        print(f"Depth {D:3d} → "
              f"sync {m['phase_synchrony']:.3f} | "
              f"coh {m['coherence']:.3f} | "
              f"fid {m['fidelity']:.3f} | "
              f"scaled {m['depth_scaled_score']:.3f}")
