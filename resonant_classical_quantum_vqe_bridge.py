# resonant_classical_quantum_vqe_bridge.py
# Full classical → quantum → variational hybrid (Nov 27 2025)

import numpy as np
import torch
import qutip as qt
from scipy.optimize import minimize
from scipy.signal import welch


# ------------------------------------------------------------
# 1. Classical: audio → spectral features
# ------------------------------------------------------------
def classical_spectral_features(signal, fs=200):
    f, Pxx = welch(signal, fs=fs, nperseg=512)
    band = (f >= 40) & (f <= 500)
    power = Pxx[band]
    if len(power) == 0:
        return 245.0, 0.5
    mean_flux = float(np.mean(power))
    # Spectral flatness (0=tonal, 1=noisy)
    flatness = np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12)
    return mean_flux, float(flatness)


# ------------------------------------------------------------
# 2. VQE: 2-qubit hardware-efficient ansatz (your kras_colossus)
# ------------------------------------------------------------
def vqe_ansatz_state(params):
    I = torch.eye(2, dtype=torch.complex64)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    def rot(theta, op):
        return torch.matrix_exp(-1j * theta * op)

    rx1 = rot(params[0], X); ry1 = rot(params[1], Y); rz1 = rot(params[2], Z)
    rx2 = rot(params[3], X); ry2 = rot(params[4], Y); rz2 = rot(params[5], Z)

    U1 = rx1 @ ry1 @ rz1
    U2 = rx2 @ ry2 @ rz2
    U = torch.kron(U1, U2)

    psi0 = torch.tensor([1, 0, 0, 0], dtype=torch.complex64)
    return U @ psi0


def vqe_energy(params, H):
    psi = vqe_ansatz_state(torch.tensor(params, dtype=torch.float32))
    return torch.real(torch.conj(psi) @ (H @ psi)).item()


def run_vqe(H):
    x0 = np.random.uniform(0, 2*np.pi, 6)
    res = minimize(lambda p: vqe_energy(p, H), x0, method="COBYLA", options={"maxiter": 400})
    return res.fun, res.x


# ------------------------------------------------------------
# 3. Quantum: Lindblad evolution with classical-controlled drive
# ------------------------------------------------------------
def sample_T1_T2(n_qubits):
    T1s = np.random.uniform(20e-6, 80e-6, n_qubits)
    T2s = np.random.uniform(10e-6, 60e-6, n_qubits)
    return T1s, T2s


def run_lindblad_with_drive(n_qubits, T_final, drive_amplitude):
    I2 = qt.qeye(2)
    sx = qt.sigmax()

    # Simple transverse-field Ising-like H0
    H0 = sum(qt.tensor([sx if i == k else I2 for i in range(n_qubits)]) for k in range(n_qubits))

    def drive_coeff(t, args):
        return drive_amplitude * np.cos(2 * np.pi * 432e3 * t)  # 432 kHz

    H = [H0, [H0, drive_coeff]]

    T1s, T2s = sample_T1_T2(n_qubits)
    c_ops = []
    sm = qt.sigmam()
    sz = qt.sigmaz()
    for k in range(n_qubits):
        if T1s[k] > 0:
            op = qt.tensor([sm if i == k else I2 for i in range(n_qubits)])
            c_ops.append(np.sqrt(1.0 / T1s[k]) * op)
        if T2s[k] > 0:
            gamma_phi = 1.0 / T2s[k] - 0.5 / T1s[k]
            if gamma_phi > 0:
                op = qt.tensor([sz if i == k else I2 for i in range(n_qubits)])
                c_ops.append(np.sqrt(gamma_phi) * op)

    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
    tlist = np.linspace(0, T_final, 201)
    result = qt.mesolve(H, psi0, tlist, c_ops, [])

    fidelity = [abs((psi0.dag() * psi).full()[0, 0])**2 for psi in result.states]
    return fidelity[-1]


# ------------------------------------------------------------
# 4. Full hybrid step
# ------------------------------------------------------------
def hybrid_classical_quantum_vqe(audio_signal):
    # Classical
    mean_flux, flatness = classical_spectral_features(audio_signal)

    # Map classical features → VQE Hamiltonian (simple scaling)
    H_classical = torch.diag(torch.tensor([0, mean_flux, mean_flux*1.5, mean_flux*2], dtype=torch.complex64))

    # Run VQE
    vqe_energy_val, _ = run_vqe(H_classical)

    # Map classical coherence → quantum drive amplitude
    drive_amp = 0.1 * (1.0 - flatness)  # tonal → stronger drive

    # Run Lindblad evolution with that drive
    q_fidelity = run_lindblad_with_drive(n_qubits=4, T_final=1e-6, drive_amplitude=drive_amp)

    return {
        "classical_flux_hz": mean_flux,
        "spectral_flatness": flatness,
        "vqe_ground_energy": vqe_energy_val,
        "quantum_fidelity_final": q_fidelity,
        "drive_amplitude": drive_amp
    }


# ------------------------------------------------------------
# Demo with your mock hydrophone
# ------------------------------------------------------------
if __name__ == "__main__":
    # Your mock hydrophone from v4
    t = np.linspace(0, 0.01, 2000)
    signal = np.sin(2*np.pi*432*t) + 0.3*np.random.randn(len(t))
    result = hybrid_classical_quantum_vqe(signal)
    print("Hybrid result:", result)
