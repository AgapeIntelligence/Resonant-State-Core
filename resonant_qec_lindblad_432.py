"""
core/resonance/v5_surface_qec_clean.py

Cleaned scientific prototype for exploring decoherence + simple Hamiltonian/driving
on small multi-qubit registers with QuTiP.

NOTES:
- Full density-matrix simulation of 64 qubits is intractable (Hilbert space 2^64).
  Use stabilizer / Clifford simulators (e.g., Stim, qecsim) or tensor-network methods
  for surface-code scale simulations. This module demonstrates physically consistent
  Lindblad/QME simulation for up to ~10 qubits (practical limit depends on RAM/CPU).
- The "surface code" conceptual scaffolding is *not* implemented at scale here;
  this file focuses on physically-correct noise + dynamics and fidelity diagnostics.
"""

from typing import List, Tuple
import numpy as np
import qutip as qt

# -----------------------------
# Physical parameters & helpers
# -----------------------------

def sample_T1_T2(n_qubits: int,
                 T1_low: float = 30e-6,
                 T1_high: float = 80e-6,
                 T2_ratio: float = 1.67,
                 rng: np.random.Generator = np.random.default_rng(42)
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample T1 times uniformly and set T2 = T2_ratio * T1 (simple model).
    Returns (T1s, T2s).
    """
    T1s = rng.uniform(T1_low, T1_high, size=n_qubits)
    T2s = T2_ratio * T1s
    return T1s, T2s

def make_collapse_operators(n_qubits: int, T1s: np.ndarray, T2s: np.ndarray) -> List[qt.Qobj]:
    """
    Construct Lindblad collapse operators for independent amplitude-damping (T1)
    and pure-dephasing channels (Tphi derived from T1 and T2).
    - Amplitude damping rate gamma1 = 1/T1
    - Dephasing (pure) rate gamma_phi = 1/Tphi, with 1/T2 = 1/(2 T1) + 1/Tphi
      => gamma_phi = 1/T2 - 1/(2 T1)
    Returns list of collapse operators (qutip.Qobj) for the whole register (tensored).
    """
    c_ops = []
    for i in range(n_qubits):
        T1 = T1s[i]
        T2 = T2s[i]
        if T1 <= 0 or T2 <= 0:
            raise ValueError("T1 and T2 must be positive.")
        gamma1 = 1.0 / T1
        gamma_phi = max(0.0, 1.0 / T2 - 0.5 * (1.0 / T1))  # ensure non-negative
        # single-qubit operators
        sm = qt.sigmam()
        sz = qt.sigmaz()
        # embed into full register
        op_sm = qt.tensor([sm if j == i else qt.qeye(2) for j in range(n_qubits)])
        op_sz = qt.tensor([sz if j == i else qt.qeye(2) for j in range(n_qubits)])
        c_ops.append(np.sqrt(gamma1) * op_sm)    # amplitude damping jump
        if gamma_phi > 0:
            c_ops.append(np.sqrt(gamma_phi) * op_sz)  # pure dephasing jump
    return c_ops

# -----------------------------
# Hamiltonian construction
# -----------------------------

def build_static_hamiltonian_chain_xx(n_qubits: int, J: float = 1.0) -> qt.Qobj:
    """
    Build nearest-neighbor XX coupling Hamiltonian on a 1D chain of n_qubits:
      H = J * sum_{i=0}^{n-2} X_i X_{i+1}
    (Useful as a simple interacting Hamiltonian.)
    """
    H = 0 * qt.qeye(1)  # placeholder; will expand below
    # Precompute single-qubit operators for efficiency
    x = qt.sigmax()
    id2 = qt.qeye(2)
    terms = []
    for i in range(n_qubits - 1):
        ops = [x if k in (i, i+1) else id2 for k in range(n_qubits)]
        terms.append(qt.tensor(ops))
    H = J * sum(terms)
    return H

def drive_coeff(t: float, args: dict) -> float:
    """
    Example time-dependent scalar drive: cosine at `freq` Hz with phase.
    args must provide 'freq' and 'phase'.
    """
    freq = args.get('freq', 432.0)   # Hz
    phase = args.get('phase', 0.0)
    return np.cos(2 * np.pi * freq * t + phase)

def build_drive_operator_global_y(n_qubits: int) -> qt.Qobj:
    """
    Build a global Y operator that acts as sum_i Y_i (one-body drive).
    The time-dependent Hamiltonian is H_drive(t) = A(t) * sum_i Y_i.
    """
    y = qt.sigmay()
    id2 = qt.qeye(2)
    terms = [qt.tensor([y if k == i else id2 for k in range(n_qubits)]) for i in range(n_qubits)]
    return sum(terms)

# -----------------------------
# Initial states and observables
# -----------------------------

def product_state_ground(n_qubits: int) -> qt.Qobj:
    """Return |0...0> state for n_qubits as a Qobj ket (tensor product)."""
    zero = qt.basis(2, 0)
    return qt.tensor([zero for _ in range(n_qubits)])

def fidelity_between_states(rho_noisy: qt.Qobj, rho_ideal: qt.Qobj) -> float:
    """
    Fidelity between density-matrices rho_noisy and rho_ideal.
    Uses qutip fidelity function, which accepts density matrices or pure states.
    """
    # qutip.metrics.fidelity can accept two Qobjs (ket or density)
    return qt.metrics.fidelity(rho_noisy, rho_ideal)

# -----------------------------
# Simulation driver
# -----------------------------

def run_lindblad_simulation(n_qubits: int,
                            T_final: float,
                            n_tsteps: int,
                            T1s: np.ndarray,
                            T2s: np.ndarray,
                            J: float = 1.0,
                            drive_amplitude: float = 0.1,
                            drive_freq: float = 432.0,
                            drive_phase: float = 0.0,
                            max_qutip_states: int = 2**14
                            ) -> dict:
    """
    Run a Lindblad master equation simulation for a small register.
    Returns a dict with time list, final rho, ideal rho, and fidelities over time.
    - max_qutip_states: a safety cap to avoid accidental huge simulations.
    """
    dim = 2**n_qubits
    if dim > max_qutip_states:
        raise MemoryError(f"Requested simulation dimension {dim} > safe cap {max_qutip_states}. "
                          "Reduce n_qubits or increase your machine resources. "
                          "For large-surface-code work, use stabilizer simulators (Stim, qecsim).")

    # collapse operators
    c_ops = make_collapse_operators(n_qubits, T1s, T2s)

    # Hamiltonians
    H_static = build_static_hamiltonian_chain_xx(n_qubits, J=J)
    H_drive_op = build_drive_operator_global_y(n_qubits)

    # time-dependent Hamiltonian list for qutip: [H_static, [H_drive_op, drive_coeff]]
    H = [H_static, [drive_drive_wrapper(H_drive_op), drive_coeff]] if False else [H_static, [H_drive_op, drive_coeff]]

    # initial states: pure ground product
    psi0 = product_state_ground(n_qubits)
    rho0 = qt.ket2dm(psi0)

    # time list
    tlist = np.linspace(0.0, T_final, n_tsteps)

    # args for time-dependent function
    args = {'freq': drive_freq, 'phase': drive_phase}

    # Solve master equation
    result = qt.mesolve(H, rho0, tlist, c_ops, [], args=args, options=qt.Options(store_states=True, atol=1e-8, rtol=1e-6))

    # compute fidelities to ideal (unitary evolution under H only, no dissipation)
    # compute ideal evolution by solving Schrodinger equation for ket (pure state)
    U_result = qt.sesolve([H_static, [H_drive_op, drive_coeff]], psi0, tlist, [], args=args, options=qt.Options(store_states=True))
    fidelities = []
    for idx, rho in enumerate(result.states):
        psi_ideal = U_result.states[idx]   # ket
        fidelities.append(float(qt.metrics.fidelity(rho, psi_ideal)))

    return {
        'tlist': tlist,
        'noisy_states': result.states,
        'ideal_states': U_result.states,
        'fidelities': np.array(fidelities),
        'final_rho': result.states[-1],
        'final_psi_ideal': U_result.states[-1],
        'c_ops': c_ops,
        'H_static': H_static,
        'H_drive_op': H_drive_op
    }

# small helper to satisfy qutip callback type safety
def drive_drive_wrapper(op: qt.Qobj):
    """Return a wrapper that qutip can accept as a time-dependent operator.
    (Kept for potential advanced usage; not strictly necessary in this cleaned file.)"""
    return op

# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    # Parameters (small n_qubits for a full density-matrix demonstration)
    n_qubits = 6   # increase only if you have resources; 6 -> 64 dims = 4096
    T_final = 2e-3  # seconds
    n_tsteps = 201

    # physical T1/T2 sampling
    T1s, T2s = sample_T1_T2(n_qubits, T1_low=30e-6, T1_high=80e-6)
    print("T1s (s):", T1s)
    print("T2s (s):", T2s)

    # simulation
    try:
        sim = run_lindblad_simulation(n_qubits=n_qubits,
                                      T_final=T_final,
                                      n_tsteps=n_tsteps,
                                      T1s=T1s,
                                      T2s=T2s,
                                      J=0.5,
                                      drive_amplitude=0.05,
                                      drive_freq=432.0,
                                      drive_phase=np.pi/3)
    except MemoryError as e:
        raise

    # Print summary diagnostics
    final_fidelity = sim['fidelities'][-1]
    avg_fidelity = sim['fidelities'].mean()
    print(f"Average fidelity over run: {avg_fidelity:.6f}")
    print(f"Final fidelity: {final_fidelity:.6f}")

    # Example: show fidelity trace (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.plot(sim['tlist'], sim['fidelities'])
        plt.xlabel('Time (s)')
        plt.ylabel('Fidelity to ideal state')
        plt.title(f'Noisy vs ideal fidelity (n_qubits={n_qubits})')
        plt.grid(True)
        plt.show()
    except Exception:
        pass

    # Save final rho if desired
    # qt.qsave(sim['final_rho'], 'final_rho.qobj')  # uncomment to save
