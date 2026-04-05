"""Nonlinear ops, SPM/XPM helpers, security metrics, QAM utils."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────────────────────────────────
# 6.  NONLINEAR ACTIVATION HELPERS + SECURITY METRICS
# ──────────────────────────────────────────────────────────────────────────────

def _saturable_absorber(E: np.ndarray, p_sat_sig: float) -> np.ndarray:
    """
    Saturable absorber nonlinear activation.

    E_out = E_in / (1 + |E_in| / p_sat_sig)

    Properties:
      - Phase preserved: arg(E_out) = arg(E_in)
      - Amplitude compressed: |E_out| = |E_in| / (1 + |E_in|/p_sat_sig)
      - Linear at |E| << p_sat_sig (gain regime)
      - Saturates toward p_sat_sig as |E| → ∞
      - Smooth (C∞), monotone, invertible — preserves information for Bob

    Physical interpretation: a saturable absorber (graphene/CNT splice) or
    EDFA gain saturation operated near the signal saturation power.
    """
    amplitude = np.abs(E)
    return E / (1.0 + amplitude / p_sat_sig)


def _saturable_absorber_inv(A: np.ndarray, p_sat_sig: float) -> np.ndarray:
    """
    Exact inverse of _saturable_absorber.

    E_in = A * p_sat_sig / (p_sat_sig - |A|)

    Valid for all |A| < p_sat_sig, which is always satisfied since the
    forward function maps to |E_out| < p_sat_sig for any finite input.
    Numerically clipped for safety near the saturation limit.
    """
    amplitude = np.abs(A)
    denom = p_sat_sig - amplitude
    # Safety clip: if |A| is numerically at or above p_sat_sig (shouldn't happen),
    # clip denominator to avoid division by zero.
    denom = np.where(np.abs(denom) < 1e-9, 1e-9 * np.sign(denom + 1e-15), denom)
    return A * p_sat_sig / denom


def _propagate_saturated(sim: "CMESimulator",
                          a0: np.ndarray,
                          pump_key: np.ndarray,
                          p_sat_sig: float,
                          length_mm: Optional[float] = None) -> np.ndarray:
    """
    Propagate complex field vector a0 through the MCF ODE with signal-induced
    gain saturation (Solution B for LR attack resistance).

    Modified gain model (Saleh & Teich Ch. 14 — signal saturation):
        g_eff(z, |E_i|²) = g_base_i · 1 / (1 + |E_i(z)|² / P_sat_sig)

    where g_base_i = Im(Δβ_i(K)) is the unsaturated gain/loss coefficient.
    The phase term Re(Δβ_i) is left unchanged (Kramers-Kronig must hold for
    both saturated and unsaturated contributions — only gain saturates, not
    the refractive index shift to first order).

    Physical mechanism: a high-power signal depletes the Er³⁺ upper-state
    population, reducing available gain per photon. This makes the effective
    propagation gain nonlinear in signal amplitude — the channel
    y = F(x, K) is no longer a linear matrix multiplication.

    IMPORTANT TRADEOFF (honest): Both Bob and Eve face the same nonlinear
    inversion problem. Bob uses a linearized H_eff (computed at QPSK RMS
    amplitude) as an approximation. At moderate saturation this approximation
    is good (BER_Bob < 2%); at strong saturation it degrades. Neither party
    obtains an asymmetric advantage from this nonlinearity alone.

    Parameters
    ----------
    sim       : CMESimulator instance (provides kappa, _beta_eff, geom)
    a0        : initial complex field vector, shape (N,)
    pump_key  : pump key K controlling Er³⁺ inversion per core
    p_sat_sig : saturation amplitude threshold (units: same as |a_i(z)|)
                Large → nearly linear. Small → strong compression.
    length_mm : fiber length [mm] (default: sim.params.length_mm)

    Returns
    -------
    a_out : complex field vector at z = length_mm, shape (N,)
    """
    L         = length_mm if length_mm is not None else sim.params.length_mm
    N         = sim.geom.N
    kappa_mat = sim.kappa                   # (N, N) coupling matrix [rad/mm]
    beta_base = sim._beta_eff(pump_key)     # complex (N,): unsaturated Δβ

    def _rhs_sat(z: float, a_flat: np.ndarray) -> np.ndarray:
        a  = a_flat[:N] + 1j * a_flat[N:]
        da = np.zeros(N, dtype=complex)
        for i in range(N):
            # Per-core signal power → gain saturation factor
            power_i = float(np.abs(a[i]) ** 2)
            sat_i   = 1.0 / (1.0 + power_i / (p_sat_sig + 1e-30))
            # Saturate the gain (imaginary Δβ); leave phase shift (real Δβ) intact
            db_sat  = complex(beta_base[i].real,
                              beta_base[i].imag * sat_i)
            da[i]   = -1j * db_sat * a[i]
            for j in range(N):
                if kappa_mat[i, j] != 0.0:
                    da[i] -= 1j * kappa_mat[i, j] * a[j]
        return np.concatenate([da.real, da.imag])

    a0_flat = np.concatenate([a0.real, a0.imag]).astype(float)
    sol = solve_ivp(_rhs_sat, (0.0, L), a0_flat,
                    method="RK45", rtol=1e-5, atol=1e-7, dense_output=False)
    return sol.y[:N, -1] + 1j * sol.y[N:, -1]


def _apply_spm(z: np.ndarray, gamma_key: np.ndarray) -> np.ndarray:
    """
    Per-core Self-Phase Modulation (SPM) phase rotation.

    For each core i: z'_i = z_i * exp(j * gamma_key_i * |z_i|^2)

    Properties:
    - Phase-only: |z'_i| = |z_i|  (amplitude unchanged)
    - Per-core: each core accumulates a different phase based on its instantaneous power
    - Key-dependent: gamma_key_i = gamma_NL * K_i (pump key controls SPM strength)

    Applied AFTER the first MCF stage: z = H1(K1)*x + noise.
    The intermediate signal z_i has varying amplitude across symbol patterns even for
    QPSK inputs (because MCF coupling mixes all 13 inputs), making the SPM phase
    symbol-dependent and thus resistant to linear regression attack.

    Parameters
    ----------
    z         : (T, Nd) or (Nd,) complex — intermediate field amplitudes
    gamma_key : (Nd,) real — effective SPM coefficient per core (key-dependent)
    """
    phi_nl = gamma_key[None, :] * np.abs(z) ** 2  # (T, Nd) nonlinear phase
    return z * np.exp(1j * phi_nl)


def _invert_spm(z_prime: np.ndarray, gamma_key: np.ndarray) -> np.ndarray:
    """
    Exact inverse of _apply_spm.

    Since |z'_i| = |z_i|, the phase correction uses z' amplitude directly:
    z_i = z'_i * exp(-j * gamma_key_i * |z'_i|^2)

    This is exact (not an approximation) because SPM preserves amplitude.
    """
    phi_nl = gamma_key[None, :] * np.abs(z_prime) ** 2
    return z_prime * np.exp(-1j * phi_nl)


def _adjacency_weight_matrix(geom, data_idx: List[int]) -> np.ndarray:
    adj_full = geom.adjacency_matrix().astype(float)
    W = adj_full[np.ix_(data_idx, data_idx)]
    row_sums = np.sum(W, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def _apply_xpm(z: np.ndarray,
               gamma_self: np.ndarray,
               gamma_cross: np.ndarray,
               W_cross: np.ndarray) -> np.ndarray:
    powers = np.abs(z) ** 2
    cross_powers = powers @ W_cross.T
    phi_nl = gamma_self[None, :] * powers + gamma_cross[None, :] * cross_powers
    return z * np.exp(1j * phi_nl)


def _invert_xpm(z_prime: np.ndarray,
                gamma_self: np.ndarray,
                gamma_cross: np.ndarray,
                W_cross: np.ndarray) -> np.ndarray:
    powers = np.abs(z_prime) ** 2
    cross_powers = powers @ W_cross.T
    phi_nl = gamma_self[None, :] * powers + gamma_cross[None, :] * cross_powers
    return z_prime * np.exp(-1j * phi_nl)


def _secrecy_capacity(H_bob: np.ndarray,
                      H_eve: np.ndarray,
                      sigma_n: Optional[float] = None,
                      snr_linear: Optional[float] = None,
                      snr_db: Optional[float] = None) -> float:
    """
    Secrecy capacity of Gaussian MIMO wiretap channel (bits/channel use).

    C_s = max(0, log2 det(I + SNR H_B H_B†) - log2 det(I + SNR H_E H_E†))

    Reference: Leung-Yan-Cheong & Hellman, IEEE Trans. Inf. Theory 24, 451 (1978)
               Li et al., IEEE Trans. Inf. Theory 56, 1666 (2010)
    """
    if snr_linear is not None:
        snr = float(snr_linear)
    elif snr_db is not None:
        snr = 10 ** (snr_db / 10.0)
    else:
        if sigma_n is None:
            raise ValueError("Provide sigma_n, snr_linear, or snr_db.")
        snr = 1.0 / (2 * sigma_n**2)

    Nd = H_bob.shape[0]
    I = np.eye(Nd)

    def cap(H):
        G = H @ H.conj().T
        eigvals = np.linalg.eigvalsh(I + snr * G)
        return float(np.sum(np.log2(np.maximum(eigvals, 1.0))))

    return max(0.0, cap(H_bob) - cap(H_eve))


def _mmse_sic_rate_qpsk(H: np.ndarray, sigma2: float, cap: float = 2.0) -> float:
    """
    Achievable rate with MMSE-SIC decoding, capped per-stream for QPSK inputs.

    The MMSE-SIC per-stream SINR is:
        SINR_k = 1 / (sigma^2 * [(H^dagger H + sigma^2 I)^{-1}]_{kk}) - 1

    This is the standard achievable rate for spatial multiplexing MIMO with
    successive interference cancellation (Foschini 1996, Telatar 1999).
    The QPSK cap of 2 bits/stream reflects the spectral efficiency ceiling of
    Gray-coded QPSK at infinite SNR.

    For the key-uncertainty secrecy model this gives a LOWER BOUND on
    Bob's achievable QPSK rate (MMSE-SIC is optimal for Gaussian channels;
    for discrete QPSK it is near-optimal at moderate SNR).

    Parameters
    ----------
    H     : (N, N) complex channel matrix
    sigma2: noise variance (scalar, uniform across streams)
    cap   : per-stream rate cap (default 2 bits for QPSK)

    Returns
    -------
    Total achievable rate in bits/channel use.
    """
    N = H.shape[0]
    G_inv = np.linalg.inv(H.conj().T @ H + sigma2 * np.eye(N))
    sinr = np.real(1.0 / np.diag(G_inv)) / sigma2 - 1.0
    sinr = np.maximum(sinr, 0.0)
    rates = np.minimum(np.log2(1.0 + sinr), cap)
    return float(np.sum(rates))


def _int_to_bits(value: int, width: int) -> np.ndarray:
    """Big-endian bit expansion for exact bit-level BER accounting."""
    return np.array([(value >> shift) & 1 for shift in range(width - 1, -1, -1)],
                    dtype=int)


def _qam_constellation_gray(M: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gray-coded square-QAM constellation and per-symbol bit labels.

    Returns
    -------
    const : (M,) complex
        Constellation points normalized to unit average power.
    labels : (M, log2(M)) int
        Exact bit labels associated with each constellation point.
    """
    m = int(np.sqrt(M))
    if m * m != M:
        raise ValueError("Only square QAM constellations are supported.")

    bits_axis = int(np.log2(m))
    axis = np.arange(-(m - 1), m, 2, dtype=float)
    gray_pos = np.arange(m) ^ (np.arange(m) >> 1)

    const = []
    labels = []
    for q_label in range(m):
        for i_label in range(m):
            const.append(axis[gray_pos[i_label]] + 1j * axis[gray_pos[q_label]])
            labels.append(np.concatenate([
                _int_to_bits(i_label, bits_axis),
                _int_to_bits(q_label, bits_axis),
            ]))

    const = np.array(const, dtype=complex)
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    return const, np.array(labels, dtype=int)


def _nearest_constellation_indices(x_hat: np.ndarray, const: np.ndarray) -> np.ndarray:
    """Nearest-neighbour detection indices for a complex constellation."""
    dist = np.abs(x_hat[:, :, None] - const[None, None, :])
    return np.argmin(dist, axis=2)


def _ase_noise_variances(pump_key: np.ndarray,
                         params: FiberParams,
                         geom: MCFGeometry) -> np.ndarray:
    """
    Per-core total noise variance = thermal + ASE contribution.

    Real EDFAs add amplified spontaneous emission proportional to gain:
        sigma^2_total_i = sigma_noise^2  +  sigma_ase_ref^2 * n_sp_i * max(G_i-1, 0)

    where:
        G_i   = exp(g_net_i * L_mm)   amplitude gain factor (from two-level model)
        n_sp_i = P_i / (P_i - P_sat)  spontaneous emission (inversion) factor
                 -> 1 at full inversion (P >> P_sat), >> 1 near transparency

    Reference: Giles & Desurvire, J. Lightwave Technol. 9, 271 (1991).
    """
    variances = np.full(geom.N, params.sigma_noise**2)
    p_sat = params.p_sat_norm
    L = params.length_mm
    for idx, core_i in enumerate(geom.erbium_core_indices):
        p = float(np.clip(pump_key[idx], 0.0, 1.0))
        g_net = params.g_max_per_mm * (p - p_sat) / (p + p_sat + 1e-12)
        G = float(np.exp(g_net * L))          # amplitude gain factor
        excess = max(G - 1.0, 0.0)            # ASE only when G > 1 (net gain)
        if p > p_sat + 1e-3:
            n_sp = p / (p - p_sat)            # spontaneous emission factor
        else:
            n_sp = 10.0                        # near transparency: large ASE
        variances[core_i] += params.sigma_ase_ref**2 * n_sp * excess
    return variances   # shape (N,)


def key_sensitivity(sim: CMESimulator,
                    pump_key: np.ndarray,
                    delta_P: float = 0.01,
                    length_mm: Optional[float] = None) -> float:
    """
    Frobenius-norm sensitivity of H to pump key perturbation δP.
    ||H(K + δP·e_i) - H(K)||_F  averaged over all key dimensions.
    """
    H0 = sim.transfer_matrix(pump_key, length_mm)
    sensitivities = []
    for i in range(len(pump_key)):
        key_pert = pump_key.copy()
        key_pert[i] = float(np.clip(pump_key[i] + delta_P, 0, 1))
        H1 = sim.transfer_matrix(key_pert, length_mm)
        sensitivities.append(np.linalg.norm(H1 - H0, 'fro'))
    return float(np.mean(sensitivities))


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ML ADVERSARY (Eve trained neural network)
