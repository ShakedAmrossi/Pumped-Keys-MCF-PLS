"""Physical model: FiberParams, MCFGeometry, HexMCFGeometry, CMESimulator, MCFEncryption."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kv as bessel_K
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from mcf_pls.nonlinear import _saturable_absorber, _saturable_absorber_inv, _secrecy_capacity, _ase_noise_variances

@dataclass
class FiberParams:
    """
    Physical parameters for a silica multi-core fiber.

    The defaults follow the 13-core reference geometry used throughout the
    manuscript. Publicly citable values are kept separate from the additional
    internally anchored operating-point notes exported by the v3 calibration
    bundle.
    """
    # Refractive indices at 1550 nm (OFS / standard silica SMF)
    n_core:   float = 1.44934    # core refractive index
    delta_n:  float = 0.00479    # core–cladding index difference (new design)

    # Geometry
    core_diam_um: float = 8.2    # core diameter [μm]
    pitch_um:     float = 16.4   # nearest-neighbour pitch [μm]

    # Wavelengths
    lam_sig_um:  float = 1.550   # signal wavelength [μm]
    lam_pump_um: float = 0.976   # pump wavelength [μm]  (976 nm, not 980 nm)

    # Fiber length
    length_mm: float = 120.0     # fiber length [mm]  (within 89–240 mm range)

    # EDFA / Erbium gain model (two-level, Peroni & Tamburrini 1990)
    g_max_per_mm: float = 0.020  # max amplitude gain coefficient [1/mm]
    p_sat_norm:   float = 1.0    # normalised saturation pump power

    # Henry (linewidth enhancement) factor for Erbium at 1550 nm
    # Relates pump-induced phase shift to gain: Δβ_real = -α_H * (g/2)
    alpha_H: float = 0.3         # dimensionless; ~0.1–0.5 for Er fibre amplifiers

    # Noise model
    sigma_noise: float = 0.03    # thermal / shot noise std (normalised, per core)

    # ASE noise model (Giles & Desurvire 1991)
    # Per-core ASE variance = sigma_ase_ref^2 * n_sp * max(G - 1, 0)
    # where G = exp(g_net * L) is the amplitude gain, n_sp the inversion factor.
    # At g_max=0.10, L=150mm, P=0.8: G ~ 5.6x, n_sp ~ 2.7 → sigma_ase_ref=0.008
    # gives ASE contribution ~ 0.012 per core (total sigma ~ 0.032, ~1.5 dB NF).
    sigma_ase_ref: float = 0.008   # ASE reference coefficient (normalised)

    # Fabrication tolerance (for manufacturing sweep)
    delta_n_std_frac: float = 0.003   # std of Δn variation as fraction of delta_n

    # ── derived quantities ──────────────────────────────────────────────────
    @property
    def n_clad(self) -> float:
        return self.n_core - self.delta_n

    @property
    def core_radius_um(self) -> float:
        return self.core_diam_um / 2.0

    @property
    def NA(self) -> float:
        return float(np.sqrt(self.n_core**2 - self.n_clad**2))

    def V(self, lam_um: Optional[float] = None) -> float:
        """Normalised frequency (V number)."""
        lam = lam_um if lam_um is not None else self.lam_sig_um
        return (2 * np.pi / lam) * self.core_radius_um * self.NA

    def _W(self, V: float) -> float:
        """
        Normalised transverse field decay parameter W in cladding for LP01 mode.

        For 1.5 ≤ V ≤ 2.5: Marcuse empirical approximation (< 0.5 % error).
        For V < 1.5 or V > 2.5: numerically solve LP01 characteristic equation
            U J₁(U) / J₀(U) = W K₁(W) / K₀(W),   V² = U² + W²

        Reference: Marcuse, J. Opt. Soc. Am. 68, 103 (1978)
                   Snyder & Love, Optical Waveguide Theory (1983) ch. 12
        """
        if V <= 0.05:
            return 1e-6

        def _char(W):
            """LP01 characteristic equation residual."""
            if W <= 0 or W >= V:
                return np.sign(W) * 1e6
            U = np.sqrt(max(V**2 - W**2, 0))
            if U < 1e-9:
                return 1e6
            from scipy.special import j0, j1
            J0u = float(j0(U)); J1u = float(j1(U))
            K0w = float(bessel_K(0, W)); K1w = float(bessel_K(1, W))
            if abs(J0u) < 1e-30 or K1w < 1e-30:
                return 1e6
            return U * J1u / J0u - W * K1w / K0w

        # Marcuse approximation as starting point
        W_marc = V * (1.0 + 0.6839*V + 0.3034*V**2) / (
                     1.0 + 0.2569*V + 0.2938*V**2 + 0.1255*V**3)
        W_marc = float(np.clip(W_marc, 1e-6, V * 0.9999))

        # For our design V ≈ 1.96, Marcuse is accurate. Only refine numerically
        # when V is outside the approximation's validated range.
        if 1.4 <= V <= 2.6:
            return W_marc

        # Numerical solve for edge cases (V < 1.4 or V > 2.6)
        try:
            from scipy.optimize import brentq
            W_low  = max(1e-6, W_marc * 0.5)
            W_high = min(V * 0.9999, W_marc * 1.5)
            # Ensure bracket straddles a root
            if _char(W_low) * _char(W_high) > 0:
                return W_marc
            W_sol = brentq(_char, W_low, W_high, xtol=1e-8, maxiter=50)
            return float(np.clip(W_sol, 1e-6, V * 0.9999))
        except Exception:
            return W_marc

    def propagation_constant(self, lam_um: Optional[float] = None) -> float:
        """Phase propagation constant β [rad/μm] for LP01 mode."""
        lam = lam_um if lam_um is not None else self.lam_sig_um
        k0 = 2 * np.pi / lam
        Vv = self.V(lam)
        Wv = self._W(Vv)
        Uv = float(np.sqrt(max(Vv**2 - Wv**2, 0)))
        beta2 = (k0 * self.n_core)**2 - (Uv / self.core_radius_um)**2
        return float(np.sqrt(max(beta2, 0)))

    def coupling_coeff(self, pitch_um: Optional[float] = None,
                       lam_um: Optional[float] = None) -> float:
        """
        Inter-core coupling coefficient κ [rad/mm] between two adjacent
        identical LP01-mode cores.

        Formula (Snyder & Love 1983, eq. 29-16):
            κ ≈ (U² / (a β V²)) · K₀(W·d/a) / K₁(W)²

        where a = core radius, d = pitch, U,W,V = normalised parameters.
        Returns κ in rad/mm (converts from rad/μm by × 1000).
        """
        d  = (pitch_um if pitch_um is not None else self.pitch_um)
        lam = lam_um if lam_um is not None else self.lam_sig_um
        a   = self.core_radius_um
        Vv  = self.V(lam)
        Wv  = self._W(Vv)
        Uv  = float(np.sqrt(max(Vv**2 - Wv**2, 1e-10)))
        beta = self.propagation_constant(lam)

        norm_pitch = Wv * d / a
        norm_pitch = max(norm_pitch, 0.5)   # avoid Bessel singularity

        K0 = float(bessel_K(0, norm_pitch))
        K1 = float(bessel_K(1, max(Wv, 1e-6)))

        if K1 < 1e-20 or beta < 1e-10:
            return 0.0

        kappa_per_um = (Uv**2 / (a * beta * Vv**2)) * (K0 / K1**2)
        return float(kappa_per_um * 1000.0)   # convert to rad/mm


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MCF GEOMETRY  (13-core hexagonal)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MCFGeometry:
    """
    13-core hexagonal MCF geometry.

    Layout (in the transverse xy-plane):
        Core  0         : central core   (Erbium-doped, pump-keyed)
        Cores 1–6       : inner ring     (Erbium-doped, pump-keyed)
        Cores 7–12      : outer ring     (Erbium-doped, pump-keyed)

    All 13 cores are Erbium-doped (fully-active MCF architecture). Each core
    receives an independent pump power P_i at 976 nm. The full 13-dimensional
    key K = (P_0 ... P_12) shapes the complete 13x13 transfer matrix H(K),
    making all inter-core coupling paths key-dependent.
    """
    pitch_um: float = 16.4   # nearest-neighbour pitch [μm]

    # ── core positions ──────────────────────────────────────────────────────
    @property
    def positions(self) -> np.ndarray:
        """
        Returns array of shape (13, 2) with (x, y) core positions in μm.
        Inner ring at radius = pitch, outer ring at radius = 2 × pitch.
        """
        pos = np.zeros((13, 2))
        # inner ring: 6 cores at radius = pitch, spaced 60°
        for k in range(6):
            angle = k * np.pi / 3.0
            pos[k + 1, 0] = self.pitch_um * np.cos(angle)
            pos[k + 1, 1] = self.pitch_um * np.sin(angle)
        # outer ring: 6 cores at radius = 2 × pitch, spaced 60° (offset 30°)
        for k in range(6):
            angle = k * np.pi / 3.0 + np.pi / 6.0
            pos[k + 7, 0] = 2 * self.pitch_um * np.cos(angle)
            pos[k + 7, 1] = 2 * self.pitch_um * np.sin(angle)
        return pos

    @property
    def erbium_core_indices(self) -> List[int]:
        """All 13 cores are Erbium-doped (fully-active MCF architecture).
        Each core receives an independent pump power as part of the key K ∈ R^13.
        This makes the full 13×13 transfer matrix key-dependent, which is required
        for strong physical-layer encryption (BER_Eve > 0.5 at key mismatch).
        """
        return list(range(13))

    @property
    def data_core_indices(self) -> List[int]:
        """All 13 cores carry data (they are simultaneously data and EDFA cores)."""
        return list(range(13))

    @property
    def N(self) -> int:
        return 13

    def distances(self) -> np.ndarray:
        """Pairwise distance matrix [μm], shape (N, N)."""
        pos = self.positions
        N = self.N
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = np.linalg.norm(pos[i] - pos[j])
        return D

    def adjacency_matrix(self, max_coupling_ratio: float = 1.05) -> np.ndarray:
        """
        Boolean adjacency matrix: True for pairs within
        max_coupling_ratio × pitch of each other (nearest neighbours only).
        """
        D = self.distances()
        return (D > 1e-3) & (D <= max_coupling_ratio * self.pitch_um)


@dataclass
class HexMCFGeometry:
    """
    Standard hexagonal MCF geometry with n_rings concentric shells.

    Core count: N = 1 + 3*n_rings*(n_rings+1)
      n_rings=1 →  7 cores   (1 center + 6 in ring 1)
      n_rings=2 → 19 cores   (7 + 12 in ring 2)
      n_rings=3 → 37 cores   (19 + 18 in ring 3)

    All cores are Erbium-doped and data-carrying (fully-active MCF).
    Positions generated via axial hex-lattice coordinates (q, r):
        x = pitch*(q + 0.5*r),  y = pitch*(sqrt(3)/2 * r)
    Cores within hex-distance <= n_rings are included.
    """
    n_rings:  int   = 2
    pitch_um: float = 16.4

    def __post_init__(self):
        self._pos = self._make_positions()

    def _make_positions(self) -> np.ndarray:
        positions = []
        n = self.n_rings
        for q in range(-n, n + 1):
            for r in range(-n, n + 1):
                if (abs(q) + abs(r) + abs(q + r)) // 2 <= n:
                    x = self.pitch_um * (q + 0.5 * r)
                    y = self.pitch_um * (np.sqrt(3) / 2.0 * r)
                    positions.append([x, y])
        positions.sort(key=lambda p: (round(np.sqrt(p[0]**2 + p[1]**2), 4),
                                      round(np.arctan2(p[1], p[0]), 4)))
        return np.array(positions)

    @property
    def positions(self) -> np.ndarray:
        return self._pos

    @property
    def N(self) -> int:
        return len(self._pos)

    @property
    def erbium_core_indices(self) -> List[int]:
        return list(range(self.N))

    @property
    def data_core_indices(self) -> List[int]:
        return list(range(self.N))

    def distances(self) -> np.ndarray:
        pos = self.positions
        N = self.N
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = np.linalg.norm(pos[i] - pos[j])
        return D

    def adjacency_matrix(self, max_coupling_ratio: float = 1.05) -> np.ndarray:
        D = self.distances()
        return (D > 1e-3) & (D <= max_coupling_ratio * self.pitch_um)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  EDFA GAIN MODEL (two-level, steady-state)
# ──────────────────────────────────────────────────────────────────────────────

def edfa_gain_coeff(pump_power_norm: float, params: FiberParams) -> complex:
    """
    Returns the complex perturbation Δβ to the propagation constant
    of an Erbium-doped core under normalised pump power p ∈ [0, 1].

    Proper two-level steady-state EDFA model (Saleh & Teich 1991, Giles &
    Desurvire 1991, Peroni & Tamburrini Opt. Lett. 1990):

        g_net(p) = g_max · (p - p_sat) / (p + p_sat)

    Physical interpretation:
        p = 0        →  g_net = -g_max  (full absorption, no inversion)
        p = p_sat    →  g_net = 0       (transparency, 50% inversion)
        p → ∞        →  g_net → +g_max  (full gain, complete inversion)

    This gives the maximum key-induced swing of 2·g_max per core, which
    is essential for creating a sufficiently key-sensitive transfer matrix.

    The pump modifies β via:
        Δβ(p) = -α_H · g_net(p)/2  +  j · g_net(p)/2
                  ↑ phase shift (Kramers-Kronig)   ↑ amplitude gain/loss

    Henry factor α_H ≈ 0.3 (semiconductor/Er-doped waveguide, Henry 1982).

    Returns:
        delta_beta [complex, rad/mm]  positive imaginary part = gain
    """
    p = float(np.clip(pump_power_norm, 0.0, 1.0))
    p_sat = params.p_sat_norm

    # Two-level net gain: negative (absorption) at low pump, positive at high pump
    g_net = params.g_max_per_mm * (p - p_sat) / (p + p_sat + 1e-12)

    # Complex propagation constant perturbation
    delta_beta = complex(-params.alpha_H * g_net / 2.0,   # phase shift (real)
                          g_net / 2.0)                     # gain/loss (imag)
    return delta_beta


# ──────────────────────────────────────────────────────────────────────────────
# 4.  COUPLED-MODE EQUATION INTEGRATOR
# ──────────────────────────────────────────────────────────────────────────────

class CMESimulator:
    """
    Integrates the coupled-mode equations (CME) for signal propagation at
    1550 nm through the 13-core MCF under a given pump key K.

    CME (in the slowly-varying-envelope approximation, Yariv 1973):
        da_i/dz = -j [β_i + Δβ_i(K)] a_i  -  j Σ_{j∈adj(i)} κ_{ij} a_j

    where z is the propagation distance [mm], a_i(z) ∈ ℂ is the field
    amplitude in core i, β_i is the bare propagation constant [rad/mm],
    Δβ_i(K) is the pump-induced perturbation, and κ_{ij} is the coupling
    coefficient between cores i and j.

    The coupling matrix Κ is computed from the inter-core distances using
    the approximate Bessel-function formula (Snyder & Love 1983).
    """

    def __init__(self, params: FiberParams, geom: MCFGeometry,
                 rng: Optional[np.random.Generator] = None,
                 delta_n_noise_frac: float = 0.0):
        """
        Parameters
        ----------
        params               : FiberParams instance
        geom                 : MCFGeometry instance
        rng                  : numpy Generator for reproducible noise
        delta_n_noise_frac   : fractional variation in Δn per core
                               (manufacturing tolerance study)
        """
        self.params = params
        self.geom   = geom
        self.rng    = rng if rng is not None else np.random.default_rng(0)
        N = geom.N

        # Per-core Δn perturbation (manufacturing tolerance)
        if delta_n_noise_frac > 0:
            self._dn_perturb = self.rng.normal(
                scale=delta_n_noise_frac * params.delta_n, size=N)
        else:
            self._dn_perturb = np.zeros(N)

        # ── bare propagation constants β_i [rad/mm] ─────────────────────
        self.beta0 = np.zeros(N)
        for i in range(N):
            p_mod = FiberParams(
                n_core=params.n_core,
                delta_n=params.delta_n + self._dn_perturb[i],
                core_diam_um=params.core_diam_um,
                pitch_um=params.pitch_um,
                lam_sig_um=params.lam_sig_um,
                lam_pump_um=params.lam_pump_um,
                length_mm=params.length_mm,
                g_max_per_mm=params.g_max_per_mm,
                p_sat_norm=params.p_sat_norm,
                alpha_H=params.alpha_H,
                sigma_noise=params.sigma_noise,
            )
            self.beta0[i] = p_mod.propagation_constant() * 1000.0  # μm→mm

        # ── coupling matrix Κ [rad/mm] ──────────────────────────────────
        D    = geom.distances()                # μm
        adj  = geom.adjacency_matrix()         # nearest neighbours only
        self.kappa = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if adj[i, j]:
                    self.kappa[i, j] = params.coupling_coeff(pitch_um=D[i, j])

    # ── CME right-hand side ────────────────────────────────────────────────
    def _rhs(self, z: float, a_flat: np.ndarray,
             delta_beta: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the CME system in the slowly-varying envelope (SVE)
        frame.  The common carrier phase exp(-j·β_mean·z) has been factored
        out so the ODE only sees δβ_i = β_i - β_mean + Δβ_i(pump).

        a_flat     : real array length 2N  [Re(a), Im(a)] in SVE frame
        delta_beta : complex array (N,)   = (β_i - β_mean) + Δβ_i(K)
        """
        N = self.geom.N
        a = a_flat[:N] + 1j * a_flat[N:]   # reconstruct complex SVE amplitude

        da = np.zeros(N, dtype=complex)
        for i in range(N):
            # Residual phase + EDFA gain/loss (both small compared to β_mean)
            da[i] = -1j * delta_beta[i] * a[i]
            # Nearest-neighbour coupling
            for j in range(N):
                if self.kappa[i, j] != 0.0:
                    da[i] -= 1j * self.kappa[i, j] * a[j]

        return np.concatenate([da.real, da.imag])

    # ── effective propagation constants under pump key K ──────────────────
    def _beta_eff(self, pump_key: np.ndarray) -> np.ndarray:
        """
        Compute δβ[i] = (β0[i] - β_mean) + Δβ_i(pump_key[i])  [rad/mm]

        Working in the SVE frame (carrier removed) keeps all entries small
        (~0.01–0.1 rad/mm) so solve_ivp can use large step sizes.
        The global phase exp(-j·β_mean·L) cancels in |H|² and BER.
        """
        N = self.geom.N
        beta_mean = float(np.mean(self.beta0))
        delta_beta = (self.beta0 - beta_mean).astype(complex)
        for idx, core_i in enumerate(self.geom.erbium_core_indices):
            delta_beta[core_i] += edfa_gain_coeff(pump_key[idx], self.params)
        return delta_beta

    # ── single-column propagation ─────────────────────────────────────────
    def propagate(self, a0: np.ndarray,
                  pump_key: np.ndarray,
                  length_mm: Optional[float] = None) -> np.ndarray:
        """
        Propagate complex field vector a0 (shape N) from z=0 to z=length_mm.

        Returns a(L) : complex array of shape (N,).
        """
        L = length_mm if length_mm is not None else self.params.length_mm
        beta_eff = self._beta_eff(pump_key)

        a0_flat = np.concatenate([a0.real, a0.imag]).astype(float)

        sol = solve_ivp(
            fun=lambda z, y: self._rhs(z, y, beta_eff),
            t_span=(0.0, L),
            y0=a0_flat,
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
            dense_output=False,
        )
        a_out = sol.y[:self.geom.N, -1] + 1j * sol.y[self.geom.N:, -1]
        return a_out

    # ── transfer matrix H(K) ──────────────────────────────────────────────
    def transfer_matrix(self, pump_key: np.ndarray,
                        length_mm: Optional[float] = None) -> np.ndarray:
        """
        Compute the N×N complex transfer matrix H(K) by propagating each
        standard basis vector e_i through the fiber.

        H[j, i] = a_j(L)  when  a(0) = e_i

        This maps input field vector x to output: y = H @ x
        """
        N = self.geom.N
        H = np.zeros((N, N), dtype=complex)
        e = np.eye(N, dtype=complex)
        for i in range(N):
            H[:, i] = self.propagate(e[:, i], pump_key, length_mm)
        return H


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MCF ENCRYPTION FRAMEWORK
# ──────────────────────────────────────────────────────────────────────────────

class MCFEncryption:
    """
    Physical-layer encryption using pump-keyed MCF transfer matrix.

    Encryption:
        Alice and Bob share a secret pump key K in [0,1]^13 (one pump per core).
        Transmission: y = H(K) x + n  (complex field vectors)
        Bob decrypts: x-hat = MMSE(H(K)) y  (knows K, recomputes H; MMSE equalizer)
        Eve:          lacks K → uses wrong or estimated H → high BER

    Data encoding:
        Symbols are QPSK on all 13 cores simultaneously (all cores carry data
        and are EDFA-doped; self.geom.data_core_indices returns all 13).
    """

    def __init__(self, params: FiberParams, geom: MCFGeometry,
                 seed: int = 42):
        self.params = params
        self.geom   = geom
        self.rng    = np.random.default_rng(seed)
        self.sim    = CMESimulator(params, geom, rng=self.rng)

    # ── symbol helpers ────────────────────────────────────────────────────
    def _qpsk_map(self, bits: np.ndarray) -> np.ndarray:
        """Map pairs of bits to QPSK symbols: {00→+1+j, 01→-1+j, ...}"""
        assert bits.ndim == 2 and bits.shape[1] == 2
        re = 2.0 * bits[:, 0].astype(float) - 1.0
        im = 2.0 * bits[:, 1].astype(float) - 1.0
        return (re + 1j * im) / np.sqrt(2.0)

    def _qpsk_demod(self, syms: np.ndarray) -> np.ndarray:
        bits = np.stack([
            (syms.real >= 0).astype(int),
            (syms.imag >= 0).astype(int)
        ], axis=1)
        return bits

    # ── core session ─────────────────────────────────────────────────────
    def run_session(self,
                    pump_key: np.ndarray,
                    T: int = 1000,
                    eve_key: Optional[np.ndarray] = None,
                    eve_partial_obs: bool = True,
                    length_mm: Optional[float] = None,
                    fab_tol_frac: float = 0.0
                    ) -> Dict:
        """
        Simulate one transmission session.

        Parameters
        ----------
        pump_key        : true key K in [0,1]^13 (13 Erbium core pump powers)
        T               : number of symbol slots
        eve_key         : Eve's assumed key (None = random wrong key)
        eve_partial_obs : Eve observes only outer ring (partial interception)
        length_mm       : override fiber length
        fab_tol_frac    : fractional Δn manufacturing variation

        Returns
        -------
        dict with BER_Bob, BER_Eve, secrecy_capacity, cond_H, ...
        """
        L = length_mm if length_mm is not None else self.params.length_mm
        N = self.geom.N
        data_idx = self.geom.data_core_indices   # 13 cores (all cores carry data)
        Nd = len(data_idx)

        # ── recompute simulator if manufacturing tolerance active ─────────
        if fab_tol_frac > 0:
            sim = CMESimulator(self.params, self.geom,
                               rng=np.random.default_rng(int(self.rng.integers(0, 2**31))),
                               delta_n_noise_frac=fab_tol_frac)
        else:
            sim = self.sim

        # ── Bob transfer matrix ───────────────────────────────────────────
        H_bob = sim.transfer_matrix(pump_key, L)

        # Restrict to data cores for encryption (Nd x Nd sub-matrix)
        Hd = H_bob[np.ix_(data_idx, data_idx)]
        cond_H = float(np.linalg.cond(Hd))

        # MMSE equalizer: W = (H^dagger H + sigma^2 I)^{-1} H^dagger
        # Superior to ZF (pinv) — never amplifies noise beyond 1/sigma.
        # Collapses to ZF at high SNR; strictly better at low SNR.
        all_var = _ase_noise_variances(pump_key, self.params, self.geom)
        sigma2_mean = float(np.mean(all_var[data_idx]))   # scalar for MMSE reg.
        I_Nd = np.eye(Nd)
        Hd_inv_mmse = np.linalg.solve(
            Hd.conj().T @ Hd + sigma2_mean * I_Nd,
            Hd.conj().T
        )
        # Keep ZF inverse for secrecy capacity computation (analytical formula)
        Hd_inv_zf = np.linalg.pinv(Hd)

        # ── Eve's assumed transfer matrix ─────────────────────────────────
        if eve_key is None:
            eve_key = self.rng.uniform(0, 1, size=pump_key.shape)
        H_eve = sim.transfer_matrix(eve_key, L)
        Hd_eve = H_eve[np.ix_(data_idx, data_idx)]
        Hd_eve_inv = np.linalg.pinv(Hd_eve)

        # Eve partial observation: only outer ring cores visible
        if eve_partial_obs:
            outer = [i for i in range(Nd) if data_idx[i] != 0]
            Hd_eve_partial = Hd_eve[np.ix_(outer, range(Nd))]
            Hd_eve_partial_inv = np.linalg.pinv(Hd_eve_partial)

        # ── generate and transmit data ────────────────────────────────────
        bits = self.rng.integers(0, 2, size=(T, Nd, 2))
        syms = np.array([self._qpsk_map(bits[t]) for t in range(T)])
        # syms shape: (T, Nd)

        # Per-core ASE + thermal noise (Giles & Desurvire 1991)
        data_var = all_var[data_idx]                # per-core variance, shape (Nd,)
        sigma_c  = np.sqrt(data_var / 2.0)          # per-quadrature std
        noise = (self.rng.normal(size=(T, Nd)) * sigma_c[None, :] +
                 1j * self.rng.normal(size=(T, Nd)) * sigma_c[None, :])

        # Transmitted (ciphertext)
        y = syms @ Hd.T + noise   # shape (T, Nd)

        # ── Bob decryption (MMSE) ─────────────────────────────────────────
        x_hat_bob = y @ Hd_inv_mmse.T
        bits_hat_bob = np.array([self._qpsk_demod(x_hat_bob[t]) for t in range(T)])
        ber_bob = float(np.mean(bits != bits_hat_bob))

        # ── Eve decryption (wrong key, ZF — Eve uses pinv, no MMSE reg.) ──
        x_hat_eve = y @ Hd_eve_inv.T
        bits_hat_eve = np.array([self._qpsk_demod(x_hat_eve[t]) for t in range(T)])
        ber_eve = float(np.mean(bits != bits_hat_eve))

        # ── Eve partial obs ───────────────────────────────────────────────
        if eve_partial_obs:
            y_partial = y[:, outer]
            x_hat_eve_p = y_partial @ Hd_eve_partial_inv.T
            bits_hat_eve_p = np.array([self._qpsk_demod(x_hat_eve_p[t]) for t in range(T)])
            ber_eve_partial = float(np.mean(bits != bits_hat_eve_p))
        else:
            ber_eve_partial = float("nan")

        # ── secrecy capacity (Gaussian wiretap, Leung-Yan-Cheong & Hellman 1978) ─
        # Uses effective mean noise variance to account for per-core ASE noise.
        sigma_eff = float(np.sqrt(np.mean(data_var)))
        Cs_bits = _secrecy_capacity(Hd, Hd_eve, sigma_eff)

        return {
            "BER_Bob":         ber_bob,
            "BER_Eve":         ber_eve,
            "BER_Eve_partial": ber_eve_partial,
            "secrecy_cap":     Cs_bits,
            "cond_H":          cond_H,
            "pump_key":        pump_key.tolist(),
        }

    # ── two-stage nonlinear session ───────────────────────────────────────
    def run_session_nonlinear(self,
                               pump_key1: np.ndarray,
                               pump_key2: np.ndarray,
                               p_sat_sig: float = 1.0,
                               T: int = 1000,
                               eve_key1: Optional[np.ndarray] = None,
                               eve_key2: Optional[np.ndarray] = None,
                               eve_p_sat_sig: Optional[float] = None,
                               ) -> Dict:
        """
        Two-stage nonlinear MCF channel with saturable absorber activation.

        Architecture
        ------------
        Alice:
          z = H₁(K₁) · x + n₁          (MCF stage 1)
          a = σ(z, P_sat_sig)           (saturable absorber: amplitude compression)
          y = H₂(K₂) · a + n₂          (MCF stage 2)

        Bob (correct keys K₁, K₂, P_sat_sig):
          ẑ = MMSE(y, H₂)              (invert stage 2)
          â = σ⁻¹(ẑ, P_sat_sig)       (invert activation — exact, known threshold)
          x̂ = MMSE(â, H₁)             (invert stage 1)
          → BER_Bob = 0 in noiseless limit

        Eve (wrong keys):
          Applies same 3-stage decoding with wrong (K₁', K₂', P_sat_sig').
          Errors compound multiplicatively across three stages.

        Eve (linear regression):
          Fits linear model y ≈ H_est · x.
          Best-linear-approximation to a nonlinear function — systematic
          error remains even with infinite observations.

        Parameters
        ----------
        pump_key1   : K₁ ∈ [0,1]^N — pump powers for MCF stage 1
        pump_key2   : K₂ ∈ [0,1]^N — pump powers for MCF stage 2
        p_sat_sig   : saturable absorber saturation amplitude
                      (signal activation threshold; comparable to |z| after stage 1)
        T           : number of symbol slots
        eve_key1/2  : Eve's guessed keys (None = random)
        eve_p_sat_sig: Eve's guessed activation threshold (None = true value — best case)
        """
        L  = self.params.length_mm
        N  = self.geom.N
        data_idx = self.geom.data_core_indices
        Nd = len(data_idx)
        I_Nd = np.eye(Nd)

        # ── Bob's transfer matrices ───────────────────────────────────────
        H1_bob = self.sim.transfer_matrix(pump_key1, L)
        H2_bob = self.sim.transfer_matrix(pump_key2, L)
        Hd1    = H1_bob[np.ix_(data_idx, data_idx)]
        Hd2    = H2_bob[np.ix_(data_idx, data_idx)]

        # MMSE regularization from ASE noise in each stage
        all_var1   = _ase_noise_variances(pump_key1, self.params, self.geom)
        all_var2   = _ase_noise_variances(pump_key2, self.params, self.geom)
        sigma2_s1  = float(np.mean(all_var1[data_idx]))
        sigma2_s2  = float(np.mean(all_var2[data_idx]))
        data_var1  = all_var1[data_idx]
        data_var2  = all_var2[data_idx]
        sigma_c1   = np.sqrt(data_var1 / 2.0)
        sigma_c2   = np.sqrt(data_var2 / 2.0)

        Hd1_inv_mmse = np.linalg.solve(
            Hd1.conj().T @ Hd1 + sigma2_s1 * I_Nd, Hd1.conj().T)
        Hd2_inv_mmse = np.linalg.solve(
            Hd2.conj().T @ Hd2 + sigma2_s2 * I_Nd, Hd2.conj().T)

        # ── Eve's transfer matrices ───────────────────────────────────────
        if eve_key1 is None:
            eve_key1 = self.rng.uniform(0, 1, size=pump_key1.shape)
        if eve_key2 is None:
            eve_key2 = self.rng.uniform(0, 1, size=pump_key2.shape)
        if eve_p_sat_sig is None:
            eve_p_sat_sig = p_sat_sig   # Eve knows the architecture (Kerckhoffs)

        H1_eve  = self.sim.transfer_matrix(eve_key1, L)
        H2_eve  = self.sim.transfer_matrix(eve_key2, L)
        Hd1_eve = H1_eve[np.ix_(data_idx, data_idx)]
        Hd2_eve = H2_eve[np.ix_(data_idx, data_idx)]
        Hd1_eve_inv = np.linalg.pinv(Hd1_eve)
        Hd2_eve_inv = np.linalg.pinv(Hd2_eve)

        # ── generate QPSK symbols ─────────────────────────────────────────
        bits = self.rng.integers(0, 2, size=(T, Nd, 2))
        syms = np.array([self._qpsk_map(bits[t]) for t in range(T)])
        # syms shape: (T, Nd)

        # ── forward pass (Alice) ──────────────────────────────────────────
        # Stage 1: z = H₁·x + n₁
        noise1 = (self.rng.normal(size=(T, Nd)) * sigma_c1[None, :] +
                  1j * self.rng.normal(size=(T, Nd)) * sigma_c1[None, :])
        z = syms @ Hd1.T + noise1              # (T, Nd)

        # Saturable absorber activation: a = σ(z, p_sat_sig)
        a = _saturable_absorber(z, p_sat_sig)  # (T, Nd)

        # Stage 2: y = H₂·a + n₂
        noise2 = (self.rng.normal(size=(T, Nd)) * sigma_c2[None, :] +
                  1j * self.rng.normal(size=(T, Nd)) * sigma_c2[None, :])
        y = a @ Hd2.T + noise2                 # (T, Nd)

        # ── Bob decoding (3-stage inverse) ────────────────────────────────
        z_hat_bob  = y @ Hd2_inv_mmse.T                          # invert stage 2
        a_hat_bob  = _saturable_absorber_inv(z_hat_bob, p_sat_sig)  # invert activation
        x_hat_bob  = a_hat_bob @ Hd1_inv_mmse.T                  # invert stage 1
        bits_hat_bob = np.array([self._qpsk_demod(x_hat_bob[t]) for t in range(T)])
        ber_bob = float(np.mean(bits != bits_hat_bob))

        # ── Eve decoding (wrong keys, ZF at each stage) ───────────────────
        z_hat_eve  = y @ Hd2_eve_inv.T                                # wrong stage-2 inv
        a_hat_eve  = _saturable_absorber_inv(z_hat_eve, eve_p_sat_sig) # wrong threshold
        x_hat_eve  = a_hat_eve @ Hd1_eve_inv.T                        # wrong stage-1 inv
        bits_hat_eve = np.array([self._qpsk_demod(x_hat_eve[t]) for t in range(T)])
        ber_eve_wk = float(np.mean(bits != bits_hat_eve))

        # ── Eve linear regression attack ──────────────────────────────────
        # Fit: y ≈ x @ H_est.T  (best linear approximation to nonlinear channel)
        # Uses all T symbols for estimation AND testing — upper bound on Eve's LR BER
        X_lr  = syms           # (T, Nd) — plaintext
        Y_lr  = y              # (T, Nd) — ciphertext
        n_train = min(max(2 * Nd, T // 2), T - 1)
        X_lr = X_lr[:n_train]
        Y_lr = Y_lr[:n_train]
        bits_te = bits[n_train:]
        H_est_T = np.linalg.lstsq(X_lr, Y_lr, rcond=None)[0]  # (Nd, Nd)
        H_est   = H_est_T.T
        H_est_inv = np.linalg.pinv(H_est)
        x_hat_lr = y[n_train:] @ H_est_inv.T
        bits_hat_lr = np.array([self._qpsk_demod(x_hat_lr[t])
                                for t in range(len(x_hat_lr))])
        ber_eve_lr = float(np.mean(bits_te != bits_hat_lr))

        # ── secrecy capacity (Gaussian UB, first stage only — conservative) ─
        H_eve_rand = self.sim.transfer_matrix(self.rng.uniform(0,1,pump_key1.shape), L)
        Hd_eve_rand = H_eve_rand[np.ix_(data_idx, data_idx)]
        sigma_eff = float(np.sqrt(np.mean(data_var1 + data_var2)))
        Cs_bits = _secrecy_capacity(Hd1, Hd_eve_rand, sigma_eff)

        return {
            "BER_Bob":      ber_bob,
            "BER_Eve_WK":   ber_eve_wk,   # wrong-key 3-stage decode
            "BER_Eve_LR":   ber_eve_lr,   # linear regression (all T pairs)
            "secrecy_cap":  Cs_bits,
            "p_sat_sig":    p_sat_sig,
        }


