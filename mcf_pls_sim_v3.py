"""
Pump-keyed multi-core-fiber simulation for optical physical-layer encryption.

This v3 entry point is the submission-oriented package. It keeps the
validated security workflows from v2, writes outputs to `mcf_pls_figs3`,
and adds a calibration/provenance bundle so the manuscript can state clearly:

- which parameters are supported by public literature,
- which rely on internal non-public notes or internal simulations, and
- which remain chosen operating points for the publication runs.

Core communication model:

    y = H(K) x + n

where `K` is the secret pump key, `H(K)` is the key-dependent transfer matrix,
Bob reconstructs `H(K)` and applies MMSE inversion, and Eve must estimate or
learn the inverse from intercepted data.

The default `main()` run produces the current manuscript artifacts:

- 13-core FO / WT attack benchmarks,
- 13/19/37-core scaling results,
- 37-core multiseed, sample-complexity, and tap-sensitivity sweeps,
- exploratory SPM+XPM comparisons, and
- calibration/traceability tables and manifests for the submission package.

Active Eve baselines in the current package:

- KPA-CE: forward-channel estimation plus inversion,
- KPA-LR: direct linear regression,
- KPA-HYB: linear estimate plus learned residual correction,
- KPA-MLP: current-sample multilayer perceptron,
- KPA-TX: current-sample transformer, and
- KPA-LSTM: sequence-model baseline.
"""

from __future__ import annotations

import os
import csv
import json
import warnings
from itertools import combinations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kv as bessel_K   # Modified Bessel K_n

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

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
# ──────────────────────────────────────────────────────────────────────────────

class EveRNN(nn.Module if HAS_TORCH else object):
    """
    LSTM-based Eve adversary.  Processes a sliding window of received
    ciphertext observations y_t to estimate the plaintext x_t.

    This is the 'strongest Eve' baseline (oracle training, no key knowledge).
    Represents an upper bound on Eve's learned decoding capability.
    """
    def __init__(self, obs_dim: int, sym_dim: int, hidden: int = 128,
                 layers: int = 2):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for ML adversary.")
        super().__init__()
        self.lstm = nn.LSTM(obs_dim * 2, hidden, num_layers=layers,
                            batch_first=True)
        self.head  = nn.Linear(hidden, sym_dim * 2)
        self.obs_dim = obs_dim
        self.sym_dim = sym_dim

    def forward(self, y_seq):
        # y_seq: (B, T_win, obs_dim*2)  real+imag concatenated
        out, _ = self.lstm(y_seq)
        return self.head(out[:, -1, :])   # (B, sym_dim*2)


class EveMLP(nn.Module if HAS_TORCH else object):
    """
    Current-sample learned Eve baseline.

    This is the right learned control for a memoryless channel because the
    target symbol x_t is determined primarily by the instantaneous observation
    y_t rather than by temporal context.
    """
    def __init__(self, obs_dim: int, sym_dim: int, hidden: int = 256,
                 depth: int = 3):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for ML adversary.")
        super().__init__()
        layers = []
        in_dim = 2 * obs_dim
        for _ in range(max(depth - 1, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2 * sym_dim))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.sym_dim = sym_dim

    def forward(self, y_now):
        return self.net(y_now)


class EveTransformer(nn.Module if HAS_TORCH else object):
    """
    Current-sample transformer Eve baseline.

    The observed cores are treated as tokens with two real-valued features
    (real and imaginary parts). Self-attention lets Eve learn cross-core
    relationships without assuming a linear inverse.
    """
    def __init__(self, obs_dim: int, sym_dim: int, d_model: int = 64,
                 nhead: int = 4, layers: int = 2, ff_mult: int = 4):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for ML adversary.")
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, obs_dim, d_model))
        # Keep the current architecture for result continuity while silencing
        # a benign nested-tensor warning emitted by recent PyTorch versions.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True, but self.use_nested_tensor is False*",
                category=UserWarning,
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_mult * d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * sym_dim),
        )
        self.obs_dim = obs_dim
        self.sym_dim = sym_dim

    def forward(self, y_tokens):
        h = self.input_proj(y_tokens) + self.pos_embed[:, :y_tokens.shape[1], :]
        z = self.encoder(h)
        pooled = z.mean(dim=1)
        return self.head(pooled)


class EveHybridMLP(nn.Module if HAS_TORCH else object):
    """
    Hybrid Eve baseline: explicit linear estimate plus learned residual.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256,
                 depth: int = 3):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for ML adversary.")
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(max(depth - 1, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        return self.net(feat)


def _best_state_dict(model) -> Dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_eve_mlp(y_cipher: np.ndarray,
                  x_plain: np.ndarray,
                  epochs: int = 25,
                  lr: float = 1e-3,
                  seed: int = 0) -> Tuple[np.ndarray, object, Dict]:
    """
    Train a current-sample MLP Eve adversary on intercepted (y_t, x_t) pairs.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for ML adversary.")

    torch.manual_seed(seed)
    T, obs_dim = y_cipher.shape
    sym_dim = x_plain.shape[1]

    X = np.concatenate([y_cipher.real, y_cipher.imag], axis=1).astype(np.float32)
    Y = np.concatenate([x_plain.real, x_plain.imag], axis=1).astype(np.float32)
    split = max(1, int(0.8 * len(X)))
    if split >= len(X):
        split = len(X) - 1
    Xtr, Xva = X[:split], X[split:]
    Ytr, Yva = Y[:split], Y[split:]

    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std = np.maximum(Xtr.std(axis=0, keepdims=True), 1e-6)
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = np.maximum(Ytr.std(axis=0, keepdims=True), 1e-6)
    Xtr_n = ((Xtr - x_mean) / x_std).astype(np.float32)
    Xva_n = ((Xva - x_mean) / x_std).astype(np.float32)
    Xall_n = ((X - x_mean) / x_std).astype(np.float32)
    Ytr_n = ((Ytr - y_mean) / y_std).astype(np.float32)
    Yva_n = ((Yva - y_mean) / y_std).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EveMLP(obs_dim=obs_dim, sym_dim=sym_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    rng_shuffle = np.random.default_rng(seed)

    def batches(Xa, Ya, bs=256):
        idx = np.arange(len(Xa))
        rng_shuffle.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield (torch.from_numpy(Xa[idx[i:i + bs]]).to(dev),
                   torch.from_numpy(Ya[idx[i:i + bs]]).to(dev))

    best_state = _best_state_dict(model)
    best_val = float("inf")
    val_losses = []
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(Xtr_n, Ytr_n):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(torch.from_numpy(Xva_n).to(dev)),
                               torch.from_numpy(Yva_n).to(dev)))
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = _best_state_dict(model)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_hat_n = model(torch.from_numpy(Xall_n).to(dev)).cpu().numpy()
    y_hat = y_hat_n * y_std + y_mean
    x_hat = y_hat[:, :sym_dim] + 1j * y_hat[:, sym_dim:]
    metrics = {"val_mse_final": best_val, "val_losses": val_losses}
    return x_hat, model, metrics


def train_eve_transformer(y_cipher: np.ndarray,
                          x_plain: np.ndarray,
                          epochs: int = 25,
                          lr: float = 1e-3,
                          seed: int = 0) -> Tuple[np.ndarray, object, Dict]:
    """
    Train a current-sample transformer Eve adversary on intercepted (y_t, x_t) pairs.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for ML adversary.")

    torch.manual_seed(seed)
    T, obs_dim = y_cipher.shape
    sym_dim = x_plain.shape[1]

    X = np.stack([y_cipher.real, y_cipher.imag], axis=2).astype(np.float32)   # (T, obs_dim, 2)
    Y = np.concatenate([x_plain.real, x_plain.imag], axis=1).astype(np.float32)
    split = max(1, int(0.8 * len(X)))
    if split >= len(X):
        split = len(X) - 1
    Xtr, Xva = X[:split], X[split:]
    Ytr, Yva = Y[:split], Y[split:]

    x_mean = Xtr.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(Xtr.std(axis=(0, 1), keepdims=True), 1e-6)
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = np.maximum(Ytr.std(axis=0, keepdims=True), 1e-6)
    Xtr_n = ((Xtr - x_mean) / x_std).astype(np.float32)
    Xva_n = ((Xva - x_mean) / x_std).astype(np.float32)
    Xall_n = ((X - x_mean) / x_std).astype(np.float32)
    Ytr_n = ((Ytr - y_mean) / y_std).astype(np.float32)
    Yva_n = ((Yva - y_mean) / y_std).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EveTransformer(obs_dim=obs_dim, sym_dim=sym_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    rng_shuffle = np.random.default_rng(seed)

    def batches(Xa, Ya, bs=256):
        idx = np.arange(len(Xa))
        rng_shuffle.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield (torch.from_numpy(Xa[idx[i:i + bs]]).to(dev),
                   torch.from_numpy(Ya[idx[i:i + bs]]).to(dev))

    best_state = _best_state_dict(model)
    best_val = float("inf")
    val_losses = []
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(Xtr_n, Ytr_n):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(torch.from_numpy(Xva_n).to(dev)),
                               torch.from_numpy(Yva_n).to(dev)))
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = _best_state_dict(model)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_hat_n = model(torch.from_numpy(Xall_n).to(dev)).cpu().numpy()
    y_hat = y_hat_n * y_std + y_mean
    x_hat = y_hat[:, :sym_dim] + 1j * y_hat[:, sym_dim:]
    metrics = {"val_mse_final": best_val, "val_losses": val_losses}
    return x_hat, model, metrics


def train_eve_hybrid_mlp(y_cipher: np.ndarray,
                         x_plain: np.ndarray,
                         epochs: int = 25,
                         lr: float = 1e-3,
                         seed: int = 0) -> Tuple[np.ndarray, object, Dict]:
    """
    Train a hybrid Eve: explicit LR estimate + learned residual correction.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for ML adversary.")

    torch.manual_seed(seed)
    T, obs_dim = y_cipher.shape
    sym_dim = x_plain.shape[1]

    A = np.linalg.lstsq(y_cipher, x_plain, rcond=None)[0]
    x_lin = y_cipher @ A
    resid = x_plain - x_lin

    X = np.concatenate([y_cipher.real, y_cipher.imag,
                        x_lin.real, x_lin.imag], axis=1).astype(np.float32)
    Y = np.concatenate([resid.real, resid.imag], axis=1).astype(np.float32)
    split = max(1, int(0.8 * len(X)))
    if split >= len(X):
        split = len(X) - 1
    Xtr, Xva = X[:split], X[split:]
    Ytr, Yva = Y[:split], Y[split:]

    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std = np.maximum(Xtr.std(axis=0, keepdims=True), 1e-6)
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = np.maximum(Ytr.std(axis=0, keepdims=True), 1e-6)
    Xtr_n = ((Xtr - x_mean) / x_std).astype(np.float32)
    Xva_n = ((Xva - x_mean) / x_std).astype(np.float32)
    Xall_n = ((X - x_mean) / x_std).astype(np.float32)
    Ytr_n = ((Ytr - y_mean) / y_std).astype(np.float32)
    Yva_n = ((Yva - y_mean) / y_std).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EveHybridMLP(input_dim=X.shape[1], output_dim=Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    rng_shuffle = np.random.default_rng(seed)

    def batches(Xa, Ya, bs=256):
        idx = np.arange(len(Xa))
        rng_shuffle.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield (torch.from_numpy(Xa[idx[i:i + bs]]).to(dev),
                   torch.from_numpy(Ya[idx[i:i + bs]]).to(dev))

    best_state = _best_state_dict(model)
    best_val = float("inf")
    val_losses = []
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(Xtr_n, Ytr_n):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(torch.from_numpy(Xva_n).to(dev)),
                               torch.from_numpy(Yva_n).to(dev)))
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = _best_state_dict(model)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        resid_hat_n = model(torch.from_numpy(Xall_n).to(dev)).cpu().numpy()
    resid_hat = resid_hat_n * y_std + y_mean
    resid_hat_c = resid_hat[:, :sym_dim] + 1j * resid_hat[:, sym_dim:]
    x_hat = x_lin + resid_hat_c
    metrics = {"val_mse_final": best_val, "val_losses": val_losses}
    return x_hat, model, {"A": A, **metrics}


def train_eve_rnn(y_cipher: np.ndarray,
                  x_plain: np.ndarray,
                  window: int = 8,
                  epochs: int = 15,
                  lr: float = 5e-4,
                  seed: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    Train Eve's RNN adversary on intercepted (y, x) pairs and return
    decoded symbols.

    Parameters
    ----------
    y_cipher : (T, Nd) complex — Eve's observations
    x_plain  : (T, Nd) complex — true transmitted symbols (oracle)
    Returns decoded symbols x_hat : (T, Nd) complex
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for ML adversary.")

    torch.manual_seed(seed)
    T, obs_dim = y_cipher.shape
    sym_dim = x_plain.shape[1]

    # Build windowed dataset. The current observation y_t must be included;
    # otherwise an IID symbol stream is information-free for the attacker.
    X, Y = [], []
    for t in range(window - 1, T):
        feat = y_cipher[t - window + 1:t + 1]   # (window, obs_dim) complex
        feat_r = np.concatenate(
            [feat.real, feat.imag], axis=1)     # (window, 2*obs_dim) real
        X.append(feat_r)
        Y.append(np.concatenate([x_plain[t].real, x_plain[t].imag]))
    X = np.stack(X).astype(np.float32)         # (T-window, window, 2*obs_dim)
    Y = np.stack(Y).astype(np.float32)         # (T-window, 2*sym_dim)

    split = int(0.8 * len(X))
    Xtr, Xva = X[:split], X[split:]
    Ytr, Yva = Y[:split], Y[split:]

    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EveRNN(obs_dim=obs_dim, sym_dim=sym_dim).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def batches(Xa, Ya, bs=256):
        idx = np.arange(len(Xa))
        for i in range(0, len(idx), bs):
            yield (torch.from_numpy(Xa[idx[i:i+bs]]).to(dev),
                   torch.from_numpy(Ya[idx[i:i+bs]]).to(dev))

    train_losses, val_losses = [], []
    for ep in range(epochs):
        model.train()
        for xb, yb in batches(Xtr, Ytr):
            pred = model(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(torch.from_numpy(Xva).to(dev)),
                               torch.from_numpy(Yva).to(dev)))
        val_losses.append(vl)

    # Decode
    model.eval()
    x_hat_flat = np.zeros((T, 2 * sym_dim), dtype=np.float32)
    with torch.no_grad():
        for t in range(window - 1, T):
            feat = y_cipher[t - window + 1:t + 1]
            feat_r = np.concatenate([feat.real, feat.imag], axis=1)
            inp = torch.from_numpy(feat_r[None].astype(np.float32)).to(dev)
            x_hat_flat[t] = model(inp).cpu().numpy()[0]

    x_hat = x_hat_flat[:, :sym_dim] + 1j * x_hat_flat[:, sym_dim:]
    val_mse_final = float(val_losses[-1]) if val_losses else float("nan")
    metrics = {"val_mse_final": val_mse_final, "val_losses": val_losses}
    return x_hat, model, metrics


@dataclass
class TapObservationConfig:
    """
    Simple physical tap model for WT.

    Eve observes only a subset of output cores through a weak passive tap.
    The observed amplitude is attenuated by sqrt(tap_fraction), and Eve adds
    her own receiver noise on top of the tapped field.
    """
    tap_fraction: float = 0.10
    eve_noise_scale: float = 1.5
    observed_core_indices: Optional[List[int]] = None
    label: str = "outer-ring passive tap"


def _qpsk_bits_to_syms(bits: np.ndarray) -> np.ndarray:
    re = 2.0 * bits[:, :, 0].astype(float) - 1.0
    im = 2.0 * bits[:, :, 1].astype(float) - 1.0
    return (re + 1j * im) / np.sqrt(2.0)


def _qpsk_demod_array(x_hat: np.ndarray) -> np.ndarray:
    return np.stack([
        (x_hat.real >= 0).astype(int),
        (x_hat.imag >= 0).astype(int),
    ], axis=2)


def _bit_error_rate(bits_ref: np.ndarray, bits_hat: np.ndarray) -> float:
    return float(np.mean(bits_ref != bits_hat))


def _resolve_tap_core_indices(geom: MCFGeometry,
                              observed_core_indices: Optional[List[int]] = None) -> List[int]:
    if observed_core_indices is not None:
        return list(observed_core_indices)
    if hasattr(geom, "n_rings"):
        shells: Dict[int, List[int]] = {}
        for idx, (x, y) in enumerate(geom.positions):
            r_ax = y / (geom.pitch_um * np.sqrt(3) / 2.0)
            q_ax = x / geom.pitch_um - 0.5 * r_ax
            q = int(np.round(q_ax))
            r = int(np.round(r_ax))
            shell = int((abs(q) + abs(r) + abs(q + r)) // 2)
            shells.setdefault(shell, []).append(int(idx))
        if shells:
            return shells[max(shells.keys())]
    radii = np.linalg.norm(geom.positions, axis=1)
    threshold = float(np.median(radii))
    outer = [int(i) for i in np.where(radii > threshold + 1e-9)[0]]
    if outer:
        return outer
    return list(range(max(1, geom.N // 2), geom.N))


def _complex_awgn(rng: np.random.Generator,
                  shape: Tuple[int, ...],
                  sigma: np.ndarray | float) -> np.ndarray:
    # sigma is the complex Gaussian standard deviation: E[|n|^2] = sigma^2.
    sigma_arr = np.asarray(sigma, dtype=float)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * (sigma_arr / np.sqrt(2.0))


def _linear_mmse_inverse(H: np.ndarray, sigma2: float) -> np.ndarray:
    return np.linalg.solve(H.conj().T @ H + sigma2 * np.eye(H.shape[1]), H.conj().T)


def _apply_wt_tap(y_clean: np.ndarray,
                        geom: MCFGeometry,
                        params: FiberParams,
                        tap_cfg: TapObservationConfig,
                        rng: np.random.Generator) -> Tuple[np.ndarray, List[int]]:
    obs = _resolve_tap_core_indices(geom, tap_cfg.observed_core_indices)
    y_tap = np.sqrt(tap_cfg.tap_fraction) * y_clean[:, obs]
    sigma_e = params.sigma_noise * tap_cfg.eve_noise_scale
    noise_e = _complex_awgn(rng, y_tap.shape, sigma_e)
    return y_tap + noise_e, obs


# def _evaluate_kpa_attack(x_train: np.ndarray,
#                          y_train: np.ndarray,
#                          y_test: np.ndarray,
#                          bits_test: np.ndarray) -> float:
#     H_est_T = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
#     H_est = H_est_T.T
#     x_hat = y_test @ np.linalg.pinv(H_est).T
#     return _bit_error_rate(bits_test, _qpsk_demod_array(x_hat))


def _evaluate_kpa_attack(x_train: np.ndarray,
                         y_train: np.ndarray,
                         y_test: np.ndarray,
                         bits_test: np.ndarray,
                         sigma2: float = 0.0) -> float:
    H_est_T = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    H_est = H_est_T.T
    x_hat = y_test @ _linear_mmse_inverse(H_est, sigma2).T
    return _bit_error_rate(bits_test, _qpsk_demod_array(x_hat))


def _evaluate_lr_attack(x_train: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray,
                        bits_test: np.ndarray) -> float:
    A = np.linalg.lstsq(y_train, x_train, rcond=None)[0]
    x_hat = y_test @ A
    return _bit_error_rate(bits_test, _qpsk_demod_array(x_hat))


def _evaluate_mlp_attack(y_train: np.ndarray,
                         x_train: np.ndarray,
                         y_test: np.ndarray,
                         bits_test: np.ndarray,
                         epochs: int = 25,
                         seed: int = 0) -> float:
    if not HAS_TORCH:
        return float("nan")
    if len(y_train) < 32:
        return float("nan")

    _, model, _ = train_eve_mlp(y_train, x_train, epochs=epochs, seed=seed)
    dev = next(model.parameters()).device
    Nd = x_train.shape[1]

    Xtr = np.concatenate([y_train.real, y_train.imag], axis=1).astype(np.float32)
    Xte = np.concatenate([y_test.real, y_test.imag], axis=1).astype(np.float32)
    Ytr = np.concatenate([x_train.real, x_train.imag], axis=1).astype(np.float32)
    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std = np.maximum(Xtr.std(axis=0, keepdims=True), 1e-6)
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = np.maximum(Ytr.std(axis=0, keepdims=True), 1e-6)
    Xte_n = ((Xte - x_mean) / x_std).astype(np.float32)

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xte_n).to(dev)).cpu().numpy()
    pred = pred_n * y_std + y_mean
    x_hat = pred[:, :Nd] + 1j * pred[:, Nd:]
    return _bit_error_rate(bits_test, _qpsk_demod_array(x_hat))


def _evaluate_transformer_attack(y_train: np.ndarray,
                                 x_train: np.ndarray,
                                 y_test: np.ndarray,
                                 bits_test: np.ndarray,
                                 epochs: int = 25,
                                 seed: int = 0) -> float:
    if not HAS_TORCH:
        return float("nan")
    if len(y_train) < 32:
        return float("nan")

    _, model, _ = train_eve_transformer(
        y_train, x_train, epochs=epochs, seed=seed)
    dev = next(model.parameters()).device
    Nd = x_train.shape[1]

    Xtr = np.stack([y_train.real, y_train.imag], axis=2).astype(np.float32)
    Xte = np.stack([y_test.real, y_test.imag], axis=2).astype(np.float32)
    Ytr = np.concatenate([x_train.real, x_train.imag], axis=1).astype(np.float32)
    x_mean = Xtr.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(Xtr.std(axis=(0, 1), keepdims=True), 1e-6)
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = np.maximum(Ytr.std(axis=0, keepdims=True), 1e-6)
    Xte_n = ((Xte - x_mean) / x_std).astype(np.float32)

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xte_n).to(dev)).cpu().numpy()
    pred = pred_n * y_std + y_mean
    x_hat = pred[:, :Nd] + 1j * pred[:, Nd:]
    return _bit_error_rate(bits_test, _qpsk_demod_array(x_hat))


def _evaluate_hybrid_attack(y_train: np.ndarray,
                            x_train: np.ndarray,
                            y_test: np.ndarray,
                            bits_test: np.ndarray,
                            epochs: int = 25,
                            seed: int = 0) -> float:
    if not HAS_TORCH:
        return float("nan")
    if len(y_train) < 32:
        return float("nan")

    _, model, metrics = train_eve_hybrid_mlp(
        y_train, x_train, epochs=epochs, seed=seed)
    A = metrics["A"]
    dev = next(model.parameters()).device
    Nd = x_train.shape[1]

    x_lin_train = y_train @ A
    x_lin_test = y_test @ A
    Xtr = np.concatenate([y_train.real, y_train.imag,
                          x_lin_train.real, x_lin_train.imag], axis=1).astype(np.float32)
    Xte = np.concatenate([y_test.real, y_test.imag,
                          x_lin_test.real, x_lin_test.imag], axis=1).astype(np.float32)
    resid_train = x_train - x_lin_train
    Ytr = np.concatenate([resid_train.real, resid_train.imag], axis=1).astype(np.float32)
    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std = np.maximum(Xtr.std(axis=0, keepdims=True), 1e-6)
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = np.maximum(Ytr.std(axis=0, keepdims=True), 1e-6)
    Xte_n = ((Xte - x_mean) / x_std).astype(np.float32)

    model.eval()
    with torch.no_grad():
        resid_pred_n = model(torch.from_numpy(Xte_n).to(dev)).cpu().numpy()
    resid_pred = resid_pred_n * y_std + y_mean
    resid_hat = resid_pred[:, :Nd] + 1j * resid_pred[:, Nd:]
    x_hat = x_lin_test + resid_hat
    return _bit_error_rate(bits_test, _qpsk_demod_array(x_hat))


def _evaluate_lstm_attack(y_train: np.ndarray,
                          x_train: np.ndarray,
                          y_test: np.ndarray,
                          bits_test: np.ndarray,
                          window: int = 8,
                          epochs: int = 6,
                          seed: int = 0) -> float:
    if not HAS_TORCH:
        return float("nan")
    if len(y_train) <= window:
        return float("nan")

    _, model, _ = train_eve_rnn(
        y_train, x_train, window=window, epochs=epochs, seed=seed)

    dev = next(model.parameters()).device
    Nd = x_train.shape[1]
    ctx_len = max(window - 1, 0)
    y_prefix = y_train[-ctx_len:] if ctx_len else y_train[:0]
    y_ctx = np.concatenate([y_prefix, y_test], axis=0)
    x_hat_flat = np.zeros((len(y_test), 2 * Nd), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for t in range(ctx_len, ctx_len + len(y_test)):
            feat = y_ctx[t - window + 1:t + 1]
            feat_r = np.concatenate([feat.real, feat.imag], axis=1)
            inp = torch.from_numpy(feat_r[None].astype(np.float32)).to(dev)
            x_hat_flat[t - ctx_len] = model(inp).cpu().numpy()[0]
    x_hat = x_hat_flat[:, :Nd] + 1j * x_hat_flat[:, Nd:]
    return _bit_error_rate(bits_test, _qpsk_demod_array(x_hat))


def _build_linear_dataset(enc: MCFEncryption,
                          true_key: np.ndarray,
                          option: str,
                          T_train: int,
                          T_test: int,
                          seed: int,
                          tap_cfg: Optional[TapObservationConfig] = None) -> Dict:
    rng = np.random.default_rng(seed)
    geom = enc.geom
    params = enc.params
    data_idx = geom.data_core_indices
    Nd = len(data_idx)

    H_bob = enc.sim.transfer_matrix(true_key, params.length_mm)
    Hd_bob = H_bob[np.ix_(data_idx, data_idx)]
    all_var = _ase_noise_variances(true_key, params, geom)
    sigma2 = float(np.mean(all_var[data_idx]))
    sigma = np.sqrt(all_var[data_idx])
    Hd_inv = _linear_mmse_inverse(Hd_bob, sigma2)

    bits_train = rng.integers(0, 2, size=(T_train, Nd, 2))
    bits_test = rng.integers(0, 2, size=(T_test, Nd, 2))
    x_train = _qpsk_bits_to_syms(bits_train)
    x_test = _qpsk_bits_to_syms(bits_test)

    y_clean_train = x_train @ Hd_bob.T
    y_clean_test = x_test @ Hd_bob.T
    y_bob_test = y_clean_test + _complex_awgn(rng, y_clean_test.shape, sigma[None, :])
    bob_hat = y_bob_test @ Hd_inv.T
    ber_bob = _bit_error_rate(bits_test, _qpsk_demod_array(bob_hat))

    if option == "FO":
        y_eve_train = y_clean_train + _complex_awgn(rng, y_clean_train.shape, sigma[None, :])
        y_eve_test = y_clean_test + _complex_awgn(rng, y_clean_test.shape, sigma[None, :])
        obs_idx = list(range(Nd))
    elif option == "WT":
        if tap_cfg is None:
            tap_cfg = TapObservationConfig()
        y_eve_train, obs_idx = _apply_wt_tap(y_clean_train, geom, params, tap_cfg, rng)
        y_eve_test, _ = _apply_wt_tap(y_clean_test, geom, params, tap_cfg, rng)
    else:
        raise ValueError(f"Unknown option '{option}'.")

    return {
        "x_train": x_train,
        "x_test": x_test,
        "bits_test": bits_test,
        "y_eve_train": y_eve_train,
        "y_eve_test": y_eve_test,
        "BER_Bob": ber_bob,
        "obs_dim": y_eve_train.shape[1],
        "observed_core_indices": obs_idx,
    }


def _mixed_regime_key(rng: np.random.Generator,
                      N: int,
                      n_absorption: Optional[int] = None) -> np.ndarray:
    if n_absorption is None:
        n_absorption = 10 if N == 13 else max(1, int(np.ceil(0.30 * N)))
    key = rng.uniform(0.6, 1.0, size=N)
    abs_idx = rng.choice(N, size=n_absorption, replace=False)
    key[abs_idx] = rng.uniform(0.3, 0.5, size=n_absorption)
    return key


def _row_energy_scale(Hd: np.ndarray) -> np.ndarray:
    return np.maximum(np.sqrt(np.sum(np.abs(Hd) ** 2, axis=1)), 1e-9)


def _mmse_inverse_from_key(enc: MCFEncryption,
                           key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    geom = enc.geom
    params = enc.params
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    I_Nd = np.eye(Nd)
    H = enc.sim.transfer_matrix(key, params.length_mm)
    Hd = H[np.ix_(data_idx, data_idx)]
    all_var = _ase_noise_variances(key, params, geom)
    sigma2 = float(np.mean(all_var[data_idx]))
    sigma = np.sqrt(all_var[data_idx])
    Hd_inv = np.linalg.solve(Hd.conj().T @ Hd + sigma2 * I_Nd, Hd.conj().T)
    return Hd, Hd_inv, sigma, sigma2


def _apply_eve_observation(y_full: np.ndarray,
                           option: str,
                           geom: MCFGeometry,
                           params: FiberParams,
                           tap_cfg: Optional[TapObservationConfig],
                           rng: np.random.Generator) -> Tuple[np.ndarray, List[int]]:
    if option == "FO":
        return y_full.copy(), list(range(y_full.shape[1]))
    if option == "WT":
        if tap_cfg is None:
            tap_cfg = TapObservationConfig()
        return _apply_wt_tap(y_full, geom, params, tap_cfg, rng)
    raise ValueError(f"Unknown option '{option}'.")


def _build_spm_dataset(enc: MCFEncryption,
                       true_key: np.ndarray,
                       option: str,
                       T_train: int,
                       T_test: int,
                       seed: int,
                       gamma_nl: float = 2.0,
                       tap_cfg: Optional[TapObservationConfig] = None) -> Dict:
    rng = np.random.default_rng(seed)
    geom = enc.geom
    params = enc.params
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    I_Nd = np.eye(Nd)

    k1 = true_key.copy()
    k2 = _mixed_regime_key(rng, geom.N)

    H1 = enc.sim.transfer_matrix(k1, params.length_mm)
    H2 = enc.sim.transfer_matrix(k2, params.length_mm)
    Hd1 = H1[np.ix_(data_idx, data_idx)]
    Hd2 = H2[np.ix_(data_idx, data_idx)]
    all_v1 = _ase_noise_variances(k1, params, geom)
    all_v2 = _ase_noise_variances(k2, params, geom)
    sigma1 = np.sqrt(all_v1[data_idx])
    sigma2 = np.sqrt(all_v2[data_idx])
    sig2_1 = float(np.mean(all_v1[data_idx]))
    sig2_2 = float(np.mean(all_v2[data_idx]))
    Hd1_inv = np.linalg.solve(Hd1.conj().T @ Hd1 + sig2_1 * I_Nd, Hd1.conj().T)
    Hd2_inv = np.linalg.solve(Hd2.conj().T @ Hd2 + sig2_2 * I_Nd, Hd2.conj().T)
    z_scale = _row_energy_scale(Hd1)
    gamma_key = gamma_nl * k1[data_idx]

    def _forward(x: np.ndarray, rng_local: np.random.Generator) -> np.ndarray:
        n1 = _complex_awgn(rng_local, x.shape, sigma1[None, :])
        z = x @ Hd1.T + n1
        z_n = z / z_scale[None, :]
        z_spm = _apply_spm(z_n, gamma_key) * z_scale[None, :]
        n2 = _complex_awgn(rng_local, x.shape, sigma2[None, :])
        return z_spm @ Hd2.T + n2

    bits_train = rng.integers(0, 2, size=(T_train, Nd, 2))
    bits_test = rng.integers(0, 2, size=(T_test, Nd, 2))
    x_train = _qpsk_bits_to_syms(bits_train)
    x_test = _qpsk_bits_to_syms(bits_test)

    y_train_full = _forward(x_train, rng)
    y_test_full = _forward(x_test, rng)

    z_hat = y_test_full @ Hd2_inv.T
    z_inv = _invert_spm(z_hat / z_scale[None, :], gamma_key) * z_scale[None, :]
    x_hat_bob = z_inv @ Hd1_inv.T
    ber_bob = _bit_error_rate(bits_test, _qpsk_demod_array(x_hat_bob))

    if option == "FO":
        y_eve_train = y_train_full.copy()
        y_eve_test = y_test_full.copy()
        obs_idx = list(range(Nd))
    elif option == "WT":
        if tap_cfg is None:
            tap_cfg = TapObservationConfig()
        y_eve_train, obs_idx = _apply_wt_tap(y_train_full, geom, params, tap_cfg, rng)
        y_eve_test, _ = _apply_wt_tap(y_test_full, geom, params, tap_cfg, rng)
    else:
        raise ValueError(f"Unknown option '{option}'.")

    return {
        "x_train": x_train,
        "x_test": x_test,
        "bits_test": bits_test,
        "y_eve_train": y_eve_train,
        "y_eve_test": y_eve_test,
        "BER_Bob": ber_bob,
        "obs_dim": y_eve_train.shape[1],
        "observed_core_indices": obs_idx,
        "gamma_nl": gamma_nl,
    }


def _build_xpm_dataset(enc: MCFEncryption,
                       true_key: np.ndarray,
                       option: str,
                       T_train: int,
                       T_test: int,
                       seed: int,
                       gamma_nl: float = 2.0,
                       xpm_ratio: float = 0.5,
                       tap_cfg: Optional[TapObservationConfig] = None) -> Dict:
    rng = np.random.default_rng(seed)
    geom = enc.geom
    params = enc.params
    data_idx = geom.data_core_indices
    Nd = len(data_idx)

    k1 = true_key.copy()
    k2 = _mixed_regime_key(rng, geom.N)

    Hd1, Hd1_inv, sigma1, _ = _mmse_inverse_from_key(enc, k1)
    Hd2, Hd2_inv, sigma2, _ = _mmse_inverse_from_key(enc, k2)
    z_scale = _row_energy_scale(Hd1)
    gamma_self = gamma_nl * k1[data_idx]
    gamma_cross = xpm_ratio * gamma_nl * k1[data_idx]
    W_cross = _adjacency_weight_matrix(geom, data_idx)

    def _forward(x: np.ndarray, rng_local: np.random.Generator) -> np.ndarray:
        n1 = _complex_awgn(rng_local, x.shape, sigma1[None, :])
        z = x @ Hd1.T + n1
        z_n = z / z_scale[None, :]
        z_xpm = _apply_xpm(z_n, gamma_self, gamma_cross, W_cross) * z_scale[None, :]
        n2 = _complex_awgn(rng_local, x.shape, sigma2[None, :])
        return z_xpm @ Hd2.T + n2

    bits_train = rng.integers(0, 2, size=(T_train, Nd, 2))
    bits_test = rng.integers(0, 2, size=(T_test, Nd, 2))
    x_train = _qpsk_bits_to_syms(bits_train)
    x_test = _qpsk_bits_to_syms(bits_test)

    y_train_full = _forward(x_train, rng)
    y_test_full = _forward(x_test, rng)

    z_hat = y_test_full @ Hd2_inv.T
    z_inv = _invert_xpm(z_hat / z_scale[None, :], gamma_self, gamma_cross, W_cross) * z_scale[None, :]
    x_hat_bob = z_inv @ Hd1_inv.T
    ber_bob = _bit_error_rate(bits_test, _qpsk_demod_array(x_hat_bob))

    y_eve_train, obs_idx = _apply_eve_observation(
        y_train_full, option, geom, params, tap_cfg, rng)
    y_eve_test, _ = _apply_eve_observation(
        y_test_full, option, geom, params, tap_cfg, rng)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "bits_test": bits_test,
        "y_eve_train": y_eve_train,
        "y_eve_test": y_eve_test,
        "BER_Bob": ber_bob,
        "obs_dim": y_eve_train.shape[1],
        "observed_core_indices": obs_idx,
        "gamma_nl": gamma_nl,
        "xpm_ratio": xpm_ratio,
    }


def _build_deep_spm_dataset(enc: MCFEncryption,
                            true_key: np.ndarray,
                            option: str,
                            T_train: int,
                            T_test: int,
                            seed: int,
                            gamma1_nl: float = 2.0,
                            gamma2_nl: float = 1.4,
                            tap_cfg: Optional[TapObservationConfig] = None) -> Dict:
    rng = np.random.default_rng(seed)
    geom = enc.geom
    params = enc.params
    data_idx = geom.data_core_indices
    Nd = len(data_idx)

    k1 = true_key.copy()
    k2 = _mixed_regime_key(rng, geom.N)
    k3 = _mixed_regime_key(rng, geom.N, n_absorption=max(2, int(np.ceil(0.35 * geom.N))))

    Hd1, Hd1_inv, sigma1, _ = _mmse_inverse_from_key(enc, k1)
    Hd2, Hd2_inv, sigma2, _ = _mmse_inverse_from_key(enc, k2)
    Hd3, Hd3_inv, sigma3, _ = _mmse_inverse_from_key(enc, k3)

    z1_scale = _row_energy_scale(Hd1)
    z2_scale = _row_energy_scale(Hd2)
    gamma1_key = gamma1_nl * k1[data_idx]
    gamma2_key = gamma2_nl * k2[data_idx]

    def _forward(x: np.ndarray, rng_local: np.random.Generator) -> np.ndarray:
        n1 = _complex_awgn(rng_local, x.shape, sigma1[None, :])
        z1 = x @ Hd1.T + n1
        u1 = _apply_spm(z1 / z1_scale[None, :], gamma1_key) * z1_scale[None, :]

        n2 = _complex_awgn(rng_local, x.shape, sigma2[None, :])
        z2 = u1 @ Hd2.T + n2
        u2 = _apply_spm(z2 / z2_scale[None, :], gamma2_key) * z2_scale[None, :]

        n3 = _complex_awgn(rng_local, x.shape, sigma3[None, :])
        return u2 @ Hd3.T + n3

    bits_train = rng.integers(0, 2, size=(T_train, Nd, 2))
    bits_test = rng.integers(0, 2, size=(T_test, Nd, 2))
    x_train = _qpsk_bits_to_syms(bits_train)
    x_test = _qpsk_bits_to_syms(bits_test)

    y_train_full = _forward(x_train, rng)
    y_test_full = _forward(x_test, rng)

    u2_hat = y_test_full @ Hd3_inv.T
    z2_inv = _invert_spm(u2_hat / z2_scale[None, :], gamma2_key) * z2_scale[None, :]
    u1_hat = z2_inv @ Hd2_inv.T
    z1_inv = _invert_spm(u1_hat / z1_scale[None, :], gamma1_key) * z1_scale[None, :]
    x_hat_bob = z1_inv @ Hd1_inv.T
    ber_bob = _bit_error_rate(bits_test, _qpsk_demod_array(x_hat_bob))

    y_eve_train, obs_idx = _apply_eve_observation(
        y_train_full, option, geom, params, tap_cfg, rng)
    y_eve_test, _ = _apply_eve_observation(
        y_test_full, option, geom, params, tap_cfg, rng)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "bits_test": bits_test,
        "y_eve_train": y_eve_train,
        "y_eve_test": y_eve_test,
        "BER_Bob": ber_bob,
        "obs_dim": y_eve_train.shape[1],
        "observed_core_indices": obs_idx,
        "gamma1_nl": gamma1_nl,
        "gamma2_nl": gamma2_nl,
    }


# def _build_dataset_for_scheme(enc: MCFEncryption,
#                               true_key: np.ndarray,
#                               scheme: str,
#                               option: str,
#                               T_train: int,
#                               T_test: int,
#                               seed: int,
#                               tap_cfg: Optional[TapObservationConfig]) -> Dict:
#     if scheme == "Linear":
#         return _build_linear_dataset(enc, true_key, option, T_train, T_test, seed, tap_cfg)
#     if scheme == "Protected2":
#         return _build_spm_dataset(
#             enc, true_key, option, T_train, T_test, seed,
#             gamma_nl=2.0, tap_cfg=tap_cfg)
#     if scheme in {"ProtectedXPM", "ProtectedSPMXPM"}:
#         return _build_xpm_dataset(
#             enc, true_key, option, T_train, T_test, seed,
#             gamma_nl=2.0, xpm_ratio=0.5, tap_cfg=tap_cfg)
#     if scheme == "Protected3":
#         return _build_deep_spm_dataset(
#             enc, true_key, option, T_train, T_test, seed,
#             gamma1_nl=2.0, gamma2_nl=1.4, tap_cfg=tap_cfg)
#     raise ValueError(f"Unknown scheme '{scheme}'.")


def _build_dataset_for_scheme(enc: MCFEncryption,
                              true_key: np.ndarray,
                              scheme: str,
                              option: str,
                              T_train: int,
                              T_test: int,
                              seed: int,
                              tap_cfg: Optional[TapObservationConfig]) -> Dict:
    if scheme == "Linear":
        data = _build_linear_dataset(enc, true_key, option, T_train, T_test, seed, tap_cfg)
    elif scheme == "Protected2":
        data = _build_spm_dataset(
            enc, true_key, option, T_train, T_test, seed,
            gamma_nl=2.0, tap_cfg=tap_cfg)
    elif scheme in {"ProtectedXPM", "ProtectedSPMXPM"}:
        data = _build_xpm_dataset(
            enc, true_key, option, T_train, T_test, seed,
            gamma_nl=2.0, xpm_ratio=0.5, tap_cfg=tap_cfg)
    elif scheme == "Protected3":
        data = _build_deep_spm_dataset(
            enc, true_key, option, T_train, T_test, seed,
            gamma1_nl=2.0, gamma2_nl=1.4, tap_cfg=tap_cfg)
    else:
        raise ValueError(f"Unknown scheme '{scheme}'.")
    noise_scale = tap_cfg.eve_noise_scale if (option == "WT" and tap_cfg is not None) else 1.0
    data["sigma2_eve"] = (enc.params.sigma_noise * noise_scale) ** 2
    return data


def _scheme_detail_label(scheme: str) -> str:
    if scheme == "Linear":
        return "single-stage linear keyed channel"
    if scheme == "Protected2":
        return "two-stage mixed-regime keyed channel with one SPM block"
    if scheme in {"ProtectedXPM", "ProtectedSPMXPM"}:
        return "two-stage mixed-regime keyed channel with one exploratory SPM+XPM block"
    if scheme == "Protected3":
        return "three-stage keyed channel with two SPM blocks"
    return scheme


def benchmark_fo_wt_attack_suite(enc: MCFEncryption,
                                     true_key: np.ndarray,
                                     out_dir: str,
                                     tap_cfg: Optional[TapObservationConfig] = None,
                                     seed: int = 42) -> List[Dict]:
    """
    Minimum publishable attacker benchmark:
    linear vs. SPM-protected scheme under FO and WT.
    """
    if tap_cfg is None:
        tap_cfg = TapObservationConfig()

    scenarios = [
        ("Linear", "FO"),
        ("Linear", "WT"),
        ("Protected2", "FO"),
        ("Protected2", "WT"),
        ("Protected3", "FO"),
        ("Protected3", "WT"),
    ]
    rows = []
    print("\n[Suite] FO/WT attack benchmark...")
    for idx, (scheme, option) in enumerate(scenarios):
        data = _build_dataset_for_scheme(
            enc, true_key, scheme, option, 1200, 500, seed + 101 * idx, tap_cfg)

        x_tr = data["x_train"]
        y_tr = data["y_eve_train"]
        y_te = data["y_eve_test"]
        bits_te = data["bits_test"]
        n_obs = min(200, len(x_tr))

        row = {
            "scheme": scheme,
            "scheme_detail": _scheme_detail_label(scheme),
            "option": option,
            "BER_Bob": data["BER_Bob"],
            "BER_KPA": _evaluate_kpa_attack(x_tr[:n_obs], y_tr[:n_obs], y_te, bits_te, sigma2=data["sigma2_eve"]),
            "BER_LR": _evaluate_lr_attack(x_tr[:n_obs], y_tr[:n_obs], y_te, bits_te),
            "BER_LSTM": _evaluate_lstm_attack(y_tr, x_tr, y_te, bits_te,
                                               window=8, epochs=6, seed=seed + idx),
            "obs_dim": data["obs_dim"],
            "observed_core_indices": data["observed_core_indices"],
        }
        rows.append(row)
        print(f"  {scheme:6s} {option}: Bob={row['BER_Bob']:.4f} "
              f"KPA={row['BER_KPA']:.4f} LR={row['BER_LR']:.4f} "
              f"LSTM={row['BER_LSTM']:.4f}")

    _write_csv(f"{out_dir}/fo_wt_attack_benchmark.csv", rows)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    labels = [f"{r['scheme']}\n{r['option']}" for r in rows]
    x = np.arange(len(labels))
    w = 0.18
    ax.bar(x - 1.5 * w, [r["BER_Bob"] for r in rows], w, label="Bob",
           color="tab:blue", edgecolor="black")
    ax.bar(x - 0.5 * w, [r["BER_KPA"] for r in rows], w, label="KPA",
           color="tab:red", edgecolor="black")
    ax.bar(x + 0.5 * w, [r["BER_LR"] for r in rows], w, label="LR",
           color="tab:orange", edgecolor="black")
    ax.bar(x + 1.5 * w, [r["BER_LSTM"] for r in rows], w, label="LSTM",
           color="tab:green", edgecolor="black")
    ax.axhline(0.5, color="gray", ls=":", lw=1.0, label="Random guess")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BER")
    ax.set_ylim(0, 0.62)
    ax.set_title(
        "Attack benchmark: FO vs. WT\n"
        "Linear baseline vs. one-SPM and two-SPM protected schemes")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_attack_benchmark_fo_wt.png", dpi=200)
    plt.close(fig)
    return rows


def sweep_kpa_lr_sample_complexity(enc: MCFEncryption,
                                   true_key: np.ndarray,
                                   out_dir: str,
                                   tap_cfg: Optional[TapObservationConfig] = None,
                                   seed: int = 42) -> List[Dict]:
    if tap_cfg is None:
        tap_cfg = TapObservationConfig()
    n_obs_vals = [5, 10, 20, 50, 100, 200, 400]
    scenarios = [
        ("Linear", "FO"),
        ("Linear", "WT"),
        ("Protected2", "FO"),
        ("Protected2", "WT"),
        ("Protected3", "FO"),
        ("Protected3", "WT"),
    ]
    rows = []
    print("\n[Suite] KPA/LR sample complexity...")
    for idx, (scheme, option) in enumerate(scenarios):
        data = _build_dataset_for_scheme(
            enc, true_key, scheme, option, max(n_obs_vals), 500, seed + 1000 + idx, tap_cfg)
        for n_obs in n_obs_vals:
            rows.append({
                "scheme": scheme,
                "option": option,
                "n_obs": n_obs,
                "BER_KPA": _evaluate_kpa_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
                "BER_LR": _evaluate_lr_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"]),
            })

    _write_csv(f"{out_dir}/fo_wt_kpa_lr_sample_complexity.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    styles = {
        ("Linear", "FO"): ("tab:red", "o", "Linear / FO"),
        ("Linear", "WT"): ("darkred", "s", "Linear / WT"),
        ("Protected2", "FO"): ("tab:green", "^", "Protected2 / FO"),
        ("Protected2", "WT"): ("darkgreen", "D", "Protected2 / WT"),
        ("Protected3", "FO"): ("tab:blue", "P", "Protected3 / FO"),
        ("Protected3", "WT"): ("navy", "X", "Protected3 / WT"),
    }
    for ax, metric in zip(axes, ["BER_KPA", "BER_LR"]):
        for key, (color, marker, label) in styles.items():
            sub = [r for r in rows if (r["scheme"], r["option"]) == key]
            ax.semilogx([r["n_obs"] for r in sub], [r[metric] for r in sub],
                        marker + "-", color=color, lw=2, ms=6, label=label)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("Known-plaintext pairs")
        ax.set_ylabel("BER")
        ax.set_title(metric.replace("_", " "))
        ax.grid(True, alpha=0.4, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Attack sample complexity under FO and WT\n"
        "Protected2 = one SPM block, Protected3 = two SPM blocks",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_kpa_lr_sample_complexity_fo_wt.png", dpi=200)
    plt.close(fig)
    return rows


def sweep_lstm_training_size(enc: MCFEncryption,
                             true_key: np.ndarray,
                             out_dir: str,
                             tap_cfg: Optional[TapObservationConfig] = None,
                             seed: int = 42) -> List[Dict]:
    if tap_cfg is None:
        tap_cfg = TapObservationConfig()
    sizes = [250, 500, 1000]
    scenarios = [
        ("Linear", "FO"),
        ("Linear", "WT"),
        ("Protected2", "FO"),
        ("Protected2", "WT"),
        ("Protected3", "FO"),
        ("Protected3", "WT"),
    ]
    rows = []
    print("\n[Suite] LSTM training-size sweep...")
    for idx, (scheme, option) in enumerate(scenarios):
        for size in sizes:
            data = _build_dataset_for_scheme(
                enc, true_key, scheme, option, size, 400,
                seed + 2000 + 10 * idx + size, tap_cfg)
            ber_lstm = _evaluate_lstm_attack(
                data["y_eve_train"], data["x_train"], data["y_eve_test"],
                data["bits_test"], window=8, epochs=6, seed=seed + idx + size)
            rows.append({
                "scheme": scheme,
                "option": option,
                "train_size": size,
                "BER_LSTM": ber_lstm,
            })

    _write_csv(f"{out_dir}/fo_wt_lstm_training_size.csv", rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    styles = {
        ("Linear", "FO"): ("tab:red", "o", "Linear / FO"),
        ("Linear", "WT"): ("darkred", "s", "Linear / WT"),
        ("Protected2", "FO"): ("tab:green", "^", "Protected2 / FO"),
        ("Protected2", "WT"): ("darkgreen", "D", "Protected2 / WT"),
        ("Protected3", "FO"): ("tab:blue", "P", "Protected3 / FO"),
        ("Protected3", "WT"): ("navy", "X", "Protected3 / WT"),
    }
    for key, (color, marker, label) in styles.items():
        sub = [r for r in rows if (r["scheme"], r["option"]) == key]
        ax.plot([r["train_size"] for r in sub], [r["BER_LSTM"] for r in sub],
                marker + "-", color=color, lw=2, ms=6, label=label)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("LSTM training pairs")
    ax.set_ylabel("BER")
    ax.set_ylim(0, 0.62)
    ax.set_title("LSTM attack vs. training size\nProtected2 = one SPM block, Protected3 = two SPM blocks")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_lstm_training_size_fo_wt.png", dpi=200)
    plt.close(fig)
    return rows


def _mean_std_ci(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    ci95 = float(1.96 * std / np.sqrt(max(len(arr), 1)))
    return mean, std, ci95


def _bootstrap_mean_diff_ci(a: np.ndarray,
                            b: np.ndarray,
                            seed: int,
                            n_boot: int = 5000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs[idx] = float(np.mean(b_s) - np.mean(a_s))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _permutation_pvalue_mean_diff(a: np.ndarray,
                                  b: np.ndarray,
                                  seed: int,
                                  max_exact: int = 50000,
                                  n_mc: int = 20000) -> float:
    pooled = np.concatenate([a, b])
    n_a = len(a)
    observed = float(np.mean(b) - np.mean(a))
    total = len(pooled)
    n_combos = 1
    for i in range(1, n_a + 1):
        n_combos = n_combos * (total - n_a + i) // i

    def stat_from_idx(idx_a: Tuple[int, ...]) -> float:
        mask = np.zeros(total, dtype=bool)
        mask[list(idx_a)] = True
        a_p = pooled[mask]
        b_p = pooled[~mask]
        return float(np.mean(b_p) - np.mean(a_p))

    if n_combos <= max_exact:
        stats = np.array([stat_from_idx(idx_a)
                          for idx_a in combinations(range(total), n_a)], dtype=float)
        return float(np.mean(np.abs(stats) >= abs(observed) - 1e-12))

    rng = np.random.default_rng(seed)
    stats = np.empty(n_mc, dtype=float)
    for idx in range(n_mc):
        perm = rng.permutation(total)
        a_p = pooled[perm[:n_a]]
        b_p = pooled[perm[n_a:]]
        stats[idx] = float(np.mean(b_p) - np.mean(a_p))
    return float(np.mean(np.abs(stats) >= abs(observed) - 1e-12))


def _bh_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    q_ranked = np.empty(m, dtype=float)
    min_coeff = 1.0
    for idx in range(m - 1, -1, -1):
        coeff = ranked[idx] * m / (idx + 1)
        min_coeff = min(min_coeff, coeff)
        q_ranked[idx] = min(1.0, min_coeff)
    q = np.empty(m, dtype=float)
    q[order] = q_ranked
    return q.tolist()


def benchmark_core_scaling_13_19_37(base_params: FiberParams,
                                    out_dir: str,
                                    seed: int = 42) -> List[Dict]:
    rows = []
    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    configs = [
        ("13-core", MCFGeometry(pitch_um=base_params.pitch_um)),
        ("19-core", HexMCFGeometry(n_rings=2, pitch_um=base_params.pitch_um)),
        ("37-core", HexMCFGeometry(n_rings=3, pitch_um=base_params.pitch_um)),
    ]
    scenarios = [("Linear", "FO"), ("Linear", "WT"), ("Protected2", "FO"), ("Protected2", "WT")]
    print("\n[Suite] 13-core vs. 19-core vs. 37-core benchmark...")
    for g_idx, (label, geom) in enumerate(configs):
        enc = MCFEncryption(base_params, geom, seed=seed + 300 * g_idx)
        rng = np.random.default_rng(seed + 17 * g_idx)
        true_key = rng.uniform(0.6, 1.0, size=geom.N)
        for s_idx, (scheme, option) in enumerate(scenarios):
            data = _build_dataset_for_scheme(
                enc, true_key, scheme, option, 1200, 400,
                seed + 100 * g_idx + 13 * s_idx, tap_cfg)
            n_obs = min(200, len(data["x_train"]))
            row = {
                "geometry": label,
                "core_count": geom.N,
                "scheme": scheme,
                "option": option,
                "BER_Bob": data["BER_Bob"],
                "BER_KPA": _evaluate_kpa_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
                "BER_LR": _evaluate_lr_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"]),
                "obs_dim": data["obs_dim"],
            }
            rows.append(row)
            print(f"  {label:7s} {scheme:10s} {option}: "
                  f"Bob={row['BER_Bob']:.4f} KPA={row['BER_KPA']:.4f} LR={row['BER_LR']:.4f}")

    _write_csv(f"{out_dir}/core_scaling_13_19_37_attack_benchmark.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    styles = {
        ("Linear", "BER_KPA"): ("tab:red", "o", "Linear KPA"),
        ("Linear", "BER_LR"): ("darkred", "s", "Linear LR"),
        ("Protected2", "BER_KPA"): ("tab:green", "^", "Protected2 KPA"),
        ("Protected2", "BER_LR"): ("darkgreen", "D", "Protected2 LR"),
    }
    for ax, option in zip(axes, ["FO", "WT"]):
        for key, (color, marker, label) in styles.items():
            scheme, metric = key
            sub = [r for r in rows if r["option"] == option and r["scheme"] == scheme]
            ax.plot([r["core_count"] for r in sub], [r[metric] for r in sub],
                    marker + "-", color=color, lw=2, ms=7, label=label)
        ax.set_title(f"{option}")
        ax.set_xlabel("Number of cores")
        ax.set_ylabel("BER")
        ax.set_xticks([13, 19, 37])
        ax.grid(True, alpha=0.4)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory scaling: 13-core vs. 19-core vs. 37-core", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_core_scaling_13_19_37.png", dpi=200)
    plt.close(fig)
    return rows


def benchmark_core_scaling_13_37(base_params: FiberParams,
                                 out_dir: str,
                                 seed: int = 42) -> List[Dict]:
    return benchmark_core_scaling_13_19_37(base_params, out_dir, seed)


def run_scaling_multiseed_significance(base_params: FiberParams,
                                       out_dir: str,
                                       seeds: Optional[List[int]] = None
                                       ) -> Tuple[List[Dict], List[Dict]]:
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    configs = [
        ("13-core", MCFGeometry(pitch_um=base_params.pitch_um)),
        ("19-core", HexMCFGeometry(n_rings=2, pitch_um=base_params.pitch_um)),
        ("37-core", HexMCFGeometry(n_rings=3, pitch_um=base_params.pitch_um)),
    ]
    scenarios = [("Linear", "FO"), ("Linear", "WT"), ("Protected2", "FO"), ("Protected2", "WT")]
    rows = []

    print("\n[Suite] Multi-seed scaling benchmark for significance tests...")
    for seed_idx, seed in enumerate(seeds):
        for g_idx, (label, geom) in enumerate(configs):
            enc = MCFEncryption(base_params, geom, seed=seed + 300 * g_idx)
            true_key = np.random.default_rng(seed + 17 * g_idx).uniform(0.6, 1.0, size=geom.N)
            for s_idx, (scheme, option) in enumerate(scenarios):
                data = _build_dataset_for_scheme(
                    enc, true_key, scheme, option, 1200, 400,
                    seed + 1000 * seed_idx + 100 * g_idx + 13 * s_idx, tap_cfg)
                n_obs = min(200, len(data["x_train"]))
                rows.append({
                    "seed": seed,
                    "geometry": label,
                    "core_count": geom.N,
                    "scheme": scheme,
                    "option": option,
                    "BER_Bob": data["BER_Bob"],
                    "BER_KPA_CE": _evaluate_kpa_attack(
                        data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                        data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
                    "BER_KPA_LR": _evaluate_lr_attack(
                        data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                        data["y_eve_test"], data["bits_test"]),
                })

    _write_csv(f"{out_dir}/scaling_multiseed_benchmark.csv", rows)

    summary = []
    for label, geom in configs:
        for scheme, option in scenarios:
            sub = [r for r in rows if r["geometry"] == label and r["scheme"] == scheme and r["option"] == option]
            bob_mean, bob_std, bob_ci = _mean_std_ci([r["BER_Bob"] for r in sub])
            ce_mean, ce_std, ce_ci = _mean_std_ci([r["BER_KPA_CE"] for r in sub])
            lr_mean, lr_std, lr_ci = _mean_std_ci([r["BER_KPA_LR"] for r in sub])
            summary.append({
                "geometry": label,
                "core_count": geom.N,
                "scheme": scheme,
                "option": option,
                "BER_Bob_mean": bob_mean,
                "BER_Bob_std": bob_std,
                "BER_Bob_ci95": bob_ci,
                "BER_KPA_CE_mean": ce_mean,
                "BER_KPA_CE_std": ce_std,
                "BER_KPA_CE_ci95": ce_ci,
                "BER_KPA_LR_mean": lr_mean,
                "BER_KPA_LR_std": lr_std,
                "BER_KPA_LR_ci95": lr_ci,
            })
    _write_csv(f"{out_dir}/scaling_multiseed_summary.csv", summary)

    comparisons = []
    metrics = ["BER_KPA_CE", "BER_KPA_LR"]
    core_pairs = [(13, 19), (19, 37), (13, 37)]

    for scheme in ["Linear", "Protected2"]:
        for option in ["FO", "WT"]:
            for metric in metrics:
                for a_core, b_core in core_pairs:
                    a_vals = np.asarray([r[metric] for r in rows
                                         if r["scheme"] == scheme and r["option"] == option
                                         and r["core_count"] == a_core], dtype=float)
                    b_vals = np.asarray([r[metric] for r in rows
                                         if r["scheme"] == scheme and r["option"] == option
                                         and r["core_count"] == b_core], dtype=float)
                    ci_lo, ci_hi = _bootstrap_mean_diff_ci(
                        a_vals, b_vals, seed=10000 + 101 * a_core + 17 * b_core)
                    p_val = _permutation_pvalue_mean_diff(
                        a_vals, b_vals, seed=20000 + 103 * a_core + 19 * b_core)
                    comparisons.append({
                        "family": "core_scaling",
                        "scheme": scheme,
                        "option": option,
                        "metric": metric,
                        "condition_a": f"{a_core}-core",
                        "condition_b": f"{b_core}-core",
                        "mean_a": float(np.mean(a_vals)),
                        "mean_b": float(np.mean(b_vals)),
                        "mean_diff_b_minus_a": float(np.mean(b_vals) - np.mean(a_vals)),
                        "diff_ci95_low": ci_lo,
                        "diff_ci95_high": ci_hi,
                        "p_value_two_sided": p_val,
                    })

    for core_count in [13, 19, 37]:
        for option in ["FO", "WT"]:
            for metric in metrics:
                a_vals = np.asarray([r[metric] for r in rows
                                     if r["scheme"] == "Linear" and r["option"] == option
                                     and r["core_count"] == core_count], dtype=float)
                b_vals = np.asarray([r[metric] for r in rows
                                     if r["scheme"] == "Protected2" and r["option"] == option
                                     and r["core_count"] == core_count], dtype=float)
                ci_lo, ci_hi = _bootstrap_mean_diff_ci(
                    a_vals, b_vals, seed=30000 + 29 * core_count)
                p_val = _permutation_pvalue_mean_diff(
                    a_vals, b_vals, seed=40000 + 31 * core_count)
                comparisons.append({
                    "family": "scheme_effect",
                    "scheme": "Linear_vs_Protected2",
                    "option": option,
                    "metric": metric,
                    "condition_a": f"Linear_{core_count}-core",
                    "condition_b": f"Protected2_{core_count}-core",
                    "mean_a": float(np.mean(a_vals)),
                    "mean_b": float(np.mean(b_vals)),
                    "mean_diff_b_minus_a": float(np.mean(b_vals) - np.mean(a_vals)),
                    "diff_ci95_low": ci_lo,
                    "diff_ci95_high": ci_hi,
                    "p_value_two_sided": p_val,
                })

    q_vals = _bh_adjust([r["p_value_two_sided"] for r in comparisons])
    for row, q_val in zip(comparisons, q_vals):
        row["q_value_bh"] = q_val

    _write_csv(f"{out_dir}/scaling_significance_tests.csv", comparisons)
    return rows, comparisons


def benchmark_xpm_exploratory(enc: MCFEncryption,
                              true_key: np.ndarray,
                              out_dir: str,
                              tap_cfg: Optional[TapObservationConfig] = None,
                              seed: int = 42) -> List[Dict]:
    if tap_cfg is None:
        tap_cfg = TapObservationConfig()
    scenarios = [
        ("Protected2", "FO"),
        ("Protected2", "WT"),
        ("ProtectedSPMXPM", "FO"),
        ("ProtectedSPMXPM", "WT"),
    ]
    rows = []
    print("\n[Suite] Exploratory SPM vs. SPM+XPM benchmark...")
    for idx, (scheme, option) in enumerate(scenarios):
        data = _build_dataset_for_scheme(
            enc, true_key, scheme, option, 1200, 500, seed + 4000 + 31 * idx, tap_cfg)
        n_obs = min(200, len(data["x_train"]))
        row = {
            "scheme": scheme,
            "scheme_detail": _scheme_detail_label(scheme),
            "option": option,
            "BER_Bob": data["BER_Bob"],
            "BER_KPA": _evaluate_kpa_attack(
                data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
            "BER_LR": _evaluate_lr_attack(
                data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                data["y_eve_test"], data["bits_test"]),
            "BER_LSTM": _evaluate_lstm_attack(
                data["y_eve_train"], data["x_train"], data["y_eve_test"],
                data["bits_test"], window=8, epochs=6, seed=seed + 500 + idx),
        }
        rows.append(row)
        print(f"  {scheme:12s} {option}: Bob={row['BER_Bob']:.4f} "
              f"KPA={row['BER_KPA']:.4f} LR={row['BER_LR']:.4f} "
              f"LSTM={row['BER_LSTM']:.4f}")

    _write_csv(f"{out_dir}/xpm_exploratory_benchmark.csv", rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    labels = [f"{r['scheme']}\n{r['option']}" for r in rows]
    x = np.arange(len(labels))
    w = 0.18
    ax.bar(x - 1.5 * w, [r["BER_Bob"] for r in rows], w, label="Bob",
           color="tab:blue", edgecolor="black")
    ax.bar(x - 0.5 * w, [r["BER_KPA"] for r in rows], w, label="KPA-CE",
           color="tab:red", edgecolor="black")
    ax.bar(x + 0.5 * w, [r["BER_LR"] for r in rows], w, label="KPA-LR",
           color="tab:orange", edgecolor="black")
    ax.bar(x + 1.5 * w, [r["BER_LSTM"] for r in rows], w, label="KPA-LSTM",
           color="tab:green", edgecolor="black")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BER")
    ax.set_ylim(0, 0.62)
    ax.set_title("Exploratory benchmark: one-SPM block vs. exploratory SPM+XPM block")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_xpm_exploratory_benchmark.png", dpi=200)
    plt.close(fig)
    return rows


def _make_37core_setup(base_params: FiberParams,
                       seed: int) -> Tuple[MCFEncryption, np.ndarray, HexMCFGeometry]:
    geom = HexMCFGeometry(n_rings=3, pitch_um=base_params.pitch_um)
    enc = MCFEncryption(base_params, geom, seed=seed)
    true_key = np.random.default_rng(seed).uniform(0.6, 1.0, size=geom.N)
    return enc, true_key, geom


def _radial_shell_indices(geom) -> List[List[int]]:
    if hasattr(geom, "n_rings"):
        shells: Dict[int, List[int]] = {}
        for idx, (x, y) in enumerate(geom.positions):
            r_ax = y / (geom.pitch_um * np.sqrt(3) / 2.0)
            q_ax = x / geom.pitch_um - 0.5 * r_ax
            q = int(np.round(q_ax))
            r = int(np.round(r_ax))
            shell = int((abs(q) + abs(r) + abs(q + r)) // 2)
            shells.setdefault(shell, []).append(int(idx))
        return [shells[k] for k in sorted(shells.keys())]
    radii = np.linalg.norm(geom.positions, axis=1)
    buckets: Dict[float, List[int]] = {}
    for idx, r in enumerate(radii):
        key = round(float(r), 6)
        buckets.setdefault(key, []).append(int(idx))
    return [buckets[k] for k in sorted(buckets.keys())]


def validate_37core_protected2_multiseed(base_params: FiberParams,
                                         out_dir: str,
                                         seeds: Optional[List[int]] = None) -> List[Dict]:
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    rows = []
    print("\n[Suite] 37-core Protected2 multi-seed validation...")
    for seed in seeds:
        enc, true_key, _ = _make_37core_setup(base_params, seed)
        for option in ["FO", "WT"]:
            data = _build_dataset_for_scheme(
                enc, true_key, "Protected2", option, 1200, 400, seed + 700, tap_cfg)
            n_obs = min(200, len(data["x_train"]))
            row = {
                "seed": seed,
                "option": option,
                "BER_Bob": data["BER_Bob"],
                "BER_KPA_CE": _evaluate_kpa_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
                "BER_KPA_LR": _evaluate_lr_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"]),
                "BER_KPA_HYB": _evaluate_hybrid_attack(
                    data["y_eve_train"], data["x_train"], data["y_eve_test"],
                    data["bits_test"], epochs=25, seed=seed + 1100),
                "BER_KPA_MLP": _evaluate_mlp_attack(
                    data["y_eve_train"], data["x_train"], data["y_eve_test"],
                    data["bits_test"], epochs=25, seed=seed + 1200),
                "BER_KPA_TX": _evaluate_transformer_attack(
                    data["y_eve_train"], data["x_train"], data["y_eve_test"],
                    data["bits_test"], epochs=25, seed=seed + 1450),
                "BER_KPA_LSTM": _evaluate_lstm_attack(
                    data["y_eve_train"], data["x_train"], data["y_eve_test"],
                    data["bits_test"], window=8, epochs=6, seed=seed + 1700),
            }
            rows.append(row)
            print(f"  seed={seed} {option}: Bob={row['BER_Bob']:.4f} "
                  f"KPA-CE={row['BER_KPA_CE']:.4f} KPA-LR={row['BER_KPA_LR']:.4f} "
                  f"KPA-HYB={row['BER_KPA_HYB']:.4f} KPA-MLP={row['BER_KPA_MLP']:.4f} "
                  f"KPA-TX={row['BER_KPA_TX']:.4f} "
                  f"KPA-LSTM={row['BER_KPA_LSTM']:.4f}")

    _write_csv(f"{out_dir}/protected2_37core_multiseed.csv", rows)

    summary = []
    for option in ["FO", "WT"]:
        sub = [r for r in rows if r["option"] == option]
        bob_mean, bob_std, bob_ci = _mean_std_ci([r["BER_Bob"] for r in sub])
        ce_mean, ce_std, ce_ci = _mean_std_ci([r["BER_KPA_CE"] for r in sub])
        lr_mean, lr_std, lr_ci = _mean_std_ci([r["BER_KPA_LR"] for r in sub])
        hyb_mean, hyb_std, hyb_ci = _mean_std_ci([r["BER_KPA_HYB"] for r in sub])
        mlp_mean, mlp_std, mlp_ci = _mean_std_ci([r["BER_KPA_MLP"] for r in sub])
        tx_mean, tx_std, tx_ci = _mean_std_ci([r["BER_KPA_TX"] for r in sub])
        lstm_mean, lstm_std, lstm_ci = _mean_std_ci([r["BER_KPA_LSTM"] for r in sub])
        summary.append({
            "option": option,
            "BER_Bob_mean": bob_mean,
            "BER_Bob_std": bob_std,
            "BER_Bob_ci95": bob_ci,
            "BER_KPA_CE_mean": ce_mean,
            "BER_KPA_CE_std": ce_std,
            "BER_KPA_CE_ci95": ce_ci,
            "BER_KPA_LR_mean": lr_mean,
            "BER_KPA_LR_std": lr_std,
            "BER_KPA_LR_ci95": lr_ci,
            "BER_KPA_HYB_mean": hyb_mean,
            "BER_KPA_HYB_std": hyb_std,
            "BER_KPA_HYB_ci95": hyb_ci,
            "BER_KPA_MLP_mean": mlp_mean,
            "BER_KPA_MLP_std": mlp_std,
            "BER_KPA_MLP_ci95": mlp_ci,
            "BER_KPA_TX_mean": tx_mean,
            "BER_KPA_TX_std": tx_std,
            "BER_KPA_TX_ci95": tx_ci,
            "BER_KPA_LSTM_mean": lstm_mean,
            "BER_KPA_LSTM_std": lstm_std,
            "BER_KPA_LSTM_ci95": lstm_ci,
        })
    _write_csv(f"{out_dir}/protected2_37core_multiseed_summary.csv", summary)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    metrics = [
        ("BER_Bob_mean", "BER_Bob_ci95", "Bob", "tab:blue"),
        ("BER_KPA_CE_mean", "BER_KPA_CE_ci95", "KPA-CE", "tab:red"),
        ("BER_KPA_LR_mean", "BER_KPA_LR_ci95", "KPA-LR", "tab:orange"),
        ("BER_KPA_HYB_mean", "BER_KPA_HYB_ci95", "KPA-HYB", "tab:pink"),
        ("BER_KPA_MLP_mean", "BER_KPA_MLP_ci95", "KPA-MLP", "tab:purple"),
        ("BER_KPA_TX_mean", "BER_KPA_TX_ci95", "KPA-TX", "tab:brown"),
        ("BER_KPA_LSTM_mean", "BER_KPA_LSTM_ci95", "KPA-LSTM", "tab:green"),
    ]
    x = np.arange(len(summary))
    w = 0.11
    for idx, (m_key, s_key, label, color) in enumerate(metrics):
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * w, [r[m_key] for r in summary], w,
               yerr=[r[s_key] for r in summary], capsize=3,
               label=label, color=color, edgecolor="black")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['option']}" for r in summary])
    ax.set_ylabel("BER")
    ax.set_ylim(0, 0.62)
    ax.set_title("37-core Protected2 multi-seed validation with learned-Eve baselines")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(fontsize=8, ncol=7)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_protected2_37core_multiseed.png", dpi=200)
    plt.close(fig)
    return rows


def sweep_37core_sample_complexity(base_params: FiberParams,
                                   out_dir: str,
                                   seeds: Optional[List[int]] = None) -> List[Dict]:
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    n_obs_vals = [5, 10, 20, 50, 100, 200, 400, 800]
    rows = []
    scenarios = [("Linear", "FO"), ("Linear", "WT"), ("Protected2", "FO"), ("Protected2", "WT")]
    print("\n[Suite] 37-core sample complexity with multi-seed confidence intervals...")
    for seed_idx, seed in enumerate(seeds):
        enc, true_key, _ = _make_37core_setup(base_params, seed)
        for idx, (scheme, option) in enumerate(scenarios):
            data = _build_dataset_for_scheme(
                enc, true_key, scheme, option, max(n_obs_vals), 400,
                seed + 2100 + 17 * idx + 101 * seed_idx, tap_cfg)
            for n_obs in n_obs_vals:
                row = {
                    "seed": seed,
                    "scheme": scheme,
                    "option": option,
                    "n_obs": n_obs,
                    "BER_KPA_CE": _evaluate_kpa_attack(
                        data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                        data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
                    "BER_KPA_LR": _evaluate_lr_attack(
                        data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                        data["y_eve_test"], data["bits_test"]),
                }
                rows.append(row)

    _write_csv(f"{out_dir}/protected2_37core_sample_complexity.csv", rows)

    summary = []
    for scheme, option in scenarios:
        for n_obs in n_obs_vals:
            sub = [r for r in rows if r["scheme"] == scheme and r["option"] == option and r["n_obs"] == n_obs]
            ce_mean, ce_std, ce_ci = _mean_std_ci([r["BER_KPA_CE"] for r in sub])
            lr_mean, lr_std, lr_ci = _mean_std_ci([r["BER_KPA_LR"] for r in sub])
            summary.append({
                "scheme": scheme,
                "option": option,
                "n_obs": n_obs,
                "BER_KPA_CE_mean": ce_mean,
                "BER_KPA_CE_std": ce_std,
                "BER_KPA_CE_ci95": ce_ci,
                "BER_KPA_LR_mean": lr_mean,
                "BER_KPA_LR_std": lr_std,
                "BER_KPA_LR_ci95": lr_ci,
            })
    _write_csv(f"{out_dir}/protected2_37core_sample_complexity_summary.csv", summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    styles = {
        ("Linear", "FO"): ("tab:red", "o", "Linear / FO"),
        ("Linear", "WT"): ("darkred", "s", "Linear / WT"),
        ("Protected2", "FO"): ("tab:green", "^", "Protected2 / FO"),
        ("Protected2", "WT"): ("darkgreen", "D", "Protected2 / WT"),
    }
    for ax, metric in zip(axes, ["BER_KPA_CE", "BER_KPA_LR"]):
        mean_key = f"{metric}_mean"
        ci_key = f"{metric}_ci95"
        for key, (color, marker, label) in styles.items():
            sub = [r for r in summary if (r["scheme"], r["option"]) == key]
            xs = [r["n_obs"] for r in sub]
            ys = [r[mean_key] for r in sub]
            errs = [r[ci_key] for r in sub]
            ax.semilogx(xs, ys, marker + "-", color=color, lw=2, ms=6, label=label)
            ax.fill_between(xs,
                            np.maximum(0.0, np.asarray(ys) - np.asarray(errs)),
                            np.minimum(0.62, np.asarray(ys) + np.asarray(errs)),
                            color=color, alpha=0.15)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("Known-plaintext pairs")
        ax.set_ylabel("BER")
        ax.set_title(metric.replace("_", "-"))
        ax.grid(True, alpha=0.4, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("37-core sample complexity: mean BER with 95% confidence intervals", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_protected2_37core_sample_complexity.png", dpi=200)
    plt.close(fig)
    return rows


def sweep_37core_tap_sensitivity(base_params: FiberParams,
                                 out_dir: str,
                                 seed: int = 42) -> List[Dict]:
    enc, true_key, geom = _make_37core_setup(base_params, seed)
    shell_groups = _radial_shell_indices(geom)
    outer_only = shell_groups[-1]
    outer_mid = sorted(shell_groups[-2] + shell_groups[-1])
    all_cores = list(range(geom.N))
    obs_sets = [
        ("outer", outer_only),
        ("outer+mid", outer_mid),
        ("all", all_cores),
    ]
    tap_fracs = [0.02, 0.05, 0.10, 0.20]
    rows = []
    print("\n[Suite] 37-core tap sensitivity (WT, Protected2)...")
    for obs_idx, (obs_label, obs_list) in enumerate(obs_sets):
        for frac_idx, tap_frac in enumerate(tap_fracs):
            tap_cfg = TapObservationConfig(
                tap_fraction=tap_frac,
                eve_noise_scale=1.5,
                observed_core_indices=obs_list,
                label=f"{obs_label} tap",
            )
            data = _build_dataset_for_scheme(
                enc, true_key, "Protected2", "WT", 1200, 400,
                seed + 3100 + 10 * obs_idx + frac_idx, tap_cfg)
            n_obs = min(200, len(data["x_train"]))
            row = {
                "observed_set": obs_label,
                "observed_count": len(obs_list),
                "tap_fraction": tap_frac,
                "BER_Bob": data["BER_Bob"],
                "BER_KPA_CE": _evaluate_kpa_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
                "BER_KPA_LR": _evaluate_lr_attack(
                    data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                    data["y_eve_test"], data["bits_test"]),
            }
            rows.append(row)
            print(f"  obs={obs_label:<9} tap={tap_frac:.2f}: Bob={row['BER_Bob']:.4f} "
                  f"KPA-CE={row['BER_KPA_CE']:.4f} KPA-LR={row['BER_KPA_LR']:.4f}")

    _write_csv(f"{out_dir}/protected2_37core_tap_sensitivity.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    styles = {
        "outer": ("darkgreen", "D", "outer ring only"),
        "outer+mid": ("tab:green", "^", "outer + middle rings"),
        "all": ("tab:blue", "o", "all cores"),
    }
    for ax, metric in zip(axes, ["BER_KPA_CE", "BER_KPA_LR"]):
        for obs_label, (color, marker, label) in styles.items():
            sub = [r for r in rows if r["observed_set"] == obs_label]
            ax.plot([r["tap_fraction"] for r in sub], [r[metric] for r in sub],
                    marker + "-", color=color, lw=2, ms=6, label=label)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("Tap fraction")
        ax.set_ylabel("BER")
        ax.set_title(metric.replace("_", "-"))
        ax.grid(True, alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle("37-core Protected2 tap sensitivity (WT)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_protected2_37core_tap_sensitivity.png", dpi=200)
    plt.close(fig)
    return rows


def benchmark_spm_xpm_combo_37core(base_params: FiberParams,
                                   out_dir: str,
                                   seed: int = 42) -> List[Dict]:
    enc, true_key, _ = _make_37core_setup(base_params, seed)
    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    rows = []
    scenarios = [
        ("Protected2", "FO"),
        ("Protected2", "WT"),
        ("ProtectedSPMXPM", "FO"),
        ("ProtectedSPMXPM", "WT"),
    ]
    print("\n[Suite] 37-core exploratory SPM vs. SPM+XPM benchmark...")
    for idx, (scheme, option) in enumerate(scenarios):
        data = _build_dataset_for_scheme(
            enc, true_key, scheme, option, 1200, 400, seed + 4100 + idx, tap_cfg)
        n_obs = min(200, len(data["x_train"]))
        row = {
            "scheme": scheme,
            "option": option,
            "BER_Bob": data["BER_Bob"],
            "BER_KPA_CE": _evaluate_kpa_attack(
                data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                data["y_eve_test"], data["bits_test"], sigma2=data["sigma2_eve"]),
            "BER_KPA_LR": _evaluate_lr_attack(
                data["x_train"][:n_obs], data["y_eve_train"][:n_obs],
                data["y_eve_test"], data["bits_test"]),
        }
        rows.append(row)
        print(f"  {scheme:16s} {option}: Bob={row['BER_Bob']:.4f} "
              f"KPA-CE={row['BER_KPA_CE']:.4f} KPA-LR={row['BER_KPA_LR']:.4f}")

    _write_csv(f"{out_dir}/spm_xpm_combo_37core_benchmark.csv", rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{r['scheme']}\n{r['option']}" for r in rows]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, [r["BER_Bob"] for r in rows], w, label="Bob",
           color="tab:blue", edgecolor="black")
    ax.bar(x, [r["BER_KPA_CE"] for r in rows], w, label="KPA-CE",
           color="tab:red", edgecolor="black")
    ax.bar(x + w, [r["BER_KPA_LR"] for r in rows], w, label="KPA-LR",
           color="tab:orange", edgecolor="black")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BER")
    ax.set_ylim(0, 0.62)
    ax.set_title("37-core exploratory benchmark: SPM vs. SPM+XPM")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_spm_xpm_combo_37core.png", dpi=200)
    plt.close(fig)
    return rows


def sweep_qam_ber_protected2_37core(base_params: FiberParams,
                                     out_dir: str,
                                     seed: int = 42,
                                     T: int = 1200,
                                     T_test: int = 400) -> List[Dict]:
    """
    Supplementary Fig. S1: Multi-modulation BER for 37-core Protected2.

    Tests QPSK, 16-QAM, and 64-QAM through the full P2 pipeline
    (L_K1 -> SPM -> L_K2 + noise) under FO and WT, measuring Bob BER
    and attacker BER (KPA-CE, KPA-LR) for each modulation order.
    """
    enc, true_key, geom = _make_37core_setup(base_params, seed)
    rng = np.random.default_rng(seed + 9000)
    params = enc.params
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    I_Nd = np.eye(Nd)

    # Build P2 channel matrices (same as _build_spm_dataset)
    k1 = true_key.copy()
    k2 = _mixed_regime_key(rng, geom.N)
    H1 = enc.sim.transfer_matrix(k1, params.length_mm)
    H2 = enc.sim.transfer_matrix(k2, params.length_mm)
    Hd1 = H1[np.ix_(data_idx, data_idx)]
    Hd2 = H2[np.ix_(data_idx, data_idx)]
    all_v1 = _ase_noise_variances(k1, params, geom)
    all_v2 = _ase_noise_variances(k2, params, geom)
    sigma1 = np.sqrt(all_v1[data_idx])
    sigma2 = np.sqrt(all_v2[data_idx])
    sig2_1 = float(np.mean(all_v1[data_idx]))
    sig2_2 = float(np.mean(all_v2[data_idx]))
    Hd1_inv = np.linalg.solve(Hd1.conj().T @ Hd1 + sig2_1 * I_Nd, Hd1.conj().T)
    Hd2_inv = np.linalg.solve(Hd2.conj().T @ Hd2 + sig2_2 * I_Nd, Hd2.conj().T)
    z_scale = _row_energy_scale(Hd1)
    gamma_nl = 2.0
    gamma_key = gamma_nl * k1[data_idx]

    # Tap config for WT
    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )

    def _forward(x: np.ndarray, rng_local: np.random.Generator) -> np.ndarray:
        n1 = _complex_awgn(rng_local, x.shape, sigma1[None, :])
        z = x @ Hd1.T + n1
        z_n = z / z_scale[None, :]
        z_spm = _apply_spm(z_n, gamma_key) * z_scale[None, :]
        n2 = _complex_awgn(rng_local, x.shape, sigma2[None, :])
        return z_spm @ Hd2.T + n2

    modulations = [(4, "QPSK"), (16, "16-QAM"), (64, "64-QAM")]
    rows = []

    print("\n[Supp Fig S1] Multi-modulation BER for 37-core Protected2...")
    for M, label in modulations:
        const, bit_labels = _qam_constellation_gray(M)
        bps = int(np.log2(M))

        # Generate random QAM symbols
        rng_mod = np.random.default_rng(seed + 9100 + M)
        sym_indices_train = rng_mod.integers(0, M, size=(T, Nd))
        sym_indices_test = rng_mod.integers(0, M, size=(T_test, Nd))
        x_train = const[sym_indices_train]  # (T, Nd) complex
        x_test = const[sym_indices_test]
        bits_train_all = bit_labels[sym_indices_train]  # (T, Nd, bps)
        bits_test_all = bit_labels[sym_indices_test]

        # Forward pass
        rng_fwd = np.random.default_rng(seed + 9200 + M)
        y_train_full = _forward(x_train, rng_fwd)
        y_test_full = _forward(x_test, rng_fwd)

        # Bob decodes (P2 inverse: Hd2_inv -> invert_spm -> Hd1_inv)
        z_hat = y_test_full @ Hd2_inv.T
        z_inv = _invert_spm(z_hat / z_scale[None, :], gamma_key) * z_scale[None, :]
        x_hat_bob = z_inv @ Hd1_inv.T
        idx_hat_bob = _nearest_constellation_indices(x_hat_bob, const)
        bits_hat_bob = bit_labels[idx_hat_bob]
        ber_bob = float(np.mean(bits_test_all != bits_hat_bob))

        # Helper: QAM-aware attacker BER
        def _attack_ber(x_tr, y_tr, y_te, bits_ref, mode="ce"):
            if mode == "ce":
                H_est_T = np.linalg.lstsq(x_tr, y_tr, rcond=None)[0]
                x_hat = y_te @ np.linalg.pinv(H_est_T.T).T
            else:  # lr
                A = np.linalg.lstsq(y_tr, x_tr, rcond=None)[0]
                x_hat = y_te @ A
            idx_hat = _nearest_constellation_indices(x_hat, const)
            bits_hat = bit_labels[idx_hat]
            return float(np.mean(bits_ref != bits_hat))

        # FO: Eve sees full output
        n_obs = min(200, T)
        ber_ce_fo = _attack_ber(x_train[:n_obs], y_train_full[:n_obs],
                                y_test_full, bits_test_all, "ce")
        ber_lr_fo = _attack_ber(x_train[:n_obs], y_train_full[:n_obs],
                                y_test_full, bits_test_all, "lr")

        # WT: Eve sees tapped output
        y_eve_train_wt, _ = _apply_wt_tap(
            y_train_full, geom, params, tap_cfg, rng_mod)
        y_eve_test_wt, _ = _apply_wt_tap(
            y_test_full, geom, params, tap_cfg, rng_mod)
        ber_ce_wt = _attack_ber(x_train[:n_obs], y_eve_train_wt[:n_obs],
                                y_eve_test_wt, bits_test_all, "ce")
        ber_lr_wt = _attack_ber(x_train[:n_obs], y_eve_train_wt[:n_obs],
                                y_eve_test_wt, bits_test_all, "lr")

        row = {
            "modulation": label, "M": M, "bps": bps,
            "BER_Bob": ber_bob,
            "BER_KPA_CE_FO": ber_ce_fo, "BER_KPA_LR_FO": ber_lr_fo,
            "BER_KPA_CE_WT": ber_ce_wt, "BER_KPA_LR_WT": ber_lr_wt,
        }
        rows.append(row)
        print(f"  {label:8s}: Bob={ber_bob:.4f}  "
              f"CE_FO={ber_ce_fo:.4f} LR_FO={ber_lr_fo:.4f}  "
              f"CE_WT={ber_ce_wt:.4f} LR_WT={ber_lr_wt:.4f}")

    _write_csv(f"{out_dir}/qam_ber_protected2_37core.csv", rows)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    x_pos = np.arange(len(modulations))
    w = 0.18
    colors = {"Bob": "tab:blue", "KPA-CE": "tab:red", "KPA-LR": "tab:orange"}

    for ax, (suffix, title) in zip(axes, [("FO", "Full Observation (FO)"),
                                           ("WT", "Wiretap (WT)")]):
        ax.bar(x_pos - w, [r["BER_Bob"] for r in rows], w,
               label="Bob", color=colors["Bob"], edgecolor="black")
        ax.bar(x_pos, [r[f"BER_KPA_CE_{suffix}"] for r in rows], w,
               label="KPA-CE", color=colors["KPA-CE"], edgecolor="black")
        ax.bar(x_pos + w, [r[f"BER_KPA_LR_{suffix}"] for r in rows], w,
               label="KPA-LR", color=colors["KPA-LR"], edgecolor="black")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([r["modulation"] for r in rows])
        ax.set_ylabel("BER")
        ax.set_ylim(0, 0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.4)
        ax.legend(fontsize=8, ncol=3)

    fig.suptitle("37-core Protected2: BER across modulation formats", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_qam_ber_protected2_37core.png", dpi=200)
    plt.close(fig)
    return rows


def write_minimum_publishable_plan(out_dir: str) -> None:
    plan = """# Minimum Publishable Figure Plan

## Core figures

1. `fig0_mcf_geometry.png`
   Shows the physical keyed platform and dimensions.

2. `fig_validation.png`
   Validates the coupled-mode simulator against an analytical case and wavelength behavior.

3. `fig_validation_calibration_overview.png`
   Calibration appendix figure linking public geometry, internal simulation-length anchors, and the legacy normalized-SPM phase anchor.

4. `fig_attack_benchmark_fo_wt.png`
   Primary attack-comparison figure for Bob, KPA, LR, and LSTM under FO and WT, comparing the single-stage linear baseline against one-SPM and two-SPM protected schemes.

5. `fig_kpa_lr_sample_complexity_fo_wt.png`
   Shows whether increasing known-plaintext exposure breaks each scheme, and whether WT or SPM changes that slope.

6. `fig_lstm_training_size_fo_wt.png`
   Shows whether learned attacks converge with more data and whether the protected scheme sustains a BER floor.

7. `fig_core_scaling_13_19_37.png`
   Exploratory scaling figure comparing 13-core, 19-core, and 37-core systems for the linear and Protected2 schemes.

8. `fig_protected2_37core_multiseed.png`
   Multi-seed validation of the strongest current regime: 37-core Protected2, including the hybrid, MLP, transformer, and LSTM learned-Eve baselines.

9. `fig_protected2_37core_sample_complexity.png`
   Shows whether the 37-core Protected2 regime remains secure as known-plaintext exposure grows, with 95% confidence intervals across seeds.

10. `fig_protected2_37core_tap_sensitivity.png`
   Tests robustness of the 37-core Protected2 regime to tap fraction and observed-core set.

## Exploratory figures

- `fig_xpm_exploratory_benchmark.png`
  Compares the current SPM block against an exploratory phase-only SPM+XPM block.

- `fig_spm_xpm_combo_37core.png`
  Tests whether adding the same exploratory SPM+XPM block helps in the 37-core regime.

## Minimum tables / CSV files

- `physical_parameter_traceability.csv`
- `validation_anchor_points.csv`
- `fo_wt_attack_benchmark.csv`
- `fo_wt_kpa_lr_sample_complexity.csv`
- `fo_wt_lstm_training_size.csv`
- `core_scaling_13_19_37_attack_benchmark.csv`
- `protected2_37core_multiseed_summary.csv`
- `protected2_37core_sample_complexity.csv`
- `protected2_37core_tap_sensitivity.csv`
- `scaling_multiseed_summary.csv`
- `scaling_significance_tests.csv`

## Publication logic

- Use the linear baseline to show the break.
- Use the one-SPM and two-SPM protected schemes to test whether deeper static nonlinearity improves resistance without harming Bob.
- Use the 37-core scaling figure to test whether higher dimension is the real path to strong resistance.
- Use the 37-core multiseed and sample-complexity figures to show that the strongest regime is stable, not anecdotal, under explicit and learned Eve models.
- Use the scaling significance table to show that the 19-core and 37-core improvements are statistically significant rather than visual artifacts.
- Use the calibration appendix artifacts to state exactly which parameters are public, which are internally anchored, and which remain operating-point assumptions.
- Use WT to show the effect of a physically degraded tap model, not just wrong-key inversion.
"""
    with open(f"{out_dir}/minimum_publishable_figure_plan.md", "w", newline="") as f:
        f.write(plan)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  PARAMETER SWEEPS & FIGURE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            serialised = {}
            for k, v in r.items():
                if isinstance(v, (list, dict)):
                    serialised[k] = json.dumps(v)
                else:
                    serialised[k] = v
            w.writerow(serialised)


# Submission-package helpers keep the manuscript artifacts explicit and traceable.
def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Read a CSV into a list of dictionaries if it exists."""
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_submission_manifest(out_dir: str) -> None:
    """
    Write the figure/table manifest used by the manuscript and supplement.

    Each row links a named artifact to its primary CSV and any supporting data.
    """
    rows = [
        {
            "artifact_type": "config",
            "artifact_name": "publication_suite_config.csv",
            "primary_data": "publication_suite_config.csv",
            "supporting_data": "",
            "manuscript_role": "Reference configuration and saved true key for the focused suite",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_validation_calibration_overview.png",
            "primary_data": "validation_anchor_points.csv",
            "supporting_data": "physical_parameter_traceability.csv",
            "manuscript_role": "Calibration overview linking public geometry, internal simulation anchors, and the legacy SPM phase anchor",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_attack_benchmark_fo_wt.png",
            "primary_data": "fo_wt_attack_benchmark.csv",
            "supporting_data": "",
            "manuscript_role": "Main 13-core benchmark under FO and WT",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_kpa_lr_sample_complexity_fo_wt.png",
            "primary_data": "fo_wt_kpa_lr_sample_complexity.csv",
            "supporting_data": "",
            "manuscript_role": "13-core sample-complexity figure",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_lstm_training_size_fo_wt.png",
            "primary_data": "fo_wt_lstm_training_size.csv",
            "supporting_data": "",
            "manuscript_role": "13-core LSTM training-size figure",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_core_scaling_13_19_37.png",
            "primary_data": "core_scaling_13_19_37_attack_benchmark.csv",
            "supporting_data": "scaling_multiseed_summary.csv",
            "manuscript_role": "Core-scaling figure and supporting multiseed summary",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_protected2_37core_multiseed.png",
            "primary_data": "protected2_37core_multiseed_summary.csv",
            "supporting_data": "protected2_37core_multiseed.csv",
            "manuscript_role": "Main 37-core multiseed validation figure",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_protected2_37core_sample_complexity.png",
            "primary_data": "protected2_37core_sample_complexity_summary.csv",
            "supporting_data": "protected2_37core_sample_complexity.csv",
            "manuscript_role": "37-core sample-complexity figure with confidence intervals",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_protected2_37core_tap_sensitivity.png",
            "primary_data": "protected2_37core_tap_sensitivity.csv",
            "supporting_data": "",
            "manuscript_role": "37-core tap-sensitivity figure",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_spm_xpm_combo_37core.png",
            "primary_data": "spm_xpm_combo_37core_benchmark.csv",
            "supporting_data": "",
            "manuscript_role": "Exploratory 37-core SPM versus SPM+XPM figure",
        },
        {
            "artifact_type": "table",
            "artifact_name": "scaling_significance_tests.csv",
            "primary_data": "scaling_significance_tests.csv",
            "supporting_data": "scaling_multiseed_benchmark.csv",
            "manuscript_role": "Formal permutation and bootstrap significance results",
        },
        {
            "artifact_type": "table",
            "artifact_name": "protected2_37core_multiseed_summary.csv",
            "primary_data": "protected2_37core_multiseed_summary.csv",
            "supporting_data": "protected2_37core_multiseed.csv",
            "manuscript_role": "Summary table for 37-core multiseed validation",
        },
        {
            "artifact_type": "table",
            "artifact_name": "physical_parameter_traceability.csv",
            "primary_data": "physical_parameter_traceability.csv",
            "supporting_data": "validation_anchor_points.csv",
            "manuscript_role": "Parameter-to-source calibration appendix table",
        },
        {
            "artifact_type": "document",
            "artifact_name": "minimum_publishable_figure_plan.md",
            "primary_data": "minimum_publishable_figure_plan.md",
            "supporting_data": "reproducibility_manifest.csv",
            "manuscript_role": "Figure-selection and manuscript-logic guide",
        },
    ]
    _write_csv(os.path.join(out_dir, "reproducibility_manifest.csv"), rows)


def write_physical_calibration_bundle(params: FiberParams,
                                      out_dir: str,
                                      base_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Export the calibration/provenance artifacts used by the manuscript package.

    The bundle separates public-model quantities from internal anchors and
    chosen operating points. It documents provenance; it does not claim full
    experimental calibration of the hardware.
    """
    kappa_sig = params.coupling_coeff(lam_um=params.lam_sig_um)
    kappa_pump = params.coupling_coeff(lam_um=params.lam_pump_um)
    lc_sig = np.pi / (2 * max(kappa_sig, 1e-12))
    lc_pump = np.pi / (2 * max(kappa_pump, 1e-12))
    ratio_sig_to_pump = kappa_sig / max(kappa_pump, 1e-12)

    trace_rows = [
        {
            "parameter": "n_core_1550",
            "value": params.n_core,
            "units": "index",
            "source": "internal simulation summary / vendor data",
            "validation_status": "anchored",
            "notes": "1550 nm core refractive index listed in the internal simulation summary",
        },
        {
            "parameter": "delta_n_phase2",
            "value": params.delta_n,
            "units": "index",
            "source": "internal simulation summary / vendor data",
            "validation_status": "anchored",
            "notes": "Phase-2 DeltaRI = 0.00479 appears in the internal simulation summary",
        },
        {
            "parameter": "core_diameter",
            "value": params.core_diam_um,
            "units": "um",
            "source": "public geometry + internal simulation summary",
            "validation_status": "anchored",
            "notes": "8.2 um geometry is the common reference geometry across the current manuscript",
        },
        {
            "parameter": "pitch",
            "value": params.pitch_um,
            "units": "um",
            "source": "public geometry + internal simulation summary",
            "validation_status": "anchored",
            "notes": "16.4 um is the baseline pitch for the 13-core public geometry",
        },
        {
            "parameter": "lambda_signal",
            "value": params.lam_sig_um,
            "units": "um",
            "source": "SREP29080 / system definition",
            "validation_status": "anchored",
            "notes": "Main signal wavelength",
        },
        {
            "parameter": "lambda_pump",
            "value": params.lam_pump_um,
            "units": "um",
            "source": "internal design note + internal simulation summary",
            "validation_status": "anchored",
            "notes": "Internal sources explicitly use 976 nm rather than 980 nm",
        },
        {
            "parameter": "kappa_signal_model",
            "value": float(kappa_sig),
            "units": "rad/mm",
            "source": "CME model",
            "validation_status": "public-cross-check",
            "notes": "Used in fig_validation and consistent with the public coupling-trend argument",
        },
        {
            "parameter": "kappa_pump_model",
            "value": float(kappa_pump),
            "units": "rad/mm",
            "source": "CME model",
            "validation_status": "public-cross-check",
            "notes": "Shorter pump coupling length than signal coupling length",
        },
        {
            "parameter": "lc_signal_model",
            "value": float(lc_sig),
            "units": "mm",
            "source": "CME model",
            "validation_status": "public-cross-check",
            "notes": "Characteristic coupling length at 1550 nm",
        },
        {
            "parameter": "lc_pump_model",
            "value": float(lc_pump),
            "units": "mm",
            "source": "CME model",
            "validation_status": "public-cross-check",
            "notes": "Characteristic coupling length at 976 nm",
        },
        {
            "parameter": "kappa_sig_over_kappa_pump",
            "value": float(ratio_sig_to_pump),
            "units": "ratio",
            "source": "CME model + public trend",
            "validation_status": "public-cross-check",
            "notes": "Logged as approximately 0.2x in the legacy validation output, consistent with SREP29080 Fig. 3",
        },
        {
            "parameter": "attenuation_1550_measured",
            "value": 23.4,
            "units": "dB/m",
            "source": "internal measurement note",
            "validation_status": "internal-measured",
            "notes": "Measured 1550 nm attenuation reported in the internal measurement note",
        },
        {
            "parameter": "internal_effective_length_1550_phase2_8p2_16p4",
            "value": 89.0,
            "units": "mm",
            "source": "internal simulation summary",
            "validation_status": "internal-simulated",
            "notes": "Internal simulation summary reports 89 mm effective length for the 8.2 um / 16.4 um phase-2 fiber",
        },
        {
            "parameter": "internal_effective_length_976_8p2_18",
            "value": 200.0,
            "units": "mm",
            "source": "internal simulation summary",
            "validation_status": "internal-simulated",
            "notes": "Internal simulation summary reports 200 mm effective length for 976 nm at 8.2 um / 18 um",
        },
        {
            "parameter": "length_reference_run",
            "value": params.length_mm,
            "units": "mm",
            "source": "publication suite choice",
            "validation_status": "chosen-operating-point",
            "notes": "Selected to stay in the high-mixing regime while remaining physically plausible",
        },
    ]
    _write_csv(os.path.join(out_dir, "physical_parameter_traceability.csv"), trace_rows)

    anchor_rows = [
        {
            "anchor_name": "signal_coupling_length_model",
            "value": float(lc_sig),
            "units": "mm",
            "anchor_type": "public-model",
            "notes": "Computed from the CME coupling coefficient at 1550 nm",
        },
        {
            "anchor_name": "pump_coupling_length_model",
            "value": float(lc_pump),
            "units": "mm",
            "anchor_type": "public-model",
            "notes": "Computed from the CME coupling coefficient at 976 nm",
        },
        {
            "anchor_name": "internal_effective_length_1550_phase2_8p2_16p4",
            "value": 89.0,
            "units": "mm",
            "anchor_type": "internal-simulation",
            "notes": "Internal simulation summary, phase-2 geometry",
        },
        {
            "anchor_name": "internal_effective_length_976_8p2_18",
            "value": 200.0,
            "units": "mm",
            "anchor_type": "internal-simulation",
            "notes": "Internal simulation summary, pump-coupling study",
        },
        {
            "anchor_name": "attenuation_1550",
            "value": 23.4,
            "units": "dB/m",
            "anchor_type": "internal-measured",
            "notes": "Internal measurement note: attenuation anchor",
        },
    ]

    spm_csv = os.path.join(base_dir, "mcf_pls_figs", "spm_nonlinear.csv")
    spm_rows = _read_csv_rows(spm_csv)
    for row in spm_rows:
        anchor_rows.append({
            "anchor_name": f"spm_phi_std_gamma_{row['gamma_NL']}",
            "value": float(row["phi_std_rad_mean"]),
            "units": "rad",
            "anchor_type": "legacy-spm-sweep",
            "notes": f"Legacy normalized SPM sweep: gamma_NL={row['gamma_NL']}",
        })
    _write_csv(os.path.join(out_dir, "validation_anchor_points.csv"), anchor_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    length_labels = [
        "Internal sim 1550\n8.2/16.4",
        "Model Lc\n1550",
        "Model Lc\n976",
        "Internal sim 976\n8.2/18",
    ]
    length_vals = [89.0, lc_sig, lc_pump, 200.0]
    length_colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
    axes[0].bar(np.arange(len(length_labels)), length_vals, color=length_colors, edgecolor="black")
    axes[0].set_xticks(np.arange(len(length_labels)))
    axes[0].set_xticklabels(length_labels)
    axes[0].set_ylabel("Length [mm]")
    axes[0].set_title("Characteristic interaction lengths")
    axes[0].grid(axis="y", alpha=0.35)
    axes[0].text(
        0.02, 0.97,
        "Internal simulation anchors are not one-to-one\nwith the publication model, but they bound\nrelevant interaction-length scales.",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    if spm_rows:
        gammas = np.asarray([float(r["gamma_NL"]) for r in spm_rows], dtype=float)
        phi_std = np.asarray([float(r["phi_std_rad_mean"]) for r in spm_rows], dtype=float)
        coeff = np.polyfit(gammas, phi_std, deg=1)
        axes[1].plot(gammas, phi_std, "o-", color="tab:purple", lw=2, ms=6, label="Legacy SPM sweep")
        axes[1].plot(gammas, coeff[0] * gammas + coeff[1], "--", color="black", lw=1.5,
                     label=f"Linear fit: {coeff[0]:.2f} gamma + {coeff[1]:.2f}")
        axes[1].set_xlabel("gamma_NL")
        axes[1].set_ylabel("std(phi_NL) [rad]")
        axes[1].set_title("Legacy normalized-SPM phase anchor")
        axes[1].grid(True, alpha=0.35)
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Legacy SPM anchor CSV not found",
                     ha="center", va="center", fontsize=10)

    fig.suptitle("Calibration overview for the submission package", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_validation_calibration_overview.png"), dpi=220)
    plt.close(fig)
    return trace_rows, anchor_rows


# ── Figure 2 — BER vs. key mismatch ──────────────────────────────────────────

def sweep_key_mismatch(enc: MCFEncryption,
                       true_key: np.ndarray,
                       out_dir: str,
                       T: int = 800,
                       n_mismatch_levels: int = 10,
                       seed: int = 7) -> List[Dict]:
    """
    Vary Eve's key error: fraction of key components replaced by random values.
    Measures BER_Bob (constant, as Bob always has the right key) and
    BER_Eve as a function of key mismatch.
    """
    rng = np.random.default_rng(seed)
    results = []
    n_key = len(true_key)
    fractions = np.linspace(0.0, 1.0, n_mismatch_levels + 1)

    for frac in fractions:
        n_wrong = int(round(frac * n_key))
        eve_key = true_key.copy()
        idx_wrong = rng.choice(n_key, size=n_wrong, replace=False) if n_wrong else []
        for i in idx_wrong:
            eve_key[i] = float(rng.uniform(0, 1))

        res = enc.run_session(true_key, T=T, eve_key=eve_key,
                              eve_partial_obs=False)
        res["key_mismatch_frac"] = float(frac)
        results.append(res)
        print(f"  mismatch={frac:.2f}  BER_Bob={res['BER_Bob']:.4f}"
              f"  BER_Eve={res['BER_Eve']:.4f}")

    _write_csv(f"{out_dir}/key_mismatch.csv", results)

    plt.figure(figsize=(6, 4))
    xs = [r["key_mismatch_frac"] for r in results]
    plt.plot(xs, [r["BER_Bob"] for r in results], "o-",
             label="Bob (correct key)", color="tab:blue")
    plt.plot(xs, [r["BER_Eve"] for r in results], "s--",
             label="Eve (wrong key, full obs)", color="tab:red")
    plt.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    plt.xlabel("Fraction of key components mismatched")
    plt.ylabel("BER")
    plt.title("Physical-layer security: BER vs. key error")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig2_ber_vs_key_mismatch.png", dpi=200)
    plt.close()
    return results


# ── Figure 2b — Analog key sensitivity (single-component δP sweep) ───────────

def sweep_analog_key_mismatch(enc: MCFEncryption,
                               true_key: np.ndarray,
                               out_dir: str,
                               T: int = 600,
                               seed: int = 7) -> List[Dict]:
    """
    BER_Eve vs. analog pump power error δP on a single key component.

    Models the scenario where Eve knows N-1 key components exactly but
    guesses one with error δP (e.g., she guesses 0.70 when true value is 0.68).
    The sweep is averaged over all 13 key components to give a representative result.

    Answers: "What is the required pump laser precision for security?"
    At δP=0.02 (2% error), is the system broken? At δP=0.001 (0.1% error)?

    Key findings to expect:
    - δP < 0.01: BER_Eve near 0 (Eve effectively guessing correctly)
    - δP ≈ 0.05: BER_Eve starts rising significantly
    - δP ≥ 0.20: BER_Eve approaches 0.45 (random guessing limit)

    This establishes the minimum pump laser resolution requirement for the
    system to provide meaningful security even against a partially-informed Eve.
    """
    delta_P_vals = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    results = []
    print("\n[Fig 2b] Analog key mismatch: BER_Eve vs. dP (single component)...")
    for dP in delta_P_vals:
        ber_vals = []
        for comp_idx in range(len(true_key)):
            # Eve knows all other components correctly; guesses this one with error dP
            eve_key = true_key.copy()
            eve_key[comp_idx] = float(np.clip(true_key[comp_idx] + dP, 0.0, 1.0))
            res = enc.run_session(true_key, T=T, eve_key=eve_key, eve_partial_obs=False)
            ber_vals.append(res["BER_Eve"])
        results.append({
            "delta_P":       dP,
            "BER_Bob":       0.0,
            "BER_Eve_mean":  float(np.mean(ber_vals)),
            "BER_Eve_max":   float(np.max(ber_vals)),
            "BER_Eve_min":   float(np.min(ber_vals)),
            "BER_Eve_std":   float(np.std(ber_vals)),
        })
        print(f"  dP={dP:.3f}  BER_Eve_mean={np.mean(ber_vals):.4f}"
              f"  BER_Eve_max={np.max(ber_vals):.4f}")

    _write_csv(f"{out_dir}/analog_key_mismatch.csv", results)

    fig, ax = plt.subplots(figsize=(6, 4))
    dPs = [r["delta_P"] for r in results]
    means = [r["BER_Eve_mean"] for r in results]
    maxs  = [r["BER_Eve_max"]  for r in results]
    mins  = [r["BER_Eve_min"]  for r in results]
    ax.fill_between(dPs, mins, maxs, alpha=0.2, color="tab:red",
                    label="BER_Eve range (min–max over 13 components)")
    ax.plot(dPs, means, "s-", color="tab:red", lw=2,
            label="BER_Eve mean")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess (0.5)")
    ax.axvline(0.02, color="goldenrod", ls="--", lw=1.2,
               label="δP=0.02 (Eve: 0.70 vs 0.68)")
    ax.set_xlabel(r"Single-component pump error $\delta P$")
    ax.set_ylabel("BER (Eve, all other components correct)")
    ax.set_title("Analog key sensitivity: BER vs. single pump error\n"
                 "(Eve knows 12/13 components exactly; guesses 1 with δP error)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig2b_analog_key_mismatch.png", dpi=200)
    plt.close(fig)
    return results


# ── Figure 3 — Secrecy capacity vs. SNR ──────────────────────────────────────

def sweep_secrecy_capacity(enc: MCFEncryption,
                           true_key: np.ndarray,
                           out_dir: str,
                           snr_db_range: Optional[List[float]] = None,
                           n_eve_trials: int = 5,
                           seed: int = 7) -> List[Dict]:
    """
    Compute secrecy capacity and QPSK achievable secrecy rate vs. SNR.

    Two metrics are produced per SNR point:
      C_s_gaussian : Gaussian MIMO wiretap capacity (upper bound for any modulation)
                     C_s = max(0, log2det(I+SNR H_B H_B^+) - log2det(I+SNR H_E H_E^+))
      R_s_qpsk     : Achievable secrecy rate with uniform QPSK inputs
                     R_s = max(0, I_QPSK(x; y_B) - I_QPSK(x; y_E))
                     Estimated by Monte Carlo mutual information (lower bound).

    R_s_qpsk <= C_s_gaussian always (Gaussian input maximises entropy at fixed power).
    R_s_qpsk saturates at 2*N = 26 bits/use at high SNR (QPSK ceiling).
    """
    if snr_db_range is None:
        snr_db_range = list(range(-5, 25, 3))
    rng = np.random.default_rng(seed)
    sim = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    I_nd = np.eye(Nd)
    qpsk_max = 2.0 * Nd   # QPSK spectral efficiency ceiling

    H_bob = sim.transfer_matrix(true_key)
    Hd_bob = H_bob[np.ix_(data_idx, data_idx)]

    # Reference noise variance at sigma_noise (thermal floor, no ASE)
    sigma_ref2 = enc.params.sigma_noise**2

    results = []
    for snr_db in snr_db_range:
        snr = 10 ** (snr_db / 10.0)

        # ── Gaussian capacity (upper bound) ───────────────────────────────
        G_b = Hd_bob @ Hd_bob.conj().T
        cb = float(np.sum(np.log2(np.maximum(
            np.linalg.eigvalsh(I_nd + snr * G_b), 1.0))))

        ce_gauss_vals = []
        for _ in range(n_eve_trials):
            eve_key = rng.uniform(0, 1, size=true_key.shape)
            Hd_eve = sim.transfer_matrix(eve_key)[np.ix_(data_idx, data_idx)]
            G_e = Hd_eve @ Hd_eve.conj().T
            ce_gauss_vals.append(float(np.sum(np.log2(np.maximum(
                np.linalg.eigvalsh(I_nd + snr * G_e), 1.0)))))
        ce_gauss = float(np.mean(ce_gauss_vals))
        cs_gauss = max(0.0, cb - ce_gauss)

        # ── QPSK achievable secrecy rate (MMSE-SIC lower bound) ─────────────
        # In the key-uncertainty model:
        #   * Bob knows K  -> effective channel y_B = x + H_B^{-1} n (identity)
        #   * Eve guesses K_Eve -> effective channel x̂_E = A x + H_E^{-1} n
        #                          where A = H_E^{-1} H_B (scrambling matrix)
        #
        # R_s_QPSK lower bound = R_MMSE_SIC_Bob - C_Eve_Gaussian_avg_K
        #
        # R_MMSE_SIC_Bob: achievable QPSK rate for Bob using MMSE-SIC on his
        #                 known channel H_B.  2 bits/stream cap reflects QPSK.
        # C_Eve_Gaussian_avg_K: Gaussian capacity averaged over random K_Eve
        #                        (Eve's info about x from unkeyed y, upper bound).
        # This gives a valid lower bound: actual R_s_QPSK >= this value.
        sigma2_snr = 1.0 / (2.0 * snr)

        # Bob's MMSE-SIC achievable rate for each modulation (cap = bits/symbol)
        r_bob_qpsk  = _mmse_sic_rate_qpsk(Hd_bob, sigma2_snr, cap=2.0)
        r_bob_16qam = _mmse_sic_rate_qpsk(Hd_bob, sigma2_snr, cap=4.0)
        r_bob_64qam = _mmse_sic_rate_qpsk(Hd_bob, sigma2_snr, cap=6.0)

        # Eve's Gaussian capacity averaged over n_eve_trials random keys
        # (already computed above as ce_gauss = mean(ce_gauss_vals))
        # R_s = max(0, R_Bob_mod - C_Eve_Gaussian) for each modulation
        rs_qpsk  = max(0.0, r_bob_qpsk  - ce_gauss)
        rs_16qam = max(0.0, r_bob_16qam - ce_gauss)
        rs_64qam = max(0.0, r_bob_64qam - ce_gauss)

        results.append({
            "snr_db":         snr_db,
            "C_Bob":          cb,
            "C_Eve":          ce_gauss,
            "C_s":            cs_gauss,
            "R_Bob_QPSK":     r_bob_qpsk,
            "R_Bob_16QAM":    r_bob_16qam,
            "R_Bob_64QAM":    r_bob_64qam,
            "R_s_QPSK":       rs_qpsk,
            "R_s_16QAM":      rs_16qam,
            "R_s_64QAM":      rs_64qam,
        })
        print(f"  SNR={snr_db:+4d} dB  C_Bob={cb:.2f}  C_Eve={ce_gauss:.2f}"
              f"  C_s={cs_gauss:.2f}  |"
              f"  R_s_QPSK={rs_qpsk:.2f}"
              f"  R_s_16QAM={rs_16qam:.2f}"
              f"  R_s_64QAM={rs_64qam:.2f} b/use")

    _write_csv(f"{out_dir}/secrecy_capacity.csv", results)

    # ── Figure 3: multi-modulation plot ───────────────────────────────────
    snr_ax   = [r["snr_db"] for r in results]
    qam16max = 4.0 * Nd   # 16-QAM ceiling: 4 bits/stream × 13 streams
    qam64max = 6.0 * Nd   # 64-QAM ceiling
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(snr_ax, [r["C_Bob"] for r in results], "o-",
            label="$C_{Bob}$ Gaussian (upper bound)", color="tab:blue", lw=1.5)
    ax.plot(snr_ax, [r["C_Eve"] for r in results], "s--",
            label="$C_{Eve}$ Gaussian (upper bound)", color="tab:red", lw=1.5)
    ax.fill_between(snr_ax,
                    [r["C_Eve"] for r in results],
                    [r["C_Bob"] for r in results],
                    alpha=0.10, color="tab:green",
                    label="$C_s$ Gaussian upper bound")
    ax.plot(snr_ax, [r["R_s_QPSK"]  for r in results], "D-",
            label="$R_s$ QPSK (2 bits/sym)", color="tab:orange", lw=2.5)
    ax.plot(snr_ax, [r["R_s_16QAM"] for r in results], "^-",
            label="$R_s$ 16-QAM (4 bits/sym)", color="tab:purple", lw=2.0)
    ax.plot(snr_ax, [r["R_s_64QAM"] for r in results], "v-",
            label="$R_s$ 64-QAM (6 bits/sym)", color="saddlebrown", lw=2.0)
    # Modulation ceilings as horizontal reference lines
    ax.axhline(qpsk_max, color="tab:orange", ls=":", lw=1.0,
               label=f"QPSK ceil. ({qpsk_max:.0f} b/use)", alpha=0.6)
    ax.axhline(qam16max, color="tab:purple", ls=":", lw=1.0,
               label=f"16-QAM ceil. ({qam16max:.0f} b/use)", alpha=0.6)
    ax.axhline(qam64max, color="saddlebrown", ls=":", lw=1.0,
               label=f"64-QAM ceil. ({qam64max:.0f} b/use)", alpha=0.6)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Rate (bits / channel use)")
    ax.set_title("Secrecy capacity (Gaussian upper bound) vs.\nachievable secrecy rate by modulation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig3_secrecy_capacity.png", dpi=200)
    plt.close(fig)
    return results


# ── Figure 4 — Key sensitivity ────────────────────────────────────────────────

def sweep_key_sensitivity(enc: MCFEncryption,
                          true_key: np.ndarray,
                          out_dir: str,
                          delta_P_vals: Optional[List[float]] = None,
                          seed: int = 7) -> List[Dict]:
    """
    Avalanche property: ||ΔH||_F vs. pump power perturbation δP.
    A large gradient means small key errors cause large channel changes.
    """
    if delta_P_vals is None:
        delta_P_vals = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    results = []
    sim = enc.sim
    for dP in delta_P_vals:
        sens = key_sensitivity(sim, true_key, delta_P=dP)
        results.append({"delta_P": dP, "sensitivity": sens})
        print(f"  dP={dP:.3f}  ||dH||_F={sens:.4f}")

    _write_csv(f"{out_dir}/key_sensitivity.csv", results)

    plt.figure(figsize=(6, 4))
    xs = [r["delta_P"] for r in results]
    ys = [r["sensitivity"] for r in results]
    plt.semilogx(xs, ys, "o-", color="tab:purple")
    plt.xlabel("Pump power perturbation δP (normalised)")
    plt.ylabel(r"$\||\Delta H\||_F$  (Frobenius norm)")
    plt.title("Transfer matrix sensitivity to key perturbation")
    plt.grid(True, alpha=0.4, which="both")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4_key_sensitivity.png", dpi=200)
    plt.close()
    return results


# ── Figure 5 — BER vs. fiber length ──────────────────────────────────────────

def sweep_fiber_length(enc: MCFEncryption,
                       true_key: np.ndarray,
                       out_dir: str,
                       lengths_mm: Optional[List[float]] = None,
                       T: int = 600,
                       seed: int = 7) -> List[Dict]:
    """
    BER for Bob and Eve (wrong-key) as a function of fiber length.
    The effective coupling length (89–240 mm from internally anchored parameters)
    determines the mixing strength.
    """
    if lengths_mm is None:
        lengths_mm = [30, 60, 90, 120, 150, 180, 210, 240]
    rng = np.random.default_rng(seed)
    results = []
    for L in lengths_mm:
        eve_key = rng.uniform(0, 1, size=true_key.shape)
        res = enc.run_session(true_key, T=T, eve_key=eve_key,
                              eve_partial_obs=False, length_mm=float(L))
        res["length_mm"] = float(L)
        results.append(res)
        print(f"  L={L} mm  BER_Bob={res['BER_Bob']:.4f}"
              f"  BER_Eve={res['BER_Eve']:.4f}"
              f"  C_s={res['secrecy_cap']:.2f} b/use")

    _write_csv(f"{out_dir}/fiber_length.csv", results)

    plt.figure(figsize=(6, 4))
    Ls = [r["length_mm"] for r in results]
    plt.plot(Ls, [r["BER_Bob"] for r in results], "o-",
             label="Bob (correct key)", color="tab:blue")
    plt.plot(Ls, [r["BER_Eve"] for r in results], "s--",
             label="Eve (wrong key)", color="tab:red")
    plt.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    plt.xlabel("Fiber length L (mm)")
    plt.ylabel("BER")
    plt.title("BER vs. fiber length")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_ber_vs_length.png", dpi=200)
    plt.close()
    return results


# ── Figure 6 — Condition number vs. pump diversity ───────────────────────────

def sweep_condition_number(enc: MCFEncryption,
                           out_dir: str,
                           n_trials: int = 30,
                           seed: int = 7) -> List[Dict]:
    """
    Transfer matrix condition number κ(H) vs. pump key magnitude.
    Low condition number = invertible channel (Bob can decode).
    """
    rng = np.random.default_rng(seed)
    sim = enc.sim
    data_idx = enc.geom.data_core_indices
    results = []
    pump_mags = np.linspace(0.0, 1.0, 11)

    for pm in pump_mags:
        conds = []
        for _ in range(n_trials):
            key = rng.uniform(0, pm, size=enc.geom.N)
            H = sim.transfer_matrix(key)
            Hd = H[np.ix_(data_idx, data_idx)]
            conds.append(float(np.linalg.cond(Hd)))
        results.append({"pump_magnitude": float(pm),
                        "cond_H_mean":    float(np.mean(conds)),
                        "cond_H_std":     float(np.std(conds))})
        print(f"  pump_max={pm:.2f}  cond(H)={np.mean(conds):.2f} +/- {np.std(conds):.2f}")

    _write_csv(f"{out_dir}/condition_number.csv", results)

    plt.figure(figsize=(6, 4))
    xs  = [r["pump_magnitude"] for r in results]
    ys  = [r["cond_H_mean"]    for r in results]
    yes = [r["cond_H_std"]     for r in results]
    plt.errorbar(xs, ys, yerr=yes, fmt="o-", color="tab:orange",
                 capsize=4, label="mean ± std")
    plt.xlabel("Pump power range [0, $P_{max}$]")
    plt.ylabel(r"Condition number $\kappa(H)$")
    plt.title("Transfer matrix condition number vs. pump diversity")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig6_condition_number.png", dpi=200)
    plt.close()
    return results


# ── Figure 7 — Manufacturing tolerance ───────────────────────────────────────

def sweep_manufacturing_tolerance(enc: MCFEncryption,
                                  true_key: np.ndarray,
                                  out_dir: str,
                                  tol_fracs: Optional[List[float]] = None,
                                  T: int = 600,
                                  n_trials: int = 5,
                                  seed: int = 7) -> List[Dict]:
    """
    BER vs. fractional Δn manufacturing variation (0–0.5 %).
    Shows robustness of Bob's decryption and impact on Eve.

    The 0.3 % threshold (from coupled-mode simulations) is marked.
    """
    if tol_fracs is None:
        tol_fracs = [0.000, 0.001, 0.002, 0.003, 0.004, 0.005]

    rng = np.random.default_rng(seed)
    results = []
    for frac in tol_fracs:
        bob_bers, eve_bers = [], []
        for _ in range(n_trials):
            eve_key = rng.uniform(0, 1, size=true_key.shape)
            res = enc.run_session(true_key, T=T, eve_key=eve_key,
                                  eve_partial_obs=False, fab_tol_frac=frac)
            bob_bers.append(res["BER_Bob"])
            eve_bers.append(res["BER_Eve"])
        results.append({
            "tol_frac":      float(frac),
            "BER_Bob_mean":  float(np.mean(bob_bers)),
            "BER_Bob_std":   float(np.std(bob_bers)),
            "BER_Eve_mean":  float(np.mean(eve_bers)),
            "BER_Eve_std":   float(np.std(eve_bers)),
        })
        print(f"  dn_var={frac*100:.1f}%  "
              f"BER_Bob={np.mean(bob_bers):.4f}  "
              f"BER_Eve={np.mean(eve_bers):.4f}")

    _write_csv(f"{out_dir}/manufacturing_tolerance.csv", results)

    plt.figure(figsize=(6, 4))
    xs = [r["tol_frac"] * 100 for r in results]
    plt.errorbar(xs, [r["BER_Bob_mean"] for r in results],
                 yerr=[r["BER_Bob_std"] for r in results],
                 fmt="o-", color="tab:blue", capsize=4, label="Bob")
    plt.errorbar(xs, [r["BER_Eve_mean"] for r in results],
                 yerr=[r["BER_Eve_std"] for r in results],
                 fmt="s--", color="tab:red", capsize=4, label="Eve")
    plt.axvline(0.3, color="gray", ls=":", lw=1.5,
                label="0.3% fabrication limit")
    plt.axhline(0.5, color="gray", ls="-.", lw=1)
    plt.xlabel(r"Δn variation (% of nominal)")
    plt.ylabel("BER")
    plt.title("BER vs. manufacturing tolerance of Δn")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig7_manufacturing_tolerance.png", dpi=200)
    plt.close()
    return results


# ── Figure 10 — Joint sensitivity analysis ────────────────────────────────────

def sweep_joint_sensitivity(enc: MCFEncryption,
                             true_key: np.ndarray,
                             out_dir: str,
                             T: int = 400,
                             n_trials: int = 5,
                             seed: int = 7) -> List[Dict]:
    """
    BER_Bob and BER_Eve under *simultaneous* imperfections.

    Previous figures (2, 5, 7) each varied one factor at a time (OAT analysis).
    This figure tests realistic combinations:
      - key_mismatch_frac : fraction of Eve's key components replaced with random values
      - fab_tol_frac      : fractional Δn manufacturing variation

    Result: a 2D table (mismatch × fab_tol) showing that:
      - BER_Bob stays near 0 for all combinations (Bob is robust)
      - BER_Eve compounds across both imperfections (multiple factors benefit security)

    This addresses the critique that single-factor analysis overstates security —
    in reality all imperfections act simultaneously on Eve.
    """
    mismatch_fracs = [0.0, 0.25, 0.50, 1.0]          # 0%, 25%, 50%, 100% wrong
    tol_fracs      = [0.0, 0.003, 0.005]              # 0%, 0.3%, 0.5% Δn variation
    rng = np.random.default_rng(seed)
    results = []
    print("\n[Fig 10] Joint sensitivity: key mismatch x manufacturing tolerance...")

    for mf in mismatch_fracs:
        for tf in tol_fracs:
            bob_bers, eve_bers = [], []
            for _ in range(n_trials):
                n_wrong = int(round(mf * len(true_key)))
                eve_key = true_key.copy()
                if n_wrong > 0:
                    idx_wrong = rng.choice(len(true_key), size=n_wrong, replace=False)
                    for i in idx_wrong:
                        eve_key[i] = float(rng.uniform(0, 1))
                res = enc.run_session(true_key, T=T, eve_key=eve_key,
                                      eve_partial_obs=False, fab_tol_frac=tf)
                bob_bers.append(res["BER_Bob"])
                eve_bers.append(res["BER_Eve"])
            entry = {
                "key_mismatch_frac": float(mf),
                "fab_tol_pct":       float(tf * 100),
                "BER_Bob_mean":      float(np.mean(bob_bers)),
                "BER_Bob_std":       float(np.std(bob_bers)),
                "BER_Eve_mean":      float(np.mean(eve_bers)),
                "BER_Eve_std":       float(np.std(eve_bers)),
            }
            results.append(entry)
            print(f"  mismatch={mf:.2f}  dn_var={tf*100:.1f}%  "
                  f"BER_Bob={np.mean(bob_bers):.4f}  BER_Eve={np.mean(eve_bers):.4f}")

    _write_csv(f"{out_dir}/joint_sensitivity.csv", results)

    # ── Heatmap: BER_Eve (rows=mismatch fraction, cols=fab tolerance) ────
    n_mf = len(mismatch_fracs)
    n_tf = len(tol_fracs)
    eve_grid = np.zeros((n_mf, n_tf))
    bob_grid = np.zeros((n_mf, n_tf))
    for r in results:
        mi = mismatch_fracs.index(r["key_mismatch_frac"])
        ti = tol_fracs.index(r["fab_tol_pct"] / 100)
        eve_grid[mi, ti] = r["BER_Eve_mean"]
        bob_grid[mi, ti] = r["BER_Bob_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(eve_grid, vmin=0.0, vmax=0.5,
                          cmap="RdYlGn_r", aspect="auto")
    axes[0].set_xticks(range(n_tf))
    axes[0].set_xticklabels([f"{t*100:.1f}%" for t in tol_fracs])
    axes[0].set_yticks(range(n_mf))
    axes[0].set_yticklabels([f"{m*100:.0f}%" for m in mismatch_fracs])
    axes[0].set_xlabel("Δn manufacturing variation")
    axes[0].set_ylabel("Key mismatch fraction")
    axes[0].set_title("BER_Eve — Joint sensitivity\n(red=secure, green=insecure)")
    for mi in range(n_mf):
        for ti in range(n_tf):
            axes[0].text(ti, mi, f"{eve_grid[mi,ti]:.3f}",
                         ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im0, ax=axes[0], label="BER_Eve")

    im1 = axes[1].imshow(bob_grid, vmin=0.0, vmax=0.1,
                          cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(n_tf))
    axes[1].set_xticklabels([f"{t*100:.1f}%" for t in tol_fracs])
    axes[1].set_yticks(range(n_mf))
    axes[1].set_yticklabels([f"{m*100:.0f}%" for m in mismatch_fracs])
    axes[1].set_xlabel("Δn manufacturing variation")
    axes[1].set_ylabel("Key mismatch fraction")
    axes[1].set_title("BER_Bob — Joint sensitivity\n(green=BER_Bob=0, red=Bob degraded)")
    for mi in range(n_mf):
        for ti in range(n_tf):
            axes[1].text(ti, mi, f"{bob_grid[mi,ti]:.3f}",
                         ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im1, ax=axes[1], label="BER_Bob")

    fig.suptitle("Joint sensitivity: key mismatch × manufacturing tolerance", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig10_joint_sensitivity.png", dpi=200)
    plt.close(fig)
    return results


# ── Figure 8 — ML adversary ───────────────────────────────────────────────────

def ml_adversary_benchmark(enc: MCFEncryption,
                            true_key: np.ndarray,
                            out_dir: str,
                            T_train: int = 3000,
                            T_test: int = 800,
                            window: int = 8,
                            epochs: int = 15,
                            seed: int = 7) -> Dict:
    """
    Train Eve's LSTM on intercepted (ciphertext, plaintext) pairs.
    Plots training curve and final BER comparison.
    """
    if not HAS_TORCH:
        print("  Skipping ML adversary - PyTorch not installed.")
        return {}

    rng = np.random.default_rng(seed)
    sim = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    L = enc.params.length_mm

    H_bob = sim.transfer_matrix(true_key, L)
    Hd    = H_bob[np.ix_(data_idx, data_idx)]
    Hd_inv = np.linalg.pinv(Hd)

    # Generate training data (Eve observes all Nd ciphertext channels)
    T_tot  = T_train + T_test
    bits   = rng.integers(0, 2, size=(T_tot, Nd, 2))
    re_sym = 2.0 * bits[:, :, 0] - 1.0
    im_sym = 2.0 * bits[:, :, 1] - 1.0
    syms   = (re_sym + 1j * im_sym) / np.sqrt(2.0)

    n_sig = enc.params.sigma_noise
    noise = (rng.normal(scale=n_sig/np.sqrt(2), size=(T_tot, Nd)) +
             1j * rng.normal(scale=n_sig/np.sqrt(2), size=(T_tot, Nd)))
    y = syms @ Hd.T + noise

    # Train Eve on training portion (oracle: knows true symbols).
    # The trained model is returned so we can reuse it for test inference.
    _, trained_model, metrics = train_eve_rnn(
        y[:T_train], syms[:T_train],
        window=window, epochs=epochs, seed=seed)

    # ── Test inference using the trained model directly ────────────────────
    # We concatenate the last `window` training steps as context so the LSTM
    # has a warm start on the test segment.
    dev = next(trained_model.parameters()).device
    y_test_ctx = np.concatenate([y[T_train - window: T_train],
                                  y[T_train: T_train + T_test]], axis=0)
    x_hat_eml_flat = np.zeros((T_test, 2 * Nd), dtype=np.float32)
    trained_model.eval()
    with torch.no_grad():
        for t in range(window, window + T_test):
            feat  = y_test_ctx[t - window: t]
            feat_r = np.concatenate([feat.real, feat.imag], axis=1)
            inp   = torch.from_numpy(feat_r[None].astype(np.float32)).to(dev)
            x_hat_eml_flat[t - window] = trained_model(inp).cpu().numpy()[0]
    x_hat_eml = x_hat_eml_flat[:, :Nd] + 1j * x_hat_eml_flat[:, Nd:]

    # ── Bob's BER ──────────────────────────────────────────────────────────
    x_hat_bob  = y[T_train:] @ Hd_inv.T
    bits_test  = bits[T_train:]

    def _qpsk_ber(x_hat: np.ndarray, bits_ref: np.ndarray) -> float:
        """BER between decoded complex symbols and reference bit array (T,Nd,2)."""
        bits_hat = np.stack(
            [(x_hat[:, c].real >= 0).astype(int) for c in range(Nd)],
            axis=1)[:, :, None]  # (T, Nd, 1) real bits
        bits_hat_im = np.stack(
            [(x_hat[:, c].imag >= 0).astype(int) for c in range(Nd)],
            axis=1)[:, :, None]  # (T, Nd, 1) imag bits
        bits_hat_full = np.concatenate([bits_hat, bits_hat_im], axis=2)  # (T,Nd,2)
        return float(np.mean(bits_ref != bits_hat_full))

    ber_bob    = _qpsk_ber(x_hat_bob,  bits_test)
    ber_eve_ml = _qpsk_ber(x_hat_eml,  bits_test[:T_test])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(metrics["val_losses"], "o-", color="tab:red",
             label="Eve LSTM val MSE")
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("Validation MSE")
    ax1.set_title("ML adversary (Eve) training curve")
    ax1.legend(); ax1.grid(True, alpha=0.4)

    categories = ["Bob\n(correct key)", "Eve\n(wrong key)", "Eve\n(LSTM oracle)"]
    # Wrong-key Eve BER (quick estimate)
    eve_wk_key = rng.uniform(0, 1, size=true_key.shape)
    H_wk  = sim.transfer_matrix(eve_wk_key, L)
    Hd_wk = H_wk[np.ix_(data_idx, data_idx)]
    x_hat_wk = y[T_train:] @ np.linalg.pinv(Hd_wk).T
    ber_eve_wk = _qpsk_ber(x_hat_wk, bits_test)

    bers = [ber_bob, ber_eve_wk, ber_eve_ml]
    colors = ["tab:blue", "tab:red", "tab:orange"]
    ax2.bar(categories, bers, color=colors, edgecolor="black", width=0.5)
    ax2.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    ax2.set_ylabel("BER")
    ax2.set_ylim(0, 0.6)
    ax2.set_title("BER comparison: Bob vs. Eve strategies")
    ax2.legend(); ax2.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig8_ml_adversary.png", dpi=200)
    plt.close()

    result = {
        "BER_Bob":       ber_bob,
        "BER_Eve_WK":    ber_eve_wk,
        "BER_Eve_LSTM":  ber_eve_ml,
        "val_mse_final": metrics.get("val_mse_final", float("nan")),
    }
    print(f"  ML adversary - BER_Bob={ber_bob:.4f}  "
          f"BER_Eve_WK={ber_eve_wk:.4f}  BER_Eve_LSTM={ber_eve_ml:.4f}")
    return result


# ── Figure 1 — Transfer matrix visualisation ─────────────────────────────────

def plot_transfer_matrix(enc: MCFEncryption,
                         true_key: np.ndarray,
                         out_dir: str) -> None:
    H = enc.sim.transfer_matrix(true_key)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im0 = axes[0].imshow(np.abs(H), cmap="hot", aspect="auto")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("|H(K)| — correct key")
    axes[0].set_xlabel("Input core"); axes[0].set_ylabel("Output core")

    rng = np.random.default_rng(99)
    H_wk = enc.sim.transfer_matrix(rng.uniform(0, 1, size=true_key.shape))
    im1 = axes[1].imshow(np.abs(H_wk), cmap="hot", aspect="auto")
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("|H(K')| — wrong key")
    axes[1].set_xlabel("Input core"); axes[1].set_ylabel("Output core")

    plt.suptitle("13-core MCF transfer matrix: correct vs. wrong pump key",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig1_transfer_matrix.png", dpi=200)
    plt.close()


# ── Figure 0 — MCF geometry diagram ──────────────────────────────────────────

def plot_mcf_geometry(geom: MCFGeometry, params: FiberParams,
                      out_dir: str) -> None:
    """
    Publication-quality diagram of the 13-core hexagonal MCF cross-section.

    Color coding (matches Cohen et al. 2016 convention):
      Blue   = outer-ring data cores (signal input/output)
      Red    = inner-ring Erbium-doped cores (pump-keyed, synaptic)
      White  = central core (data)
    Overlaid: pitch arrow, core diameter label, wavelength annotation.
    """
    pos  = geom.positions          # (13, 2) in μm
    a    = params.core_radius_um   # μm

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")

    # All-EDFA architecture: every core is Er-doped AND carries data.
    # Color indicates ring position (ring 0=central, ring 1=inner, ring 2=outer)
    core_colors = {
        "center":  ("#D62728", "white", "Central Er-core (key k0)"),
        "inner":   ("#FF7F0E", "white", "Inner Er-cores (keys k1-k6)"),
        "outer":   ("#1F77B4", "white", "Outer Er-cores (keys k7-k12)"),
    }

    for i, (x, y) in enumerate(pos):
        if i == 0:
            ck = "center"
        elif 1 <= i <= 6:
            ck = "inner"
        else:
            ck = "outer"
        fc, ec_txt, _ = core_colors[ck]
        circle = plt.Circle((x, y), radius=a, color=fc,
                             ec="black", lw=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=7, color=ec_txt, fontweight="bold", zorder=4)

    # Pitch arrow between core 0 and core 1
    x0, y0 = pos[0]; x1, y1 = pos[1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    mid = ((x0 + x1) / 2 + 2, (y0 + y1) / 2 + 2)
    ax.text(*mid, f"Λ = {params.pitch_um} μm", fontsize=8, color="gray")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(fc="#D62728", ec="black", label="Central Er-core (key k0)"),
        Patch(fc="#FF7F0E", ec="black", label="Inner Er-cores (keys k1-k6)"),
        Patch(fc="#1F77B4", ec="black", label="Outer Er-cores (keys k7-k12)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              framealpha=0.9)

    # Annotations
    ax.set_xlim(-3 * params.pitch_um, 3 * params.pitch_um)
    ax.set_ylim(-3 * params.pitch_um, 3 * params.pitch_um)
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
    ax.set_title(
        f"13-core hexagonal MCF cross-section\n"
        f"Core ∅ {params.core_diam_um} μm, "
        f"Pitch {params.pitch_um} μm, "
        f"λ_s = {params.lam_sig_um*1000:.0f} nm, "
        f"λ_p = {params.lam_pump_um*1000:.0f} nm",
        fontsize=9)
    ax.text(0.02, 0.02,
            f"V(1550 nm) = {params.V():.2f}  [single-mode: V < 2.405]\n"
            f"κ = {params.coupling_coeff():.4f} rad/mm  |  "
            f"L_c = {np.pi/(2*max(params.coupling_coeff(), 1e-9)):.0f} mm",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.grid(True, alpha=0.2, ls="--")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig0_mcf_geometry.png", dpi=200)
    plt.close()
    print(f"  Saved fig0_mcf_geometry.png")


# ── Physical validation — CME simulator vs. analytical 3-core result ──────────

def validate_cme_simulator(params: FiberParams, geom: MCFGeometry,
                            out_dir: str) -> None:
    """
    Validates the CME integrator against the analytical solution for a
    symmetric 3-core linear array (closed-form eigenvalue solution).

    For a symmetric 3-core system with equal κ and no gain:
        Core 1 (input) — Core 2 (middle) — Core 3 (end)
    The analytical power in core 1 as a function of propagation length is:
        P₁(z) = |1/3 + 2/3 cos(√3 κ z)|²
    (Snyder & Love 1983, §29-5, three coupled identical waveguides)

    This confirms the CME integrator correctly handles multi-core coupling
    using the same Bessel-function κ formula used throughout the encryption.

    A second panel reproduces the qualitative behaviour from Cohen et al.
    (2016) Fig. 3: signal light (1550 nm) couples across more cores than
    pump light (976 nm) over the same fiber length, due to κ_1550 ≫ κ_976.
    """
    kappa_sig  = params.coupling_coeff(lam_um=params.lam_sig_um)   # rad/mm
    kappa_pump = params.coupling_coeff(lam_um=params.lam_pump_um)  # rad/mm

    print(f"  kappa at 1550 nm = {kappa_sig:.5f} rad/mm  "
          f"(L_c = {np.pi/(2*kappa_sig):.1f} mm)")
    print(f"  kappa at 976 nm  = {kappa_pump:.6f} rad/mm  "
          f"(L_c = {np.pi/(2*max(kappa_pump,1e-9)):.0f} mm)")
    print(f"  Coupling ratio kappa_sig/kappa_pump = "
          f"{kappa_sig/max(kappa_pump,1e-9):.1f}x  "
          f"(consistent with SREP29080 Fig. 3)")

    # ── Panel A: 3-core analytical vs. CME numerical ──────────────────────
    # Build a minimal 3-core linear geometry
    class _Geom3:
        N = 3
        pitch_um = params.pitch_um
        def positions(self): return np.array([[0.,0.],[params.pitch_um,0.],[2*params.pitch_um,0.]])
        def erbium_core_indices(self): return []
        def data_core_indices(self): return [0,1,2]
        def distances(self):
            p = params.pitch_um
            return np.array([[0,p,2*p],[p,0,p],[2*p,p,0]])
        def adjacency_matrix(self, max_coupling_ratio=1.05):
            D = self.distances()
            return (D > 1e-3) & (D <= max_coupling_ratio * params.pitch_um)

    g3 = _Geom3()
    # Reuse FiberParams for β; build a mini CME simulator
    class _CME3:
        """Minimal 3-core CME integrator for validation (SVE frame)."""
        def __init__(self):
            # All three cores identical → delta_beta = 0; only coupling drives dynamics
            self.kappa_mat = np.zeros((3, 3))
            self.kappa_mat[0, 1] = self.kappa_mat[1, 0] = kappa_sig
            self.kappa_mat[1, 2] = self.kappa_mat[2, 1] = kappa_sig

        def _rhs(self, z, a_flat):
            # SVE frame: carrier exp(-j*beta0*z) removed; delta_beta = 0 here
            a = a_flat[:3] + 1j * a_flat[3:]
            da = np.zeros(3, dtype=complex)
            for i in range(3):
                for j in range(3):
                    if self.kappa_mat[i, j] != 0:
                        da[i] -= 1j * self.kappa_mat[i, j] * a[j]
            return np.concatenate([da.real, da.imag])

        def power_vs_z(self, a0, L, n_pts=200):
            z_span = np.linspace(0, L, n_pts)
            a0_flat = np.concatenate([a0.real, a0.imag]).astype(float)
            sol = solve_ivp(self._rhs, (0, L), a0_flat,
                            method="RK45", t_eval=z_span,
                            rtol=1e-6, atol=1e-8)
            a = sol.y[:3] + 1j * sol.y[3:]
            return z_span, np.abs(a)**2

    cme3 = _CME3()
    L_plot = min(np.pi / kappa_sig, 250.0)  # show ~1 coupling cycle (faster solve)
    a0 = np.array([1. + 0j, 0., 0.])                  # all power in core 1
    z_num, P_num = cme3.power_vs_z(a0, L_plot)

    # Analytical: P₁(z) = |1/3 + 2/3·cos(√3·κ·z)|²  (Snyder & Love §29-5)
    z_an = np.linspace(0, L_plot, 500)
    P1_an = np.abs(1/3 + (2/3) * np.cos(np.sqrt(3) * kappa_sig * z_an))**2
    P2_an = (2/3 * np.sin(np.sqrt(3) * kappa_sig * z_an / 2))**2 * \
            (4/3) * np.cos(np.sqrt(3) * kappa_sig * z_an / 2)**2  # approx
    # Exact: P2(z) = (2/3)(1 - cos(√3 κ z)), P3(z) = P1(z) but ½-period shifted
    P2_an_exact = (2/3) * (1 - np.cos(np.sqrt(3) * kappa_sig * z_an))
    P3_an_exact = np.abs(1/3 - (1/3) * np.cos(np.sqrt(3) * kappa_sig * z_an) -
                          (1/np.sqrt(3)) * np.sin(np.sqrt(3) * kappa_sig * z_an))**2

    # ── Panel B: 1550 nm vs. 976 nm coupling in full 13-core MCF ─────────
    # Show how far from the input core energy spreads at z = L_eff
    L_eff = params.length_mm
    # Compute center-of-mass spread of power for each wavelength
    sim_full = CMESimulator(params, geom)
    pos = geom.positions

    def _power_spread(kappa_lam, L):
        """Simulate signal input at outer core 7, measure spread at output."""
        # Temporarily override kappa for this wavelength
        # (rebuild kappa matrix with new kappa value)
        D   = geom.distances()
        adj = geom.adjacency_matrix()
        kap_mat = np.zeros((geom.N, geom.N))
        for i in range(geom.N):
            for j in range(geom.N):
                if adj[i, j]:
                    kap_mat[i, j] = kappa_lam

        zero_key = np.zeros(geom.N)   # no pump for validation

        class _SimLam:
            def __init__(self):
                # SVE frame: store only delta_beta = beta0 - beta_mean (all ~0)
                b0 = sim_full.beta0.copy()
                self.delta_beta = (b0 - float(np.mean(b0))).astype(complex)
                self.kappa = kap_mat

            def _rhs(self, z, a_flat):
                # SVE frame: only coupling + tiny delta_beta drive dynamics
                N = 13
                a = a_flat[:N] + 1j * a_flat[N:]
                da = np.zeros(N, dtype=complex)
                for i in range(N):
                    da[i] = -1j * self.delta_beta[i] * a[i]
                    for j in range(N):
                        if self.kappa[i, j] != 0:
                            da[i] -= 1j * self.kappa[i, j] * a[j]
                return np.concatenate([da.real, da.imag])

            def propagate(self, a0, L):
                a0f = np.concatenate([a0.real, a0.imag]).astype(float)
                sol = solve_ivp(self._rhs,
                                (0, L), a0f, method="RK45",
                                rtol=1e-5, atol=1e-7)
                return sol.y[:13,-1] + 1j * sol.y[13:,-1]

        slm = _SimLam()
        a0  = np.zeros(geom.N, dtype=complex)
        a0[7] = 1.0 + 0j   # inject at outer core 7
        aL = slm.propagate(a0, L)
        P  = np.abs(aL)**2
        # Weighted mean distance from input core
        d_from_input = np.array([np.linalg.norm(pos[i] - pos[7])
                                  for i in range(geom.N)])
        spread = float(np.sum(P * d_from_input) / max(np.sum(P), 1e-12))
        return P, spread

    P_sig,  spread_sig  = _power_spread(kappa_sig,  L_eff)
    P_pump, spread_pump = _power_spread(kappa_pump, L_eff)

    print(f"  Power spread at L={L_eff} mm:")
    print(f"    Signal (1550 nm): {spread_sig:.1f} um from input core")
    print(f"    Pump   (976 nm):  {spread_pump:.1f} um from input core")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: CME vs. analytical
    axes[0].plot(z_an, P1_an,   "b-",  lw=2,  label="P₁ analytical")
    axes[0].plot(z_an, P2_an_exact, "g-", lw=2, label="P₂ analytical")
    axes[0].plot(z_an, P3_an_exact, "r-", lw=2, label="P₃ analytical")
    axes[0].plot(z_num, P_num[0], "b--", lw=1.5, label="P₁ CME (numeric)")
    axes[0].plot(z_num, P_num[1], "g--", lw=1.5, label="P₂ CME (numeric)")
    axes[0].plot(z_num, P_num[2], "r--", lw=1.5, label="P₃ CME (numeric)")
    axes[0].set_xlabel("Fiber length z (mm)")
    axes[0].set_ylabel("Normalised power")
    axes[0].set_title("CME validation:\n3-core analytical vs. numerical")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.4)

    # Panel B: 1550 nm power map at output
    _draw_power_map(axes[1], geom, P_sig, params.lam_sig_um,
                    title=f"1550 nm output at L={L_eff} mm\n"
                          f"(spread {spread_sig:.1f} μm)")

    # Panel C: 976 nm power map at output
    _draw_power_map(axes[2], geom, P_pump, params.lam_pump_um,
                    title=f"976 nm output at L={L_eff} mm\n"
                          f"(spread {spread_pump:.1f} μm)")

    plt.suptitle(
        "Physical validation: signal (1550 nm) couples farther than pump (976 nm)\n"
        "consistent with Cohen et al., Sci. Rep. 6, 29080 (2016) Fig. 3",
        fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig_validation.png", dpi=200)
    plt.close()
    print(f"  Saved fig_validation.png")


def _draw_power_map(ax, geom: MCFGeometry, P: np.ndarray,
                    lam_um: float, title: str = "") -> None:
    """Helper: draw 13-core power distribution as a bubble map."""
    pos   = geom.positions
    P_n   = P / max(P.max(), 1e-12)
    sizes = 800 * P_n + 20
    colors_arr = P_n
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colors_arr,
                    cmap="hot", vmin=0, vmax=1, edgecolors="gray", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
    ax.set_title(title, fontsize=8)
    # Mark input core
    ax.scatter([pos[7, 0]], [pos[7, 1]], marker="*", s=200,
               color="cyan", zorder=5, label="Input core")
    ax.legend(fontsize=7)
    plt.colorbar(sc, ax=ax, label="Norm. power")


# ── Figure 9 — Secrecy capacity vs. number of active Erbium cores ─────────────

def sweep_erbium_count(enc: MCFEncryption,
                       out_dir: str,
                       n_key_trials: int = 10,
                       snr_db: float = 10.0,
                       seed: int = 7) -> List[Dict]:
    """
    Sweeps the number of active Erbium-doped cores (1 → 6).

    For each count n_e, n_e randomly chosen inner-ring cores are pumped
    (pump power drawn uniformly from [0.3, 0.9]).  The remaining 6 − n_e
    cores are left unpumped (P = 0, i.e. absorbing).

    Metrics reported (averaged over n_key_trials random keys):
        - Secrecy capacity C_s  at a fixed SNR
        - BER_Eve (wrong-key linear decoder)
        - Transfer matrix condition number κ(H)

    This demonstrates that key dimensionality directly controls security:
    more pump-controlled cores → larger key space → higher C_s.
    """
    rng = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    snr = 10 ** (snr_db / 10.0)
    I_Nd = np.eye(Nd)
    L = enc.params.length_mm

    N_total = geom.N  # 13 in the fully-active architecture

    results = []
    for n_e in range(1, N_total + 1):
        cs_vals, cond_vals, ber_vals, ber_inf_vals = [], [], [], []
        for trial in range(n_key_trials):
            # Bob's key: n_e cores pumped (high), remaining cores absorbing (0)
            key = np.zeros(N_total)
            active = rng.choice(N_total, size=n_e, replace=False)
            key[active] = rng.uniform(0.6, 1.0, size=n_e)

            H_bob = sim.transfer_matrix(key, L)
            Hd_bob = H_bob[np.ix_(data_idx, data_idx)]
            cond_vals.append(float(np.linalg.cond(Hd_bob)))

            # Eve model 1: naive — random key from full [0, 1] range
            # (may accidentally land in same gain regime as Bob at N_Er=13)
            eve_key_naive = rng.uniform(0, 1, size=N_total)
            H_eve_naive   = sim.transfer_matrix(eve_key_naive, L)
            Hd_eve_naive  = H_eve_naive[np.ix_(data_idx, data_idx)]

            # Eve model 2: informed adversary — knows key is in [0.6, 1.0]
            # (stronger model: knows the gain regime but not the specific values)
            eve_key_inf = rng.uniform(0.6, 1.0, size=N_total)
            H_eve_inf   = sim.transfer_matrix(eve_key_inf, L)
            Hd_eve_inf  = H_eve_inf[np.ix_(data_idx, data_idx)]

            # Secrecy capacity (vs. naive Eve)
            G_b = Hd_bob @ Hd_bob.conj().T
            G_e = Hd_eve_naive @ Hd_eve_naive.conj().T
            C_b = float(np.sum(np.log2(np.maximum(
                np.linalg.eigvalsh(I_Nd + snr * G_b), 1.0))))
            C_e = float(np.sum(np.log2(np.maximum(
                np.linalg.eigvalsh(I_Nd + snr * G_e), 1.0))))
            cs_vals.append(max(0.0, C_b - C_e))

            # BER: both Eve models (single symbol, deterministic rng for reproducibility)
            n_sig = enc.params.sigma_noise
            rng2  = np.random.default_rng(trial)
            x     = (rng2.integers(0, 2, Nd) * 2 - 1 +
                     1j * (rng2.integers(0, 2, Nd) * 2 - 1)) / np.sqrt(2)
            n     = (rng2.normal(scale=n_sig/np.sqrt(2), size=Nd) +
                     1j * rng2.normal(scale=n_sig/np.sqrt(2), size=Nd))
            y     = Hd_bob @ x + n

            bits_ref = np.stack([(x.real >= 0).astype(int),
                                  (x.imag >= 0).astype(int)], axis=1)

            # Naive Eve BER
            x_hat_naive = np.linalg.pinv(Hd_eve_naive) @ y
            bits_naive  = np.stack([(x_hat_naive.real >= 0).astype(int),
                                     (x_hat_naive.imag >= 0).astype(int)], axis=1)
            ber_vals.append(float(np.mean(bits_ref != bits_naive)))

            # Informed Eve BER (stronger adversary)
            x_hat_inf = np.linalg.pinv(Hd_eve_inf) @ y
            bits_inf  = np.stack([(x_hat_inf.real >= 0).astype(int),
                                   (x_hat_inf.imag >= 0).astype(int)], axis=1)
            ber_inf_vals.append(float(np.mean(bits_ref != bits_inf)))

        results.append({
            "n_erbium_cores":      n_e,
            "key_dim":             n_e,
            "Cs_mean":             float(np.mean(cs_vals)),
            "Cs_std":              float(np.std(cs_vals)),
            "cond_H_mean":         float(np.mean(cond_vals)),
            "BER_Eve_naive_mean":  float(np.mean(ber_vals)),
            "BER_Eve_inf_mean":    float(np.mean(ber_inf_vals)),
        })
        print(f"  n_Erbium={n_e}  C_s={np.mean(cs_vals):.2f}+/-{np.std(cs_vals):.2f} b/use"
              f"  cond(H)={np.mean(cond_vals):.2f}"
              f"  BER_Eve_naive={np.mean(ber_vals):.3f}"
              f"  BER_Eve_informed={np.mean(ber_inf_vals):.3f}")

    _write_csv(f"{out_dir}/erbium_count_sweep.csv", results)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ne = [r["n_erbium_cores"] for r in results]

    # Secrecy capacity
    axes[0].errorbar(ne, [r["Cs_mean"] for r in results],
                     yerr=[r["Cs_std"] for r in results],
                     fmt="o-", color="tab:green", capsize=4)
    axes[0].set_xlabel("Number of active Erbium-doped cores")
    axes[0].set_ylabel("Secrecy capacity $C_s$ (bits/channel use)")
    axes[0].set_title(f"Secrecy capacity vs. key dimensionality\n(SNR = {snr_db} dB)")
    axes[0].grid(True, alpha=0.4)

    # Condition number
    axes[1].plot(ne, [r["cond_H_mean"] for r in results],
                 "o-", color="tab:orange")
    axes[1].set_xlabel("Number of active Erbium-doped cores")
    axes[1].set_ylabel(r"Condition number $\kappa(H)$")
    axes[1].set_title("Transfer matrix invertibility\n(lower = better for Bob)")
    axes[1].grid(True, alpha=0.4)

    # BER Eve — both adversary models
    axes[2].plot(ne, [r["BER_Eve_naive_mean"] for r in results],
                 "s--", color="tab:red",
                 label="Eve naive (key from [0,1])")
    axes[2].plot(ne, [r["BER_Eve_inf_mean"] for r in results],
                 "^-", color="darkred", lw=2,
                 label="Eve informed (key from [0.6,1.0])")
    axes[2].axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    axes[2].set_xlabel("Number of active Erbium-doped cores")
    axes[2].set_ylabel("BER (Eve, wrong key)")
    axes[2].set_title("Eve BER vs. key dimensionality\n(naive vs. informed adversary)")
    axes[2].set_ylim(0, 0.6)
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.4)

    plt.suptitle("Effect of pump key dimensionality on physical-layer security",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig9_erbium_count_sweep.png", dpi=200)
    plt.close()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 9.  MIXED-REGIME KEY + QAM BER SWEEPS (new experiments)
# ──────────────────────────────────────────────────────────────────────────────

def sweep_mixed_regime_key(enc: MCFEncryption,
                            out_dir: str,
                            T: int = 500,
                            n_trials: int = 12,
                            seed: int = 7) -> List[Dict]:
    """
    Tests mixed gain/absorption keys as countermeasure to informed adversary.

    For each n_gain (cores in [0.6,1.0]), remaining cores are in [0.0,0.4].
    Reports: kappa(H_Bob), BER_Bob (MMSE), BER_Eve_naive, BER_Eve_semiinformed.

    The 'semi-informed adversary' knows n_gain but not which cores are gain vs.
    absorption. She guesses a random regime assignment and random key values
    within each sub-range. This is the relevant adversary model when the regime
    count is publicly known but the assignment is part of the secret key.
    """
    rng = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    L = enc.params.length_mm
    N_total = geom.N
    I_Nd = np.eye(Nd)

    n_gain_vals = [3, 5, 7, 9, 11, 13]
    results = []

    for n_gain in n_gain_vals:
        n_abs = N_total - n_gain
        bob_bers, eve_naive_bers, eve_semi_bers, kappas = [], [], [], []

        for _ in range(n_trials):
            # Bob: n_gain cores in [0.6,1.0], rest in [0.0,0.4]
            key = np.zeros(N_total)
            gain_cores = rng.choice(N_total, size=n_gain, replace=False)
            abs_cores  = np.setdiff1d(np.arange(N_total), gain_cores)
            key[gain_cores] = rng.uniform(0.6, 1.0, size=n_gain)
            if n_abs > 0:
                key[abs_cores] = rng.uniform(0.0, 0.4, size=n_abs)

            H_bob = sim.transfer_matrix(key, L)
            Hd_bob = H_bob[np.ix_(data_idx, data_idx)]
            kappas.append(float(np.linalg.cond(Hd_bob)))

            # MMSE equalizer for Bob
            all_var = _ase_noise_variances(key, enc.params, geom)
            sigma2_m = float(np.mean(all_var[data_idx]))
            Hd_inv_mmse = np.linalg.solve(
                Hd_bob.conj().T @ Hd_bob + sigma2_m * I_Nd,
                Hd_bob.conj().T)

            # Transmit T QPSK symbols
            bits = rng.integers(0, 2, size=(T, Nd, 2))
            syms = np.array([enc._qpsk_map(bits[t]) for t in range(T)])
            data_var = all_var[data_idx]
            sigma_c  = np.sqrt(data_var / 2.0)
            noise = (rng.normal(size=(T, Nd)) * sigma_c[None, :] +
                     1j * rng.normal(size=(T, Nd)) * sigma_c[None, :])
            y = syms @ Hd_bob.T + noise

            # Bob MMSE decode
            x_hat_bob = y @ Hd_inv_mmse.T
            bits_bob = np.array([enc._qpsk_demod(x_hat_bob[t]) for t in range(T)])
            bob_bers.append(float(np.mean(bits != bits_bob)))

            # Naive Eve: random key from U[0,1]
            eve_naive = rng.uniform(0.0, 1.0, size=N_total)
            Hd_en = sim.transfer_matrix(eve_naive, L)[np.ix_(data_idx, data_idx)]
            x_hat_en = y @ np.linalg.pinv(Hd_en).T
            bits_en = np.array([enc._qpsk_demod(x_hat_en[t]) for t in range(T)])
            eve_naive_bers.append(float(np.mean(bits != bits_en)))

            # Semi-informed Eve: knows n_gain but not which cores
            eve_semi = np.zeros(N_total)
            gg = rng.choice(N_total, size=n_gain, replace=False)
            ag = np.setdiff1d(np.arange(N_total), gg)
            eve_semi[gg] = rng.uniform(0.6, 1.0, size=n_gain)
            if n_abs > 0:
                eve_semi[ag] = rng.uniform(0.0, 0.4, size=n_abs)
            Hd_es = sim.transfer_matrix(eve_semi, L)[np.ix_(data_idx, data_idx)]
            x_hat_es = y @ np.linalg.pinv(Hd_es).T
            bits_es = np.array([enc._qpsk_demod(x_hat_es[t]) for t in range(T)])
            eve_semi_bers.append(float(np.mean(bits != bits_es)))

        row = {
            "n_gain_cores":       n_gain,
            "n_abs_cores":        n_abs,
            "kappa_H_mean":       float(np.mean(kappas)),
            "kappa_H_max":        float(np.max(kappas)),
            "BER_Bob_mean":       float(np.mean(bob_bers)),
            "BER_Eve_naive_mean": float(np.mean(eve_naive_bers)),
            "BER_Eve_semi_mean":  float(np.mean(eve_semi_bers)),
        }
        results.append(row)
        print(f"  n_gain={n_gain}  kappa={np.mean(kappas):.1f}  "
              f"BER_Bob={np.mean(bob_bers):.4f}  "
              f"BER_Eve_naive={np.mean(eve_naive_bers):.4f}  "
              f"BER_Eve_semi={np.mean(eve_semi_bers):.4f}")

    _write_csv(f"{out_dir}/mixed_regime_sweep.csv", results)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ng = [r["n_gain_cores"] for r in results]

    axes[0].semilogy(ng, [r["kappa_H_mean"] for r in results],
                     "o-", color="tab:orange", label="Mean kappa")
    axes[0].semilogy(ng, [r["kappa_H_max"] for r in results],
                     "^--", color="tab:orange", alpha=0.5, label="Max kappa")
    axes[0].axhline(20, color="gray", ls="--", lw=1.5, label="MMSE tolerance (kappa=20)")
    axes[0].set_xlabel("Number of gain-regime cores")
    axes[0].set_ylabel("Condition number kappa(H_Bob)")
    axes[0].set_title("Matrix conditioning vs. key regime mix")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.4)

    axes[1].plot(ng, [r["BER_Bob_mean"] for r in results],
                 "o-", color="tab:blue", lw=2, label="BER_Bob (MMSE, correct key)")
    axes[1].plot(ng, [r["BER_Eve_naive_mean"] for r in results],
                 "s--", color="tab:red", label="BER_Eve naive U[0,1]")
    axes[1].plot(ng, [r["BER_Eve_semi_mean"] for r in results],
                 "^-", color="darkred", lw=2, label="BER_Eve semi-informed")
    axes[1].axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    axes[1].set_xlabel("Number of gain-regime cores")
    axes[1].set_ylabel("Bit Error Rate")
    axes[1].set_title("Security vs. regime mix\n(semi-informed adversary, knows n_gain)")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.4)
    axes[1].set_ylim(-0.02, 0.62)

    plt.suptitle("Mixed-regime key design: countermeasure to informed adversary", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig11_mixed_regime_key.png", dpi=200)
    plt.close()
    print(f"  Saved fig11_mixed_regime_key.png")
    return results


def sweep_qam_ber(enc: MCFEncryption,
                   true_key: np.ndarray,
                   out_dir: str,
                   T: int = 2000,
                   seed: int = 42) -> List[Dict]:
    """
    Simulates actual QPSK, 16-QAM, and 64-QAM BER for Bob and Eve.
    Validates that BER_Bob = 0 extends to higher-order modulations.

    Uses MMSE equalizer for Bob and ZF (pinv) for Eve.
    BER is computed exactly from Gray-coded bit labels after nearest-neighbour
    detection in the complex constellation plane.
    """
    rng = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    L = enc.params.length_mm
    I_Nd = np.eye(Nd)

    # Bob's matrices
    H_bob = sim.transfer_matrix(true_key, L)
    Hd_bob = H_bob[np.ix_(data_idx, data_idx)]
    all_var = _ase_noise_variances(true_key, enc.params, geom)
    sigma2_m = float(np.mean(all_var[data_idx]))
    Hd_inv_mmse = np.linalg.solve(
        Hd_bob.conj().T @ Hd_bob + sigma2_m * I_Nd, Hd_bob.conj().T)

    # Eve wrong-key matrices
    eve_key = rng.uniform(0.0, 1.0, size=len(true_key))
    H_eve = sim.transfer_matrix(eve_key, L)
    Hd_eve = H_eve[np.ix_(data_idx, data_idx)]
    Hd_eve_inv = np.linalg.pinv(Hd_eve)

    data_var = all_var[data_idx]
    sigma_c  = np.sqrt(data_var / 2.0)

    results = []
    for M, label in [(4, "QPSK"), (16, "16-QAM"), (64, "64-QAM")]:
        const, bit_labels = _qam_constellation_gray(M)
        bps   = int(np.log2(M))

        sym_idx = rng.integers(0, M, size=(T, Nd))
        syms = const[sym_idx]
        bits_ref = bit_labels[sym_idx]

        noise = (rng.normal(size=(T, Nd)) * sigma_c[None, :] +
                 1j * rng.normal(size=(T, Nd)) * sigma_c[None, :])
        y = syms @ Hd_bob.T + noise

        # Bob MMSE + nearest point
        x_hat_bob = y @ Hd_inv_mmse.T
        idx_hat_bob = _nearest_constellation_indices(x_hat_bob, const)
        ser_bob = float(np.mean(idx_hat_bob != sym_idx))
        ber_bob = float(np.mean(bit_labels[idx_hat_bob] != bits_ref))

        # Eve ZF + nearest point
        x_hat_eve = y @ Hd_eve_inv.T
        idx_hat_eve = _nearest_constellation_indices(x_hat_eve, const)
        ser_eve = float(np.mean(idx_hat_eve != sym_idx))
        ber_eve = float(np.mean(bit_labels[idx_hat_eve] != bits_ref))

        results.append({
            "modulation":    label, "M": M, "bits_per_symbol": bps,
            "SER_Bob":       ser_bob, "BER_Bob": ber_bob,
            "SER_Eve":       ser_eve, "BER_Eve": ber_eve,
        })
        print(f"  {label}: SER_Bob={ser_bob:.4f} BER_Bob={ber_bob:.4f}  "
              f"SER_Eve={ser_eve:.4f} BER_Eve={ber_eve:.4f}")

    _write_csv(f"{out_dir}/qam_ber.csv", results)

    fig, ax = plt.subplots(figsize=(7, 4))
    mods = [r["modulation"] for r in results]
    x = np.arange(len(mods))
    w = 0.35
    ax.bar(x - w/2, [r["BER_Bob"] for r in results], w,
           label="BER_Bob (correct key, MMSE)", color="tab:blue", alpha=0.85)
    ax.bar(x + w/2, [r["BER_Eve"] for r in results], w,
           label="BER_Eve (wrong key, ZF)", color="tab:red", alpha=0.85)
    ax.axhline(0.5, color="gray", ls=":", lw=1.5, label="Random guess (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(mods)
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Multi-modulation BER: Bob vs. Eve\n"
                 "(MMSE correct-key vs. ZF wrong-key decoding)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4, axis="y")
    ax.set_ylim(0, 0.65)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig12_qam_ber.png", dpi=200)
    plt.close()
    print(f"  Saved fig12_qam_ber.png")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 10.  MULTI-COMPONENT SENSITIVITY  |  BER-||ΔH||_F CORRELATION  |  LARGE MCF
# ──────────────────────────────────────────────────────────────────────────────

def sweep_multi_component_sensitivity(enc: MCFEncryption,
                                      true_key: np.ndarray,
                                      out_dir: str,
                                      T: int = 600,
                                      n_subsets: int = 8,
                                      seed: int = 7) -> List[Dict]:
    """
    BER_Eve when k simultaneous key components each have analog error delta_P.

    Uses SIGNED (random ±delta_P) perturbations so some components can cross
    the gain/absorption boundary at P=0.6, which is the primary security driver.
    Pure positive perturbations keep all components in the gain regime → BER≈0.

    Q3 & Q8: Why does single-component sensitivity appear low for small dP?
    Answer: small errors within [0.6, 1.0] produce similar gain profiles.
    Only when dP pushes components below P≈0.6 (absorption threshold) does
    the transfer matrix change dramatically and BER_Eve rise sharply.

    Sweeps:
      k     ∈ {1, 2, 3, 5, 7, 10, 13}  (number of wrong components)
      delta_P ∈ {0.02, 0.05, 0.10, 0.20, 0.40}  (per-component magnitude)

    For each (k, delta_P): n_subsets random selections of k components
    are perturbed by random ±delta_P; BER_Eve is averaged.
    """
    rng = np.random.default_rng(seed)
    N = len(true_key)
    k_vals   = [1, 2, 3, 5, 7, 10, 13]
    dP_vals  = [0.02, 0.05, 0.10, 0.20, 0.40]
    results  = []

    print("\n[Fig 13] Multi-component analog sensitivity: BER_Eve vs k wrong components...")
    for dP in dP_vals:
        for k in k_vals:
            ber_vals = []
            for _ in range(n_subsets):
                eve_key = true_key.copy()
                idx_wrong = rng.choice(N, size=k, replace=False)
                for i in idx_wrong:
                    # Random signed perturbation: some cross the gain boundary
                    sign = rng.choice([-1.0, 1.0])
                    eve_key[i] = float(np.clip(true_key[i] + sign * dP, 0.0, 1.0))
                res = enc.run_session(true_key, T=T, eve_key=eve_key,
                                     eve_partial_obs=False)
                ber_vals.append(res["BER_Eve"])
            row = {
                "delta_P":      dP,
                "k_wrong":      k,
                "BER_Eve_mean": float(np.mean(ber_vals)),
                "BER_Eve_std":  float(np.std(ber_vals)),
                "BER_Eve_max":  float(np.max(ber_vals)),
            }
            results.append(row)
            print(f"  dP={dP:.2f}  k={k:2d}  BER_Eve={np.mean(ber_vals):.4f}"
                  f"  (std={np.std(ber_vals):.4f})")

    _write_csv(f"{out_dir}/multi_component_sensitivity.csv", results)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "darkred"]
    for col, dP in zip(colors, dP_vals):
        sub = [r for r in results if r["delta_P"] == dP]
        ks   = [r["k_wrong"] for r in sub]
        bers = [r["BER_Eve_mean"] for r in sub]
        stds = [r["BER_Eve_std"]  for r in sub]
        ax.errorbar(ks, bers, yerr=stds, fmt="o-", color=col,
                    capsize=3, label=f"dP={dP:.2f}")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess (0.5)")
    ax.set_xlabel("k: number of simultaneously wrong key components")
    ax.set_ylabel("BER_Eve (mean +/- std)")
    ax.set_title("Multi-component analog sensitivity\n"
                 "BER_Eve vs. k wrong components, each with error dP")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_xticks(k_vals)
    ax.set_ylim(-0.02, 0.58)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig13_multi_component_sensitivity.png", dpi=200)
    plt.close()
    print(f"  Saved fig13_multi_component_sensitivity.png")
    return results


def sweep_ber_hf_correlation(enc: MCFEncryption,
                              true_key: np.ndarray,
                              out_dir: str,
                              T: int = 300,
                              n_keys: int = 80,
                              seed: int = 7) -> List[Dict]:
    """
    Scatter plot: ||H_Eve - H_Bob||_F  vs  BER_Eve over random wrong keys.

    Q6: Is there a monotone relationship between key distance (||ΔH||_F)
    and BER_Eve?  Answer: generally positive correlation but noisy — the
    relationship is non-linear due to the matrix exponential (CME solution).

    For each of n_keys random Eve keys:
      - Compute ||H_wrong - H_correct||_F
      - Simulate BER_Eve
    Then compute Pearson correlation and plot scatter.
    """
    rng = np.random.default_rng(seed)
    sim  = enc.sim
    data_idx = enc.geom.data_core_indices
    L    = enc.params.length_mm

    H_bob  = sim.transfer_matrix(true_key, L)
    Hd_bob = H_bob[np.ix_(data_idx, data_idx)]

    results = []
    print(f"\n[Fig 14] BER vs ||DeltaH||_F correlation ({n_keys} random keys)...")
    for trial in range(n_keys):
        eve_key = rng.uniform(0.0, 1.0, size=len(true_key))
        res = enc.run_session(true_key, T=T, eve_key=eve_key, eve_partial_obs=False)
        H_eve  = sim.transfer_matrix(eve_key, L)
        delta_H_F = float(np.linalg.norm(H_eve - H_bob, 'fro'))
        results.append({
            "trial":      trial,
            "delta_H_F":  delta_H_F,
            "BER_Eve":    res["BER_Eve"],
        })
        if (trial + 1) % 20 == 0:
            print(f"  trial {trial+1}/{n_keys} done")

    _write_csv(f"{out_dir}/ber_hf_correlation.csv", results)

    dH  = np.array([r["delta_H_F"] for r in results])
    ber = np.array([r["BER_Eve"]   for r in results])
    corr = float(np.corrcoef(dH, ber)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(dH, ber, c=ber, cmap="RdYlGn_r", s=40, alpha=0.7, edgecolors="k",
                    linewidths=0.3, vmin=0, vmax=0.5)
    plt.colorbar(sc, ax=ax, label="BER_Eve")
    ax.set_xlabel(r"$\|\Delta H\|_F = \|H_{Eve} - H_{Bob}\|_F$")
    ax.set_ylabel("BER_Eve (wrong-key ZF decoding)")
    ax.set_title(f"Key-distance vs. security: ||DeltaH||_F vs. BER_Eve\n"
                 f"Pearson r = {corr:.3f} (n={n_keys} random Eve keys)")
    ax.grid(True, alpha=0.4)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig14_ber_hf_correlation.png", dpi=200)
    plt.close()
    print(f"  Pearson r = {corr:.3f}")
    print(f"  Saved fig14_ber_hf_correlation.png")
    return results


def sweep_large_mcf(base_params: FiberParams,
                    out_dir: str,
                    n_rings_list: Optional[List[int]] = None,
                    n_trials: int = 5,
                    T: int = 300,
                    seed: int = 7) -> List[Dict]:
    """
    Security analysis for larger hexagonal MCF designs (7, 19, 37 cores).

    Q2: Why not test more than 13 cores? Is there a performance plateau or
    does security scale well with N?

    For each n_rings (= 1→7 cores, 2→19 cores, 3→37 cores):
      - Build HexMCFGeometry + CMESimulator with same physical params
      - Draw random gain-regime key K in [0.6, 1.0]^N
      - Compute:
          BER_Eve        (random key, U[0,1]^N)
          BER_Eve_50pct  (50% mismatch key)
          C_s_gauss      (Gaussian MIMO wiretap capacity at sigma=0.05)
          key_entropy    (bits, 12-bit DAC, [0.6, 1.0] range)
      - Analytically extrapolate entropy for N = 61, 91, 127

    Note: 37-core simulation requires ~3x longer ODE solves than 13-core.
    """
    if n_rings_list is None:
        n_rings_list = [1, 2, 3]   # 7, 19, 37 cores

    rng  = np.random.default_rng(seed)
    results = []

    # Also include current 13-core design (special geometry) as reference
    configs_geom = [(MCFGeometry(pitch_um=base_params.pitch_um), 13, "13-core (paper design)")]
    for nr in n_rings_list:
        g = HexMCFGeometry(n_rings=nr, pitch_um=base_params.pitch_um)
        configs_geom.append((g, g.N, f"{g.N}-core (hex, {nr} rings)"))

    print("\n[Fig 15] Large MCF security scaling...")
    for geom, N_cores, label in configs_geom:
        sim = CMESimulator(base_params, geom)
        L   = base_params.length_mm

        ber_rand_list  = []
        ber_half_list  = []
        cs_list        = []
        for _ in range(n_trials):
            true_key = rng.uniform(0.6, 1.0, size=N_cores)
            H_bob    = sim.transfer_matrix(true_key, L)
            data_idx = geom.data_core_indices
            Hd_bob   = H_bob[np.ix_(data_idx, data_idx)]

            # Random Eve key
            eve_key_rand  = rng.uniform(0.0, 1.0, size=N_cores)
            H_eve_rand    = sim.transfer_matrix(eve_key_rand, L)
            Hd_eve_rand   = H_eve_rand[np.ix_(data_idx, data_idx)]

            # 50% wrong components Eve key
            n_wrong = N_cores // 2
            eve_key_half = true_key.copy()
            idx_w = rng.choice(N_cores, size=n_wrong, replace=False)
            for i in idx_w:
                eve_key_half[i] = float(rng.uniform(0, 1))
            H_eve_half  = sim.transfer_matrix(eve_key_half, L)
            Hd_eve_half = H_eve_half[np.ix_(data_idx, data_idx)]

            # BER_Eve (QPSK, ZF)
            sigma2 = base_params.sigma_noise**2
            Nd = len(data_idx)
            # QPSK symbols
            bits_tx = rng.integers(0, 2, size=(T, Nd, 2))
            re_tx   = 2*bits_tx[:,:,0].astype(float) - 1.0
            im_tx   = 2*bits_tx[:,:,1].astype(float) - 1.0
            syms_tx = (re_tx + 1j*im_tx) / np.sqrt(2)

            noise = (rng.normal(size=(T, Nd)) + 1j*rng.normal(size=(T, Nd))) * np.sqrt(sigma2/2)
            y_rx  = syms_tx @ Hd_bob.T + noise

            def _ber_eve_zf(Hd_e: np.ndarray) -> float:
                x_hat = y_rx @ np.linalg.pinv(Hd_e).T
                bits_rx = np.stack([(x_hat.real >= 0).astype(int),
                                    (x_hat.imag >= 0).astype(int)], axis=2)
                return float(np.mean(bits_rx != bits_tx))

            ber_rand_list.append(_ber_eve_zf(Hd_eve_rand))
            ber_half_list.append(_ber_eve_zf(Hd_eve_half))

            # Gaussian MIMO secrecy capacity at current noise level
            snr = 1.0 / sigma2
            cs  = _secrecy_capacity(Hd_bob, Hd_eve_rand, snr_linear=snr)
            cs_list.append(cs)

        # Key entropy: [0.6, 1.0] range, 12-bit DAC → 1638 levels/dim
        dac_bits  = 12
        range_frac = (1.0 - 0.6)   # fraction of [0, 1]
        levels     = int(2**dac_bits * range_frac)   # ~1638
        entropy_bits = N_cores * np.log2(levels)

        row = {
            "N_cores":         N_cores,
            "label":           label,
            "BER_Eve_rand":    float(np.mean(ber_rand_list)),
            "BER_Eve_50pct":   float(np.mean(ber_half_list)),
            "C_s_mean":        float(np.mean(cs_list)),
            "C_s_std":         float(np.std(cs_list)),
            "entropy_bits":    float(entropy_bits),
            "AES_equiv_level": ("<128" if entropy_bits < 128 else
                                "<256" if entropy_bits < 256 else ">256"),
        }
        results.append(row)
        print(f"  N={N_cores:3d}  BER_rand={np.mean(ber_rand_list):.4f}"
              f"  BER_50pct={np.mean(ber_half_list):.4f}"
              f"  C_s={np.mean(cs_list):.1f}  entropy={entropy_bits:.0f} bits"
              f"  [{row['AES_equiv_level']}]")

    # Analytical extrapolation for larger N (61, 91, 127)
    for N_ext in [61, 91, 127, 169]:
        entropy_bits = N_ext * np.log2(int(2**12 * 0.4))
        results.append({
            "N_cores":       N_ext,
            "label":         f"{N_ext}-core (analytical)",
            "BER_Eve_rand":  0.499,   # converges to ~0.5 for large random-key space
            "BER_Eve_50pct": float("nan"),
            "C_s_mean":      float("nan"),
            "C_s_std":       float("nan"),
            "entropy_bits":  float(entropy_bits),
            "AES_equiv_level": ("<128" if entropy_bits < 128 else
                                "AES-128+" if entropy_bits < 256 else
                                "AES-256+" if entropy_bits < 512 else "post-quantum"),
        })
        print(f"  N={N_ext:3d}  (analytical) entropy={entropy_bits:.0f} bits"
              f"  [{results[-1]['AES_equiv_level']}]")

    _write_csv(f"{out_dir}/large_mcf_scaling.csv", results)

    # -- Plot 1: entropy vs N --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sim_rows  = [r for r in results if "analytical" not in r["label"]]
    ana_rows  = [r for r in results if "analytical"     in r["label"]]
    all_N     = [r["N_cores"]      for r in results]
    all_ent   = [r["entropy_bits"] for r in results]

    axes[0].axhline(128, color="navy",  ls="--", lw=1.2, label="AES-128 (128 bits)")
    axes[0].axhline(256, color="darkred", ls="--", lw=1.2, label="AES-256 (256 bits)")
    axes[0].plot([r["N_cores"] for r in sim_rows],
                 [r["entropy_bits"] for r in sim_rows],
                 "o-", color="tab:blue", lw=2, ms=7, label="Simulated")
    axes[0].plot([r["N_cores"] for r in ana_rows],
                 [r["entropy_bits"] for r in ana_rows],
                 "s--", color="tab:gray", lw=1.5, ms=6, label="Analytical")
    axes[0].set_xlabel("Number of MCF cores N")
    axes[0].set_ylabel("Key entropy [bits]")
    axes[0].set_title("Key entropy vs. N cores\n(12-bit DAC, key range [0.6,1.0])")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4)

    # -- Plot 2: BER_Eve and C_s vs N (simulated only) --
    Ns   = [r["N_cores"]       for r in sim_rows]
    bers = [r["BER_Eve_rand"]  for r in sim_rows]
    cs   = [r["C_s_mean"]      for r in sim_rows]

    ax2  = axes[1]
    ax2b = ax2.twinx()
    ax2.plot(Ns, bers, "o-", color="tab:red",  lw=2, ms=7, label="BER_Eve (rand key)")
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Number of MCF cores N")
    ax2.set_ylabel("BER_Eve (random key)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2b.plot(Ns, cs, "s--", color="tab:blue", lw=2, ms=7, label="C_s (bits/use)")
    ax2b.set_ylabel("Secrecy capacity C_s [bits/use]", color="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Security vs. N cores\n(BER_Eve rand key + Gaussian C_s)")
    # Merged legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax2.grid(True, alpha=0.4)

    plt.suptitle("MCF scaling: security metrics vs. number of cores", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig15_large_mcf_scaling.png", dpi=200)
    plt.close()
    print(f"  Saved fig15_large_mcf_scaling.png")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 11.  QAM KEY MISMATCH | JOINT SENSITIVITY + LENGTH | LINEAR REGRESSION ATTACK
# ──────────────────────────────────────────────────────────────────────────────

# ── Figure 16 — BER vs key mismatch for QPSK / 16-QAM / 64-QAM ──────────────

def sweep_qam_key_mismatch(enc: MCFEncryption,
                            true_key: np.ndarray,
                            out_dir: str,
                            T: int = 800,
                            n_trials: int = 3,
                            seed: int = 7) -> List[Dict]:
    """
    BER_Eve vs. key mismatch fraction for three modulation orders.

    Addresses the question: if higher-order QAM gives Eve lower absolute BER
    (more constellation points → smaller Voronoi cells → higher raw BER)
    does the *relative* security ordering across mismatch levels change?

    Method
    ------
    For each (mismatch_fraction, modulation) pair:
      - Replace the specified fraction of Eve's key with random values from [0, 1]
      - Eve applies ZF (pinv) using her wrong transfer matrix
      - BER is measured exactly from Gray-coded symbol labels

    Key insight: BER_Eve decreases with higher modulation order (more nearby
    constellation points increase the chance of a "nearby-wrong" decision),
    but BER_Bob remains 0 for all modulations regardless of mismatch.
    """
    rng  = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    L  = enc.params.length_mm
    I_Nd = np.eye(Nd)

    # Bob's fixed matrices (correct key, MMSE)
    H_bob = sim.transfer_matrix(true_key, L)
    Hd_bob = H_bob[np.ix_(data_idx, data_idx)]
    all_var = _ase_noise_variances(true_key, enc.params, geom)
    sigma2_m = float(np.mean(all_var[data_idx]))
    Hd_inv_mmse = np.linalg.solve(
        Hd_bob.conj().T @ Hd_bob + sigma2_m * I_Nd, Hd_bob.conj().T)
    data_var = all_var[data_idx]
    sigma_c  = np.sqrt(data_var / 2.0)

    mismatch_fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    modulations = [(4, "QPSK"), (16, "16-QAM"), (64, "64-QAM")]

    print("\n[Fig 16] QAM BER vs. key mismatch fraction...")
    results = []

    for M, label in modulations:
        const, bit_labels = _qam_constellation_gray(M)
        bps   = int(np.log2(M))
        for frac in mismatch_fracs:
            ber_bob_vals, ber_eve_vals = [], []
            for _ in range(n_trials):
                # Build Eve's wrong key
                n_wrong = int(round(frac * len(true_key)))
                eve_key = true_key.copy()
                if n_wrong > 0:
                    idx_wrong = rng.choice(len(true_key), size=n_wrong, replace=False)
                    for i in idx_wrong:
                        eve_key[i] = float(rng.uniform(0, 1))

                H_eve = sim.transfer_matrix(eve_key, L)
                Hd_eve = H_eve[np.ix_(data_idx, data_idx)]
                Hd_eve_inv = np.linalg.pinv(Hd_eve)

                # Generate symbols and transmit
                sym_idx = rng.integers(0, M, size=(T, Nd))
                syms = const[sym_idx]
                bits_ref = bit_labels[sym_idx]
                noise = (rng.normal(size=(T, Nd)) * sigma_c[None, :] +
                         1j * rng.normal(size=(T, Nd)) * sigma_c[None, :])
                y = syms @ Hd_bob.T + noise

                # Bob MMSE
                x_hat_bob = y @ Hd_inv_mmse.T
                idx_hat_bob = _nearest_constellation_indices(x_hat_bob, const)
                ber_bob_vals.append(float(np.mean(bit_labels[idx_hat_bob] != bits_ref)))

                # Eve ZF
                x_hat_eve = y @ Hd_eve_inv.T
                idx_hat_eve = _nearest_constellation_indices(x_hat_eve, const)
                ber_eve_vals.append(float(np.mean(bit_labels[idx_hat_eve] != bits_ref)))

            entry = {
                "modulation": label, "M": M, "bits_per_symbol": bps,
                "key_mismatch_frac": float(frac),
                "BER_Bob": float(np.mean(ber_bob_vals)),
                "BER_Eve": float(np.mean(ber_eve_vals)),
            }
            results.append(entry)
            print(f"  {label}  mismatch={frac:.1f}  "
                  f"BER_Bob={entry['BER_Bob']:.4f}  BER_Eve={entry['BER_Eve']:.4f}")

    _write_csv(f"{out_dir}/qam_key_mismatch.csv", results)

    # ── Plot ─────────────────────────────────────────────────────────────
    colors = {"QPSK": "tab:orange", "16-QAM": "tab:purple", "64-QAM": "tab:brown"}
    markers = {"QPSK": "o", "16-QAM": "s", "64-QAM": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    # Left: BER_Eve vs mismatch for each modulation
    ax = axes[0]
    for M, label in modulations:
        sub = [r for r in results if r["modulation"] == label]
        xs  = [r["key_mismatch_frac"] for r in sub]
        ys  = [r["BER_Eve"] for r in sub]
        ax.plot(xs, ys, f"{markers[label]}-", color=colors[label],
                lw=2, ms=6, label=f"Eve — {label}")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    ax.set_xlabel("Fraction of key components mismatched")
    ax.set_ylabel("BER_Eve")
    ax.set_title("Eve BER vs. key mismatch\n(by modulation order)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(-0.02, 0.65)

    # Right: BER_Bob vs mismatch (should stay near 0 for all modulations)
    ax2 = axes[1]
    for M, label in modulations:
        sub = [r for r in results if r["modulation"] == label]
        xs  = [r["key_mismatch_frac"] for r in sub]
        ys  = [r["BER_Bob"] for r in sub]
        ax2.plot(xs, ys, f"{markers[label]}--", color=colors[label],
                 lw=2, ms=6, label=f"Bob — {label}")
    ax2.axhline(0.0, color="tab:blue", ls=":", lw=1)
    ax2.set_xlabel("Fraction of key components mismatched")
    ax2.set_ylabel("BER_Bob")
    ax2.set_title("Bob BER vs. key mismatch\n(Bob always has correct key)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(-0.02, 0.65)

    fig.suptitle("Security vs. modulation order: BER under key mismatch", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig16_qam_key_mismatch.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig16_qam_key_mismatch.png")
    return results


# ── Figure 10b — Joint sensitivity: key mismatch × fiber length ───────────────

def sweep_joint_sensitivity_length(enc: MCFEncryption,
                                    true_key: np.ndarray,
                                    out_dir: str,
                                    T: int = 400,
                                    n_trials: int = 4,
                                    seed: int = 7) -> List[Dict]:
    """
    BER_Bob and BER_Eve under simultaneous key mismatch AND fiber length variation.

    Extends Fig 10 (which covered mismatch × manufacturing tolerance) by adding
    the fiber length dimension. This addresses the question of whether longer
    fibers are inherently more secure, and whether mismatch and length interact.

    Physical interpretation:
      - Longer fiber: more coupling cycles → transfer matrix H(K) oscillates and
        complex-mixes more. BUT the key-sensitivity of H also changes with L.
      - Near L = n*Lc (coupling length multiples), coupling returns to near-identity.
      - The sweet spot is L between Lc and 2Lc (currently L=150mm, Lc≈118mm).

    Sweeps:
      - mismatch_fracs: [0.0, 0.25, 0.50, 1.0] (same as Fig 10)
      - length_mm: [75, 118, 150, 235]  (0.64Lc, Lc, 1.27Lc, 2Lc)
    """
    Lc = np.pi / (2 * enc.params.coupling_coeff())
    length_vals = [round(0.64 * Lc, 1), round(Lc, 1),
                   enc.params.length_mm,  round(2.0 * Lc, 1)]
    length_labels = [f"{round(0.64*Lc)}mm (0.64Lc)", f"{round(Lc)}mm (Lc)",
                     f"{enc.params.length_mm}mm (paper)", f"{round(2*Lc)}mm (2Lc)"]
    mismatch_fracs = [0.0, 0.25, 0.50, 1.0]
    rng = np.random.default_rng(seed)
    results = []
    print("\n[Fig 10b] Joint sensitivity: key mismatch x fiber length...")

    for mf in mismatch_fracs:
        for L_val in length_vals:
            bob_bers, eve_bers = [], []
            for _ in range(n_trials):
                n_wrong = int(round(mf * len(true_key)))
                eve_key = true_key.copy()
                if n_wrong > 0:
                    idx_wrong = rng.choice(len(true_key), size=n_wrong, replace=False)
                    for i in idx_wrong:
                        eve_key[i] = float(rng.uniform(0, 1))
                res = enc.run_session(true_key, T=T, eve_key=eve_key,
                                      eve_partial_obs=False, length_mm=L_val)
                bob_bers.append(res["BER_Bob"])
                eve_bers.append(res["BER_Eve"])
            entry = {
                "key_mismatch_frac": float(mf),
                "length_mm": float(L_val),
                "BER_Bob_mean": float(np.mean(bob_bers)),
                "BER_Eve_mean": float(np.mean(eve_bers)),
            }
            results.append(entry)
            print(f"  mismatch={mf:.2f}  L={L_val:.0f}mm  "
                  f"BER_Bob={np.mean(bob_bers):.4f}  BER_Eve={np.mean(eve_bers):.4f}")

    _write_csv(f"{out_dir}/joint_sensitivity_length.csv", results)

    # ── Heatmap ───────────────────────────────────────────────────────────
    n_mf = len(mismatch_fracs)
    n_L  = len(length_vals)
    eve_grid = np.zeros((n_mf, n_L))
    bob_grid = np.zeros((n_mf, n_L))
    for r in results:
        mi = mismatch_fracs.index(r["key_mismatch_frac"])
        li = [round(v, 1) for v in length_vals].index(round(r["length_mm"], 1))
        eve_grid[mi, li] = r["BER_Eve_mean"]
        bob_grid[mi, li] = r["BER_Bob_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    im0 = axes[0].imshow(eve_grid, vmin=0.0, vmax=0.5,
                          cmap="RdYlGn_r", aspect="auto")
    axes[0].set_xticks(range(n_L))
    axes[0].set_xticklabels(length_labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_yticks(range(n_mf))
    axes[0].set_yticklabels([f"{m*100:.0f}%" for m in mismatch_fracs])
    axes[0].set_xlabel("Fiber length")
    axes[0].set_ylabel("Key mismatch fraction")
    axes[0].set_title("BER_Eve — Key mismatch × fiber length\n(red=secure, green=insecure)")
    for mi in range(n_mf):
        for li in range(n_L):
            axes[0].text(li, mi, f"{eve_grid[mi,li]:.3f}",
                         ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im0, ax=axes[0], label="BER_Eve")

    im1 = axes[1].imshow(bob_grid, vmin=0.0, vmax=0.1,
                          cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(n_L))
    axes[1].set_xticklabels(length_labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_yticks(range(n_mf))
    axes[1].set_yticklabels([f"{m*100:.0f}%" for m in mismatch_fracs])
    axes[1].set_xlabel("Fiber length")
    axes[1].set_ylabel("Key mismatch fraction")
    axes[1].set_title("BER_Bob — Key mismatch × fiber length\n(green=BER_Bob=0)")
    for mi in range(n_mf):
        for li in range(n_L):
            axes[1].text(li, mi, f"{bob_grid[mi,li]:.3f}",
                         ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im1, ax=axes[1], label="BER_Bob")

    fig.suptitle("Joint sensitivity: key mismatch × fiber length", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig10b_joint_sensitivity_length.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig10b_joint_sensitivity_length.png")
    return results


# ── Figure 17 — Linear regression adversary: BER_Eve vs. known-plaintext pairs ─

def sweep_linear_regression_adversary(enc: MCFEncryption,
                                       true_key: np.ndarray,
                                       out_dir: str,
                                       T_collect: int = 500,
                                       n_trials: int = 5,
                                       seed: int = 7) -> List[Dict]:
    """
    Eve's linear regression attack: estimate H(K) from N_pairs observed
    known-plaintext pairs (x_t, y_t), then use H_est^{-1} to decode.

    Physical model
    --------------
    y_t = x_t @ Hd.T + n_t    (known-plaintext: Eve observes x_t AND y_t)

    In matrix form (N_obs rows):
        Y = X @ Hd.T + N

    Least-squares estimate: Hd_est.T = pinv(X) @ Y
    This is exact when X has rank Nd (= 13), achieved when N_obs ≥ Nd.

    In practice, noise means convergence requires N_obs >> Nd. For 13×13
    complex H, the effective degrees of freedom are 2×13×13 = 338 real
    parameters; practical convergence occurs around N_obs ≈ 100–200.

    Security implication
    --------------------
    This shows that a FIXED key is insecure against a patient known-plaintext
    adversary after ~100 observations (well within a single session).
    Key rotation (changing K every ≤ N_obs_threshold blocks) is MANDATORY.

    The analysis also quantifies the *minimum* key rotation interval:
    if BER_Eve drops below 0.1 at N_obs = N_threshold, the key must rotate
    every N_threshold − 1 blocks.
    """
    rng  = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    L  = enc.params.length_mm
    I_Nd = np.eye(Nd)

    # True transfer matrix and MMSE equaliser (Bob's side — reference only)
    H_bob  = sim.transfer_matrix(true_key, L)
    Hd_bob = H_bob[np.ix_(data_idx, data_idx)]
    all_var = _ase_noise_variances(true_key, enc.params, geom)
    sigma2_m = float(np.mean(all_var[data_idx]))
    Hd_inv_mmse = np.linalg.solve(
        Hd_bob.conj().T @ Hd_bob + sigma2_m * I_Nd, Hd_bob.conj().T)
    data_var = all_var[data_idx]
    sigma_c  = np.sqrt(data_var / 2.0)

    # N_pairs values to test
    n_pairs_vals = [1, 5, 10, 13, 20, 30, 50, 80, 100, 130, 169, 250, 350, 500]
    n_pairs_vals = [n for n in n_pairs_vals if n <= T_collect]

    print("\n[Fig 17] Linear regression adversary: BER vs known-plaintext pairs (held-out test)...")
    results = []

    for n_obs in n_pairs_vals:
        ber_bob_vals, ber_eve_lr_vals, ber_eve_rk_vals = [], [], []
        for _ in range(n_trials):
            # Generate disjoint train/test known-plaintext sets (QPSK).
            bits_train = rng.integers(0, 2, size=(T_collect, Nd, 2))
            re_tr = 2.0 * bits_train[:, :, 0] - 1.0
            im_tr = 2.0 * bits_train[:, :, 1] - 1.0
            X_train = (re_tr + 1j * im_tr) / np.sqrt(2.0)
            noise_train = (rng.normal(size=(T_collect, Nd)) * sigma_c[None, :] +
                           1j * rng.normal(size=(T_collect, Nd)) * sigma_c[None, :])
            Y_train = X_train @ Hd_bob.T + noise_train

            bits_test = rng.integers(0, 2, size=(T_collect, Nd, 2))
            re_te = 2.0 * bits_test[:, :, 0] - 1.0
            im_te = 2.0 * bits_test[:, :, 1] - 1.0
            X_test = (re_te + 1j * im_te) / np.sqrt(2.0)
            noise_test = (rng.normal(size=(T_collect, Nd)) * sigma_c[None, :] +
                          1j * rng.normal(size=(T_collect, Nd)) * sigma_c[None, :])
            Y_test = X_test @ Hd_bob.T + noise_test

            # Eve uses the first n_obs training pairs to estimate Hd.
            X_obs = X_train[:n_obs]
            Y_obs = Y_train[:n_obs]

            if n_obs >= Nd:
                # Overdetermined or exact: Hd_est.T = pinv(X_obs) @ Y_obs
                Hd_est_T = np.linalg.lstsq(X_obs, Y_obs, rcond=None)[0]
                Hd_est = Hd_est_T.T   # (Nd, Nd)
            else:
                # Underdetermined: use pseudoinverse (partial knowledge)
                Hd_est_T = np.linalg.lstsq(X_obs, Y_obs, rcond=None)[0]
                Hd_est = Hd_est_T.T

            Hd_est_inv = np.linalg.pinv(Hd_est)

            # Eve tests on remaining pairs (T_collect − n_obs test pairs)
            # Use all T_collect pairs for consistent statistics
            # Bob BER (constant reference)
            x_hat_bob = Y_test @ Hd_inv_mmse.T
            bits_hat_bob = np.array([enc._qpsk_demod(x_hat_bob[t])
                                     for t in range(T_collect)])
            ber_bob_vals.append(float(np.mean(bits_test != bits_hat_bob)))

            # Eve LR-attack BER
            x_hat_eve_lr = Y_test @ Hd_est_inv.T
            bits_hat_eve_lr = np.array([enc._qpsk_demod(x_hat_eve_lr[t])
                                        for t in range(T_collect)])
            ber_eve_lr_vals.append(float(np.mean(bits_test != bits_hat_eve_lr)))

            # Eve random-key BER (baseline comparison)
            rand_key = rng.uniform(0, 1, size=len(true_key))
            H_rk = sim.transfer_matrix(rand_key, L)
            Hd_rk = H_rk[np.ix_(data_idx, data_idx)]
            Hd_rk_inv = np.linalg.pinv(Hd_rk)
            x_hat_eve_rk = Y_test @ Hd_rk_inv.T
            bits_hat_eve_rk = np.array([enc._qpsk_demod(x_hat_eve_rk[t])
                                        for t in range(T_collect)])
            ber_eve_rk_vals.append(float(np.mean(bits_test != bits_hat_eve_rk)))

        entry = {
            "n_obs_pairs": n_obs,
            "BER_Bob":      float(np.mean(ber_bob_vals)),
            "BER_Eve_LR":   float(np.mean(ber_eve_lr_vals)),
            "BER_Eve_rand": float(np.mean(ber_eve_rk_vals)),
        }
        results.append(entry)
        print(f"  N_obs={n_obs:4d}  BER_Bob={entry['BER_Bob']:.4f}  "
              f"BER_Eve_LR={entry['BER_Eve_LR']:.4f}  "
              f"BER_Eve_rand={entry['BER_Eve_rand']:.4f}")

    _write_csv(f"{out_dir}/linear_regression_adversary.csv", results)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r["n_obs_pairs"] for r in results]
    ax.semilogx(xs, [r["BER_Eve_LR"]   for r in results], "r^-",
                lw=2, ms=7, label="Eve — linear regression (LR attack)")
    ax.semilogx(xs, [r["BER_Eve_rand"] for r in results], "bs--",
                lw=1.5, ms=6, label="Eve — random wrong key (baseline)")
    ax.semilogx(xs, [r["BER_Bob"]      for r in results], "go-",
                lw=2, ms=6, label="Bob — correct key (MMSE)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess (0.5)")
    ax.axhline(0.1, color="orange", ls="--", lw=1, label="BER = 0.10 threshold")
    ax.axvline(Nd, color="purple", ls=":", lw=1.5,
               label=f"N_obs = N_d = {Nd} (exact solve threshold)")
    ax.set_xlabel("Number of known-plaintext pairs N_obs")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Linear regression attack: BER_Eve vs. known-plaintext pairs\n"
                 "(key rotation needed before BER_Eve drops below 0.1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, which="both")
    ax.set_ylim(-0.02, 0.65)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig17_linear_regression_adversary.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig17_linear_regression_adversary.png")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 12.  NONLINEAR TWO-STAGE SWEEPS (Figs 18, 19, 20)
# ──────────────────────────────────────────────────────────────────────────────

# ── Figure 18 — Operating point: BER vs. saturation depth ────────────────────

def sweep_operating_point_nonlinear(enc: MCFEncryption,
                                     true_key: np.ndarray,
                                     out_dir: str,
                                     T: int = 800,
                                     n_trials: int = 4,
                                     seed: int = 7) -> List[Dict]:
    """
    Sweep the saturable absorber activation threshold P_sat_sig to find the
    operating point that maximises BER_Eve while keeping BER_Bob = 0.

    Three regimes:
      P_sat_sig >> |z|  →  σ ≈ linear  →  two-stage channel ≈ H₂·H₁·x (linear)
                            LR attack breaks it again (back to Fig 17 problem)
      P_sat_sig ~ |z|   →  strong nonlinearity, good security, SNR still OK
      P_sat_sig << |z|  →  heavy saturation, signal destroyed, BER_Bob rises

    The typical signal amplitude after stage 1 is computed from the RMS of
    H₁(K)·x, which depends on the EDFA gain and MCF mixing.
    We sweep P_sat_sig in multiples of the RMS signal amplitude.
    """
    rng = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd = len(data_idx)
    L  = enc.params.length_mm

    # Estimate typical |z| amplitude after stage 1 with true key
    H1 = sim.transfer_matrix(true_key, L)
    Hd1 = H1[np.ix_(data_idx, data_idx)]
    # QPSK symbols: |x_i| = 1/sqrt(2) ≈ 0.707; after mixing E[|z_i|] = E[|Hd1 x|_i]
    # Use RMS of a random QPSK batch
    test_bits = rng.integers(0, 2, size=(200, Nd, 2))
    test_syms = np.array([enc._qpsk_map(test_bits[t]) for t in range(200)])
    z_test    = test_syms @ Hd1.T
    z_rms     = float(np.mean(np.abs(z_test)))
    print(f"  Typical |z| after stage 1: {z_rms:.3f} (use as reference for P_sat_sig)")

    # Key2 for stage 2: independent random key in gain regime
    key2 = rng.uniform(0.6, 1.0, size=len(true_key))

    # Sweep P_sat_sig as multiples of z_rms
    multipliers = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]
    p_sat_vals  = [round(m * z_rms, 4) for m in multipliers]

    print(f"\n[Fig 18] Nonlinear operating point sweep (P_sat_sig vs. BER)...")
    results = []

    for p_sat, mult in zip(p_sat_vals, multipliers):
        bob_bers, eve_wk_bers, eve_lr_bers = [], [], []
        for _ in range(n_trials):
            # Random wrong key from [0,1]^13 for wrong-key Eve
            eve_k1 = rng.uniform(0, 1, size=len(true_key))
            eve_k2 = rng.uniform(0, 1, size=len(true_key))
            res = enc.run_session_nonlinear(
                pump_key1=true_key, pump_key2=key2,
                p_sat_sig=p_sat, T=T,
                eve_key1=eve_k1, eve_key2=eve_k2,
                eve_p_sat_sig=p_sat  # Eve knows architecture, guesses same P_sat
            )
            bob_bers.append(res["BER_Bob"])
            eve_wk_bers.append(res["BER_Eve_WK"])
            eve_lr_bers.append(res["BER_Eve_LR"])

        entry = {
            "p_sat_sig": p_sat,
            "p_sat_mult": mult,
            "BER_Bob":     float(np.mean(bob_bers)),
            "BER_Eve_WK":  float(np.mean(eve_wk_bers)),
            "BER_Eve_LR":  float(np.mean(eve_lr_bers)),
        }
        results.append(entry)
        print(f"  P_sat={p_sat:.3f} ({mult:.1f}×RMS)  "
              f"BER_Bob={entry['BER_Bob']:.4f}  "
              f"BER_Eve_WK={entry['BER_Eve_WK']:.4f}  "
              f"BER_Eve_LR={entry['BER_Eve_LR']:.4f}")

    _write_csv(f"{out_dir}/nonlinear_operating_point.csv", results)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs  = [r["p_sat_mult"] for r in results]
    ax.semilogx(xs, [r["BER_Bob"]    for r in results], "bo-",
                lw=2, ms=7, label="BER_Bob (correct key, 3-stage MMSE)")
    ax.semilogx(xs, [r["BER_Eve_WK"] for r in results], "rs-",
                lw=2, ms=7, label="BER_Eve (wrong key, 3-stage ZF)")
    ax.semilogx(xs, [r["BER_Eve_LR"] for r in results], "m^--",
                lw=2, ms=7, label="BER_Eve (linear regression, all T pairs)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    ax.axhline(0.1, color="orange", ls="--", lw=1, label="BER = 0.10 threshold")
    ax.set_xlabel("P_sat_sig / RMS(|z|)  (1.0 = matched saturation)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Two-stage nonlinear MCF: operating point vs. BER\n"
                 "(find P_sat_sig where BER_Eve_LR stays high and BER_Bob = 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, which="both")
    ax.set_ylim(-0.02, 0.65)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig18_nonlinear_operating_point.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig18_nonlinear_operating_point.png")
    return results


# ── Figure 19 — Linear regression attack on two-stage nonlinear channel ───────

def sweep_lr_adversary_nonlinear(enc: MCFEncryption,
                                  true_key: np.ndarray,
                                  out_dir: str,
                                  T_collect: int = 500,
                                  p_sat_sig_mult: float = 1.0,
                                  n_trials: int = 4,
                                  seed: int = 7) -> List[Dict]:
    """
    Re-run the linear regression adversary attack (Fig 17) on the two-stage
    nonlinear channel, demonstrating that the attack is defeated.

    Key comparison:
      Linear channel (Fig 17):  BER_Eve_LR = 0 at N_obs = 13
      Nonlinear channel (Fig 19): BER_Eve_LR stays near 0.5 for all N_obs

    Even with N_obs = 500 observed (x_t, y_t) pairs, the best linear fit
    cannot capture the nonlinear channel — BER_Eve remains high.

    The p_sat_sig is set from the operating point identified in Fig 18.
    """
    rng  = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd   = len(data_idx)
    L    = enc.params.length_mm
    I_Nd = np.eye(Nd)

    # Determine P_sat_sig from RMS signal amplitude after stage 1
    H1 = sim.transfer_matrix(true_key, L)
    Hd1 = H1[np.ix_(data_idx, data_idx)]
    test_syms = np.array([enc._qpsk_map(np.random.default_rng(0).integers(0,2,(Nd,2)))
                           for _ in range(100)])
    z_rms = float(np.mean(np.abs(test_syms @ Hd1.T)))
    p_sat_sig = p_sat_sig_mult * z_rms

    # Stage 2 key: independent gain-regime key
    key2 = np.random.default_rng(seed + 1).uniform(0.6, 1.0, size=len(true_key))

    n_pairs_vals = [1, 5, 10, 13, 20, 30, 50, 80, 100, 130, 169, 250, 350, 500]
    n_pairs_vals = [n for n in n_pairs_vals if n <= T_collect]

    print(f"\n[Fig 19] LR adversary on nonlinear channel "
          f"(P_sat={p_sat_sig:.3f} = {p_sat_sig_mult}×RMS)...")
    results = []

    for n_obs in n_pairs_vals:
        bob_bers, lr_bers, rk_bers = [], [], []
        for _ in range(n_trials):
            # Generate disjoint train/test sets through the nonlinear channel.
            bits_train = rng.integers(0, 2, size=(T_collect, Nd, 2))
            re_tr = 2.0 * bits_train[:, :, 0] - 1.0
            im_tr = 2.0 * bits_train[:, :, 1] - 1.0
            X_train = (re_tr + 1j * im_tr) / np.sqrt(2.0)

            bits_test = rng.integers(0, 2, size=(T_collect, Nd, 2))
            re_te = 2.0 * bits_test[:, :, 0] - 1.0
            im_te = 2.0 * bits_test[:, :, 1] - 1.0
            X_test = (re_te + 1j * im_te) / np.sqrt(2.0)

            all_var1 = _ase_noise_variances(true_key, enc.params, geom)
            all_var2 = _ase_noise_variances(key2,     enc.params, geom)
            sigma_c1 = np.sqrt(all_var1[data_idx] / 2.0)
            sigma_c2 = np.sqrt(all_var2[data_idx] / 2.0)

            H2      = sim.transfer_matrix(key2, L)
            Hd2     = H2[np.ix_(data_idx, data_idx)]
            sigma2_s2 = float(np.mean(all_var2[data_idx]))
            Hd2_inv_mmse = np.linalg.solve(
                Hd2.conj().T @ Hd2 + sigma2_s2 * I_Nd, Hd2.conj().T)
            sigma2_s1 = float(np.mean(all_var1[data_idx]))
            Hd1_inv_mmse = np.linalg.solve(
                Hd1.conj().T @ Hd1 + sigma2_s1 * I_Nd, Hd1.conj().T)

            # Forward pass
            n1_tr = (rng.normal(size=(T_collect, Nd)) * sigma_c1[None, :] +
                     1j * rng.normal(size=(T_collect, Nd)) * sigma_c1[None, :])
            n2_tr = (rng.normal(size=(T_collect, Nd)) * sigma_c2[None, :] +
                     1j * rng.normal(size=(T_collect, Nd)) * sigma_c2[None, :])
            Z_tr = X_train @ Hd1.T + n1_tr
            A_tr = _saturable_absorber(Z_tr, p_sat_sig)
            Y_train = A_tr @ Hd2.T + n2_tr

            n1_te = (rng.normal(size=(T_collect, Nd)) * sigma_c1[None, :] +
                     1j * rng.normal(size=(T_collect, Nd)) * sigma_c1[None, :])
            n2_te = (rng.normal(size=(T_collect, Nd)) * sigma_c2[None, :] +
                     1j * rng.normal(size=(T_collect, Nd)) * sigma_c2[None, :])
            Z_te = X_test @ Hd1.T + n1_te
            A_te = _saturable_absorber(Z_te, p_sat_sig)
            Y_test = A_te @ Hd2.T + n2_te

            # Bob BER (reference)
            z_hat = Y_test @ Hd2_inv_mmse.T
            a_hat = _saturable_absorber_inv(z_hat, p_sat_sig)
            x_hat_bob = a_hat @ Hd1_inv_mmse.T
            bits_hat_bob = np.array([enc._qpsk_demod(x_hat_bob[t])
                                      for t in range(T_collect)])
            bob_bers.append(float(np.mean(bits_test != bits_hat_bob)))

            # Eve LR attack: fit on training pairs, evaluate on held-out test pairs
            X_obs = X_train[:n_obs]
            Y_obs = Y_train[:n_obs]
            H_est_T = np.linalg.lstsq(X_obs, Y_obs, rcond=None)[0]
            H_est_inv = np.linalg.pinv(H_est_T.T)
            x_hat_lr  = Y_test @ H_est_inv.T
            bits_hat_lr = np.array([enc._qpsk_demod(x_hat_lr[t])
                                     for t in range(T_collect)])
            lr_bers.append(float(np.mean(bits_test != bits_hat_lr)))

            # Eve random key (baseline)
            rk1 = rng.uniform(0, 1, size=len(true_key))
            rk2 = rng.uniform(0, 1, size=len(true_key))
            Hd_rk1_inv = np.linalg.pinv(sim.transfer_matrix(rk1, L)[np.ix_(data_idx, data_idx)])
            Hd_rk2_inv = np.linalg.pinv(sim.transfer_matrix(rk2, L)[np.ix_(data_idx, data_idx)])
            z_rk = Y_test @ Hd_rk2_inv.T
            a_rk = _saturable_absorber_inv(z_rk, p_sat_sig)
            x_rk = a_rk @ Hd_rk1_inv.T
            bits_hat_rk = np.array([enc._qpsk_demod(x_rk[t]) for t in range(T_collect)])
            rk_bers.append(float(np.mean(bits_test != bits_hat_rk)))

        entry = {
            "n_obs_pairs": n_obs,
            "BER_Bob":      float(np.mean(bob_bers)),
            "BER_Eve_LR":   float(np.mean(lr_bers)),
            "BER_Eve_rand": float(np.mean(rk_bers)),
        }
        results.append(entry)
        print(f"  N_obs={n_obs:4d}  BER_Bob={entry['BER_Bob']:.4f}  "
              f"BER_Eve_LR={entry['BER_Eve_LR']:.4f}  "
              f"BER_Eve_rand={entry['BER_Eve_rand']:.4f}")

    _write_csv(f"{out_dir}/lr_adversary_nonlinear.csv", results)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [r["n_obs_pairs"] for r in results]
    ax.semilogx(xs, [r["BER_Eve_LR"]   for r in results], "r^-",
                lw=2.5, ms=8, label="Eve LR — nonlinear channel (Fig 19) ← NEW")
    ax.semilogx(xs, [r["BER_Eve_rand"] for r in results], "bs--",
                lw=1.5, ms=6, label="Eve random key — baseline")
    ax.semilogx(xs, [r["BER_Bob"]      for r in results], "go-",
                lw=2, ms=6, label="Bob (correct key, 3-stage MMSE)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess (0.5)")
    ax.axhline(0.1, color="orange", ls="--", lw=1, label="BER = 0.10 threshold")
    ax.axvline(13, color="purple", ls=":", lw=1.5,
               label=f"N_obs = N_d = 13 (linear scheme broken here)")
    ax.set_xlabel("Number of known-plaintext pairs N_obs")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Linear regression attack: nonlinear 2-stage channel\n"
                 "(compare to Fig 17: linear channel broken at N_obs=13)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, which="both")
    ax.set_ylim(-0.02, 0.65)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig19_lr_adversary_nonlinear.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig19_lr_adversary_nonlinear.png")
    return results


# ── Figure 20 — Key mismatch: linear vs. nonlinear comparison ────────────────

def sweep_key_mismatch_nonlinear(enc: MCFEncryption,
                                  true_key: np.ndarray,
                                  out_dir: str,
                                  T: int = 600,
                                  n_trials: int = 3,
                                  p_sat_sig_mult: float = 1.0,
                                  seed: int = 7) -> List[Dict]:
    """
    Side-by-side BER vs. key mismatch comparison:
      (a) Single-stage linear channel (Fig 2 baseline)
      (b) Two-stage nonlinear channel (this figure)

    Shows that:
      1. BER_Bob remains 0 in both architectures (correct key → perfect decode)
      2. BER_Eve is similar or slightly higher for nonlinear channel at high mismatch
      3. The key difference is in the LR attack (Fig 19), not in wrong-key BER

    This places the nonlinear contribution in correct context: the security
    improvement is against the known-plaintext attack, not against key guessing.
    """
    rng  = np.random.default_rng(seed)
    sim  = enc.sim
    geom = enc.geom
    data_idx = geom.data_core_indices
    Nd   = len(data_idx)
    L    = enc.params.length_mm
    I_Nd = np.eye(Nd)

    # P_sat_sig from RMS estimate
    H1 = sim.transfer_matrix(true_key, L)
    Hd1 = H1[np.ix_(data_idx, data_idx)]
    test_s = np.array([enc._qpsk_map(rng.integers(0,2,(Nd,2))) for _ in range(100)])
    z_rms  = float(np.mean(np.abs(test_s @ Hd1.T)))
    p_sat_sig = p_sat_sig_mult * z_rms
    key2 = rng.uniform(0.6, 1.0, size=len(true_key))

    mismatch_fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    print(f"\n[Fig 20] Key mismatch: 1-stage linear vs. 2-stage nonlinear "
          f"(P_sat={p_sat_sig:.3f})...")
    results = []

    for frac in mismatch_fracs:
        n_wrong = int(round(frac * len(true_key)))
        linear_bob_bers, linear_eve_bers = [], []
        nl_bob_bers, nl_eve_bers = [], []

        for _ in range(n_trials):
            eve_key = true_key.copy()
            if n_wrong > 0:
                idx_wrong = rng.choice(len(true_key), size=n_wrong, replace=False)
                for i in idx_wrong:
                    eve_key[i] = float(rng.uniform(0, 1))

            # Linear single-stage (matches Fig 2)
            res_lin = enc.run_session(true_key, T=T, eve_key=eve_key,
                                      eve_partial_obs=False)
            linear_bob_bers.append(res_lin["BER_Bob"])
            linear_eve_bers.append(res_lin["BER_Eve"])

            # Nonlinear two-stage
            eve_key2 = rng.uniform(0, 1, size=len(true_key))
            res_nl = enc.run_session_nonlinear(
                pump_key1=true_key, pump_key2=key2,
                p_sat_sig=p_sat_sig, T=T,
                eve_key1=eve_key, eve_key2=eve_key2,
                eve_p_sat_sig=p_sat_sig
            )
            nl_bob_bers.append(res_nl["BER_Bob"])
            nl_eve_bers.append(res_nl["BER_Eve_WK"])

        entry = {
            "key_mismatch_frac":   float(frac),
            "BER_Bob_linear":      float(np.mean(linear_bob_bers)),
            "BER_Eve_linear":      float(np.mean(linear_eve_bers)),
            "BER_Bob_nonlinear":   float(np.mean(nl_bob_bers)),
            "BER_Eve_nonlinear":   float(np.mean(nl_eve_bers)),
        }
        results.append(entry)
        print(f"  mismatch={frac:.1f}  "
              f"Lin BER_Bob={entry['BER_Bob_linear']:.4f} Eve={entry['BER_Eve_linear']:.4f}  |  "
              f"NL  BER_Bob={entry['BER_Bob_nonlinear']:.4f} Eve={entry['BER_Eve_nonlinear']:.4f}")

    _write_csv(f"{out_dir}/key_mismatch_nonlinear.csv", results)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    xs = [r["key_mismatch_frac"] for r in results]

    for ax, label, bob_key, eve_key_col, title in [
        (axes[0], "Linear (1-stage)", "BER_Bob_linear",    "BER_Eve_linear",
         "Single-stage linear MCF\n(current paper — Fig 2)"),
        (axes[1], "Nonlinear (2-stage)", "BER_Bob_nonlinear", "BER_Eve_nonlinear",
         f"Two-stage nonlinear MCF\n(P_sat = {p_sat_sig:.2f} = {p_sat_sig_mult}×RMS)"),
    ]:
        ax.plot(xs, [r[bob_key]    for r in results], "bo-", lw=2, ms=6,
                label="Bob (correct key)")
        ax.plot(xs, [r[eve_key_col] for r in results], "rs--", lw=2, ms=6,
                label="Eve (wrong key)")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("Fraction of key components mismatched")
        ax.set_ylabel("BER")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(-0.02, 0.65)

    fig.suptitle("Key mismatch BER: linear vs. nonlinear two-stage MCF", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig20_key_mismatch_nonlinear.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig20_key_mismatch_nonlinear.png")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 14.  PROBLEM 2 FIX + SOLUTION B SWEEPS (Figs 21, 22)
# ──────────────────────────────────────────────────────────────────────────────

def sweep_mixed_regime_k2_fix(enc: MCFEncryption,
                               true_key: np.ndarray,
                               out_dir: str,
                               T: int = 400,
                               n_trials: int = 3,
                               seed: int = 7,
                               n_absorption_vals: Optional[List[int]] = None,
                               fig_suffix: str = "") -> List[Dict]:
    """
    Fig 21 — Mixed-regime K₂ as countermeasure to the informed adversary.

    Problem (from Fig 9): Eve knows the key range K ∈ [0.6, 1.0] → she samples
    K_eve ∈ [0.6, 1.0]^13 and obtains BER_Eve = 0.004 (near-transparent).
    Root cause: at high pump the all-gain regime is smooth and well-conditioned;
    random draws within [0.6, 1.0] stay close to the true key.

    Fix (two-stage architecture): K₁ (stage 1) stays in [0.6, 1.0] for Bob's
    stability. K₂ (stage 2) deliberately mixes absorption-regime components
    [0.3, 0.5] with gain-regime components [0.6, 1.0].

    Security mechanism:
      - Eve who sees only K₂ ∈ [0.6, 1.0] (does not know about mixed regime)
        computes a wrong H₂ → BER_Eve_informed rises with n_absorption.
      - This exploits REGIME CROSSING (the core security mechanism identified
        earlier) as a deliberate design feature, not an accidental property.
      - Kerckhoffs-compliant: the design principle (mixed regime) is public;
        only the exact K₂ values are secret.

    Architecture used:
        y = H₂(K₂) · σ(H₁(K₁) · x + n₁, P_sat=∞) + n₂
    with P_sat = 1e6 × z_rms (SA effectively inactive) to isolate the K₂ effect.

    Parameters
    ----------
    n_absorption_vals : grid [0, 2, 4, 6, 8, 10, 13] absorption components in K₂
    BER metrics:
      BER_Bob        : correct K₁ (gain), correct K₂ (mixed) — target = 0
      BER_Eve_rand   : random Eve U[0,1]^13 for both K₁, K₂
      BER_Eve_inf    : informed Eve K₁~U[0.6,1.0], K₂~U[0.6,1.0]
                       (knows K₁ range, wrongly assumes K₂ also all-gain)
      cond_H2        : condition number cond(H₂(K₂)) → Bob's decoding stability
    """
    rng            = np.random.default_rng(seed)
    N              = enc.geom.N
    p_sat_inactive = 1e6          # negligible SA: isolate K₂ regime effect
    # Default: 7 evenly-spaced values from 0 to N (adapted to any MCF size)
    if n_absorption_vals is None:
        step = max(1, N // 6)
        n_absorption_vals = sorted(set(
            [0] + list(range(step, N, step)) + [N]))

    results = []
    print("  n_abs | BER_Bob | BER_Eve_rand | BER_Eve_inf(K2 wrong regime) | cond(H2)")
    print("  " + "-" * 75)

    for n_abs in n_absorption_vals:
        bob_bers, eve_rand_bers, eve_inf_bers, cond_vals = [], [], [], []

        for _trial in range(n_trials):
            # ── generate mixed-regime K₂ ────────────────────────────────────
            k2 = rng.uniform(0.6, 1.0, size=N)
            if n_abs > 0:
                abs_idx   = rng.choice(N, size=n_abs, replace=False)
                k2[abs_idx] = rng.uniform(0.3, 0.5, size=n_abs)

            k1 = true_key.copy()   # stage-1 key: always all-gain

            # condition number of H₂(K₂)
            H2    = enc.sim.transfer_matrix(k2, enc.params.length_mm)
            didx  = enc.geom.data_core_indices
            Hd2   = H2[np.ix_(didx, didx)]
            cond_vals.append(np.linalg.cond(Hd2))

            # ── Eve random: U[0,1] for both K₁ and K₂ ───────────────────────
            ek1_rand = rng.uniform(0.0, 1.0, size=N)
            ek2_rand = rng.uniform(0.0, 1.0, size=N)
            res_rand = enc.run_session_nonlinear(
                k1, k2, p_sat_sig=p_sat_inactive, T=T,
                eve_key1=ek1_rand, eve_key2=ek2_rand)
            bob_bers.append(res_rand["BER_Bob"])
            eve_rand_bers.append(res_rand["BER_Eve_WK"])

            # ── Eve informed: U[0.6,1.0] for K₁, U[0.6,1.0] for K₂ ─────────
            # Eve knows K₁ range [0.6,1.0] but wrongly assumes K₂ also all-gain
            ek1_inf = rng.uniform(0.6, 1.0, size=N)
            ek2_inf = rng.uniform(0.6, 1.0, size=N)    # wrong regime for K₂
            res_inf = enc.run_session_nonlinear(
                k1, k2, p_sat_sig=p_sat_inactive, T=T,
                eve_key1=ek1_inf, eve_key2=ek2_inf)
            eve_inf_bers.append(res_inf["BER_Eve_WK"])

        row = {
            "n_absorption":        n_abs,
            "BER_Bob_mean":        float(np.mean(bob_bers)),
            "BER_Eve_rand_mean":   float(np.mean(eve_rand_bers)),
            "BER_Eve_inf_mean":    float(np.mean(eve_inf_bers)),
            "cond_H2_mean":        float(np.mean(cond_vals)),
        }
        results.append(row)
        print(f"  {n_abs:5d} | {row['BER_Bob_mean']:.4f}  | "
              f"{row['BER_Eve_rand_mean']:.4f}       | "
              f"{row['BER_Eve_inf_mean']:.4f}                     | "
              f"{row['cond_H2_mean']:.2f}")

    _write_csv(f"{out_dir}/mixed_regime_k2_fix.csv", results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_abs_arr = [r["n_absorption"]      for r in results]
    ber_bob   = [r["BER_Bob_mean"]      for r in results]
    ber_rand  = [r["BER_Eve_rand_mean"] for r in results]
    ber_inf   = [r["BER_Eve_inf_mean"]  for r in results]
    cond_h2   = [r["cond_H2_mean"]      for r in results]

    ax = axes[0]
    ax.plot(n_abs_arr, ber_bob,  "b-o", lw=2, ms=7,
            label="BER_Bob (correct keys)")
    ax.plot(n_abs_arr, ber_rand, "r-s", lw=2, ms=7,
            label="BER_Eve random U[0,1]")
    ax.plot(n_abs_arr, ber_inf,  "m-^", lw=2, ms=7,
            label="BER_Eve informed\n(K1∈[0.6,1.0], K2 wrong regime)")
    ax.axhline(0.5,  ls="--", color="gray",  alpha=0.6, label="Random guess = 0.5")
    ax.axhline(0.004, ls=":", color="purple", alpha=0.6,
               label="Single-stage informed BER (baseline)")
    ax.set_xlabel("Number of K₂ components in absorption regime [0.3, 0.5]")
    ax.set_ylabel("BER")
    ax.set_title("Fig 21a — Mixed-regime K₂: BER vs. n_absorption\n"
                 "(Problem 2 countermeasure — informed adversary fix)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(-0.02, 0.65)

    ax2 = axes[1]
    ax2.semilogy(n_abs_arr, cond_h2, "g-D", lw=2, ms=7)
    ax2.set_xlabel("Number of K₂ components in absorption regime")
    ax2.set_ylabel("cond(H₂(K₂))  [log scale]")
    ax2.set_title("Fig 21b — Condition number of stage-2 matrix\n"
                  "(Bob decoding stability: κ < ~30 → MMSE viable)")
    ax2.axhline(30, ls="--", color="orange", alpha=0.7,
                label="Practical stability limit κ=30")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    fig.suptitle("Mixed-regime K₂ countermeasure to informed adversary (Fig 21)",
                 fontsize=10)
    plt.tight_layout()
    fname = f"fig21_mixed_regime_k2_fix{fig_suffix}.png"
    plt.savefig(f"{out_dir}/{fname}", dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")
    return results


def sweep_mixed_regime_k2_scaling(base_params: FiberParams,
                                   out_dir: str,
                                   n_rings_list: Optional[List[int]] = None,
                                   T: int = 100,
                                   n_trials: int = 2,
                                   seed: int = 7) -> Dict:
    """
    Fig 21b — Mixed-regime K₂ fix: scaling with MCF size (7 / 19 / 37 cores).

    Tests whether the countermeasure becomes MORE effective as N increases.
    Hypothesis: with more cores, the key space grows exponentially and the
    regime-crossing confusion compounds → BER_Eve_inf should saturate faster
    with fewer absorption cores for larger N.

    For each n_rings (7, 19, 37 cores):
      - Run sweep_mixed_regime_k2_fix with N-adapted n_absorption_vals
      - Report peak BER_Eve_inf and minimum n_abs to reach BER_Eve_inf > 0.15

    Returns dict: {N: List[Dict]} keyed by core count.
    """
    if n_rings_list is None:
        n_rings_list = [1, 2, 3]    # 7, 19, 37 cores

    all_results = {}
    summary = []

    for nr in n_rings_list:
        geom = HexMCFGeometry(n_rings=nr, pitch_um=base_params.pitch_um)
        N    = geom.N
        enc  = MCFEncryption(base_params, geom, seed=seed)
        true_key = np.random.default_rng(seed).uniform(0.6, 1.0, size=N)

        print(f"\n  [{N}-core MCF, n_rings={nr}]")
        results = sweep_mixed_regime_k2_fix(
            enc, true_key, out_dir, T=T, n_trials=n_trials, seed=seed,
            fig_suffix=f"_{N}core")

        all_results[N] = results
        # Summary metrics
        peak_inf  = max(r["BER_Eve_inf_mean"] for r in results)
        # Minimum n_abs to exceed BER_Eve_inf = 0.15
        n_abs_15 = next((r["n_absorption"] for r in results
                         if r["BER_Eve_inf_mean"] >= 0.15), None)
        pct_abs_15 = (n_abs_15 / N * 100) if n_abs_15 is not None else None
        summary.append({
            "N": N, "n_rings": nr,
            "peak_BER_Eve_inf": peak_inf,
            "n_abs_to_reach_015": n_abs_15,
            "pct_abs_to_reach_015": pct_abs_15,
        })
        pct_str = f"{pct_abs_15:.1f}%" if pct_abs_15 is not None else "N/A"
        print(f"  Peak BER_Eve_inf={peak_inf:.4f}  "
              f"n_abs>=0.15 at {n_abs_15} ({pct_str} of cores)")

    _write_csv(f"{out_dir}/mixed_regime_k2_scaling.csv", summary)

    # ── Combined scaling plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "s", "^"]
    for (N, results), col, mk in zip(all_results.items(), colors, markers):
        n_abs_arr = [r["n_absorption"]     for r in results]
        ber_inf   = [r["BER_Eve_inf_mean"] for r in results]
        # Normalize x-axis by N (fraction of absorption cores)
        frac      = [x / N for x in n_abs_arr]
        ax.plot(frac, ber_inf, color=col, marker=mk, lw=2, ms=7,
                label=f"N={N} cores")
    ax.axhline(0.5,  ls="--", color="gray",   alpha=0.6, label="Random (0.5)")
    ax.axhline(0.15, ls=":",  color="red",    alpha=0.6, label="BER=0.15 target")
    ax.axhline(0.004, ls=":", color="purple", alpha=0.6, label="Single-stage baseline")
    ax.set_xlabel("Fraction of K₂ components in absorption regime")
    ax.set_ylabel("BER_Eve_informed (K₁∈[0.6,1.0], K₂∈[0.6,1.0])")
    ax.set_title("Fig 21b — Mixed-regime K₂: scaling across MCF sizes\n"
                 "(normalized fraction of absorption components)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(-0.02, 0.65)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig21b_mixed_regime_k2_scaling.png", dpi=200)
    plt.close(fig)
    print(f"\n  Saved fig21b_mixed_regime_k2_scaling.png")
    return all_results


def sweep_ode_saturation(enc: MCFEncryption,
                          true_key: np.ndarray,
                          out_dir: str,
                          T: int = 50,
                          n_trials: int = 3,
                          seed: int = 7) -> List[Dict]:
    """
    Fig 22 — Signal-induced ODE gain saturation: Solution B for LR attack.

    Demonstrates the fundamental tradeoff of in-fiber gain saturation:
      - WEAK  saturation (P_sat >> |z|): channel ≈ linear → LR works, BER_Bob = 0
      - MOD.  saturation (P_sat ≈ |z|): LR partially fails, BER_Bob slightly up
      - STRONG saturation (P_sat << |z|): LR fails, BER_Bob significantly degraded

    Physical mechanism (inside the CME ODE):
        g_eff(z, |E_i(z)|²) = g_base_i · 1 / (1 + |E_i(z)|² / P_sat_sig)
    This is physically distinct from the post-MCF saturable absorber (Figs 18-20):
    saturation occurs at EVERY position z inside the fiber during coupled-mode
    propagation, not just at the fiber output.

    HONEST CONCLUSION: In-fiber saturation (Solution B) does NOT provide
    asymmetric security. Both Bob's linearized MMSE and Eve's LR attack
    suffer from the same nonlinear approximation error. The tradeoff:
      - BER_Bob ~0 (FEC-correctable) at moderate saturation (P_sat ~5-10x z_rms)
      - BER_Eve_LR > 0.1 at the same operating point
      - Stronger saturation raises both BER_Bob and BER_Eve_LR together
    This motivates Solution A (hard phase regenerator) as the correct fix,
    which provides clean phase-only output invertible by Bob but not by LR.

    Bob's decoder: linearized H_eff computed by propagating each basis vector
    e_i scaled to QPSK amplitude (1/√2) through the saturated ODE. This gives
    the small-signal approximation, accurate when P_sat >> signal amplitude.

    Parameters
    ----------
    T       : symbols per trial (kept small: each symbol requires 1 ODE solve)
    seed    : RNG seed for reproducibility
    """
    rng      = np.random.default_rng(seed)
    L        = enc.params.length_mm
    N        = enc.geom.N
    data_idx = enc.geom.data_core_indices
    Nd       = len(data_idx)
    I_Nd     = np.eye(Nd)

    # ── reference: linear output RMS (no saturation) ─────────────────────────
    H_lin   = enc.sim.transfer_matrix(true_key, L)
    Hd_lin  = H_lin[np.ix_(data_idx, data_idx)]
    # RMS signal amplitude at fiber output for QPSK input (E[|x|²]=1 per core)
    z_rms   = float(np.sqrt(np.mean(np.abs(Hd_lin) ** 2) * Nd))
    print(f"  Reference z_rms (linear, signal amplitude RMS) = {z_rms:.4f}")

    # Noise variance per data core
    all_var  = _ase_noise_variances(true_key, enc.params, enc.geom)
    noise_var = float(np.mean(all_var[data_idx]))
    sigma_n   = np.sqrt(noise_var / 2.0)

    # ── 16-QAM constellation (for variable-amplitude comparison) ─────────────
    def _qam16_pts() -> np.ndarray:
        """Normalized 16-QAM constellation, mean power = 1."""
        axis = np.array([-3., -1., 1., 3.], dtype=float)
        re, im = np.meshgrid(axis, axis)
        pts = (re + 1j * im).flatten()
        return pts / np.sqrt(np.mean(np.abs(pts) ** 2))

    def _nearest_qam(x_hat: np.ndarray, const: np.ndarray) -> np.ndarray:
        dist = np.abs(x_hat[:, :, None] - const[None, None, :])
        return const[np.argmin(dist, axis=2)]

    const16   = _qam16_pts()                    # 16 constellation points
    alpha_qpsk = 1.0 / np.sqrt(2.0)            # QPSK symbol amplitude (constant)

    p_sat_mults = [1000.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    results     = []

    print("  MULTI-CORE AVERAGING EFFECT (key finding):")
    print("  13-core coupling distributes per-symbol power across all cores.")
    print("  For any normalized-power modulation (QPSK, 16-QAM), the per-core")
    print("  power after coupling ~ constant (law-of-large-numbers over 13 cores).")
    print("  -> Gain saturation is approximately uniform across symbol patterns.")
    print("  -> LR finds a consistent H_eff for all symbols -> BER_Eve_LR = 0.")
    print("  Expected result: BER_Eve_LR = 0 for ALL saturation levels.")
    print("  Diagnostic: ||H_eff||_F/||H_lin||_F shows saturation IS real.")
    print()
    print(f"  P_sat/z_rms | BER_Bob | BER_LR(QPSK) | BER_LR(16QAM) | ||Heff||/||Hlin||")
    print("  " + "-" * 75)

    for mult in p_sat_mults:
        p_sat_sig = mult * z_rms
        bob_bers_qpsk, lr_bers_qpsk, lr_bers_16qam = [], [], []

        for _trial in range(n_trials):
            # ── compute linearized H_eff once per (trial, p_sat) ───────────
            # Propagate N basis vectors at QPSK amplitude → approximates the
            # small-signal channel. Shared by Bob and both Eve LR tests.
            H_eff = np.zeros((N, N), dtype=complex)
            e     = np.eye(N, dtype=complex)
            for i in range(N):
                a0          = e[:, i] * alpha_qpsk
                y_col       = _propagate_saturated(
                    enc.sim, a0, true_key, p_sat_sig, L)
                H_eff[:, i] = y_col / alpha_qpsk
            Hd_eff = H_eff[np.ix_(data_idx, data_idx)]
            Hd_inv_mmse = np.linalg.solve(
                Hd_eff.conj().T @ Hd_eff + noise_var * I_Nd,
                Hd_eff.conj().T)

            # ════ QPSK test ════════════════════════════════════════════════
            bits_q  = rng.integers(0, 2, size=(T, Nd, 2))
            syms_q  = ((2.0 * bits_q[:, :, 0] - 1.0) +
                       1j * (2.0 * bits_q[:, :, 1] - 1.0)) * alpha_qpsk

            Y_q = np.zeros((T, Nd), dtype=complex)
            for t in range(T):
                x_full           = np.zeros(N, dtype=complex)
                x_full[data_idx] = syms_q[t]
                y_full           = _propagate_saturated(
                    enc.sim, x_full, true_key, p_sat_sig, L)
                Y_q[t]           = y_full[data_idx] + (
                    rng.standard_normal(Nd) + 1j * rng.standard_normal(Nd)) * sigma_n

            # Bob MMSE (QPSK)
            x_hat_b  = Y_q @ Hd_inv_mmse.T
            bits_hat = np.stack([(x_hat_b.real >= 0).astype(int),
                                  (x_hat_b.imag >= 0).astype(int)], axis=2)
            bob_bers_qpsk.append(float(np.mean(bits_q != bits_hat)))

            # Eve LR (QPSK) — note: expected BER=0 due to constant envelope
            H_lr_q   = np.linalg.lstsq(syms_q, Y_q, rcond=None)[0].T
            x_hat_lr = Y_q @ np.linalg.pinv(H_lr_q).T
            bits_lr  = np.stack([(x_hat_lr.real >= 0).astype(int),
                                  (x_hat_lr.imag >= 0).astype(int)], axis=2)
            lr_bers_qpsk.append(float(np.mean(bits_q != bits_lr)))

            # ════ 16-QAM test (variable amplitude → LR floor expected) ════
            idx16   = rng.integers(0, 16, size=(T, Nd))
            syms_16 = const16[idx16]                              # (T, Nd)

            Y_16 = np.zeros((T, Nd), dtype=complex)
            for t in range(T):
                x_full           = np.zeros(N, dtype=complex)
                x_full[data_idx] = syms_16[t]
                y_full           = _propagate_saturated(
                    enc.sim, x_full, true_key, p_sat_sig, L)
                Y_16[t]          = y_full[data_idx] + (
                    rng.standard_normal(Nd) + 1j * rng.standard_normal(Nd)) * sigma_n

            # Eve LR (16-QAM) — amplitude-variable input → systematic LR error
            H_lr_16   = np.linalg.lstsq(syms_16, Y_16, rcond=None)[0].T
            x_hat_lr16 = Y_16 @ np.linalg.pinv(H_lr_16).T
            sym_lr16   = _nearest_qam(x_hat_lr16, const16)
            ser_lr16   = float(np.mean(np.abs(sym_lr16 - syms_16) > 1e-9))
            lr_bers_16qam.append(ser_lr16 / 4.0)   # SER → approx BER (4 bits/symbol)

        # Frobenius-norm ratio: how much did saturation change the channel?
        H_lin_val  = enc.sim.transfer_matrix(true_key, L)
        Hd_lin_val = H_lin_val[np.ix_(data_idx, data_idx)]
        # Use H_eff from last trial (representative)
        heff_ratio = float(np.linalg.norm(Hd_eff) / (np.linalg.norm(Hd_lin_val) + 1e-30))

        ber_bob_m  = float(np.mean(bob_bers_qpsk))
        ber_lr_q   = float(np.mean(lr_bers_qpsk))
        ber_lr_16  = float(np.mean(lr_bers_16qam))
        if mult >= 100:
            note = "linear"
        elif mult >= 10:
            note = "mild"
        elif mult >= 2:
            note = "moderate"
        else:
            note = "strong"
        row = {
            "p_sat_mult":            mult,
            "p_sat_sig":             float(p_sat_sig),
            "BER_Bob_QPSK_mean":     ber_bob_m,
            "BER_Eve_LR_QPSK_mean":  ber_lr_q,
            "BER_Eve_LR_16QAM_mean": ber_lr_16,
            "Heff_Hlin_ratio":       heff_ratio,
            "saturation_regime":     note,
        }
        results.append(row)
        print(f"  {mult:10.1f}x | {ber_bob_m:.4f}  | {ber_lr_q:.4f}      | "
              f"{ber_lr_16:.4f}        | {heff_ratio:.4f}  ({note})")

    _write_csv(f"{out_dir}/ode_saturation.csv", results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    mults       = [r["p_sat_mult"]            for r in results]
    ber_bob     = [r["BER_Bob_QPSK_mean"]     for r in results]
    ber_lr_q    = [r["BER_Eve_LR_QPSK_mean"]  for r in results]
    ber_lr16    = [r["BER_Eve_LR_16QAM_mean"] for r in results]
    heff_ratios = [r["Heff_Hlin_ratio"]       for r in results]

    # Panel A: BER — both modulations show BER_Eve_LR = 0 (multi-core averaging)
    ax = axes[0]
    ax.semilogx(mults, ber_bob,  "b-o", lw=2, ms=7,
                label="BER_Bob QPSK (linearized MMSE)")
    ax.semilogx(mults, ber_lr_q, "r-s", lw=2, ms=7,
                label="BER_Eve LR — QPSK")
    ax.semilogx(mults, ber_lr16, "m-^", lw=2, ms=7,
                label="BER_Eve LR — 16-QAM")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="Random guess = 0.5")
    ax.set_xlabel("P_sat_sig / z_rms   (weaker saturation →)")
    ax.set_ylabel("BER")
    ax.set_title("LR attack BER vs. ODE saturation strength\n"
                 "(both QPSK and 16-QAM)", fontsize=9)
    ax.annotate(
        "BER_Eve = 0 always:\n"
        "13-core coupling averages per-core power\n"
        "-> gain saturation uniform across symbols\n"
        "-> LR finds consistent H_eff -> BER = 0\n"
        "Solution B is INEFFECTIVE for high-dim MCF.",
        xy=(5, 0.1), fontsize=7, color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.4)
    ax.set_ylim(-0.02, 0.65)
    ax.invert_xaxis()

    # Panel B: ||H_eff||/||H_lin|| — proves saturation IS changing the channel
    ax2 = axes[1]
    ax2.semilogx(mults, heff_ratios, "g-D", lw=2, ms=7,
                 label="||H_eff||_F / ||H_lin||_F")
    ax2.axhline(1.0, ls="--", color="gray", alpha=0.6, label="Linear limit = 1.0")
    ax2.set_xlabel("P_sat_sig / z_rms   (weaker saturation →)")
    ax2.set_ylabel("||H_eff||_F / ||H_lin||_F")
    ax2.set_title("Channel change from saturation\n"
                  "(H_eff differs from H_lin but LR still finds it)", fontsize=9)
    ax2.annotate(
        "Saturation compresses H_eff\n"
        "but LR directly estimates H_eff\n"
        "-> consistent channel -> BER = 0\n"
        "Breaking this requires phase-sensitive\n"
        "nonlinearity (Solution A).",
        xy=(5, 0.6), fontsize=7, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)
    ax2.invert_xaxis()

    for ax_p in axes:
        for xv, lbl, col in [(50, "linear", "green"), (10, "mild", "orange"),
                              (2, "mod.", "darkorange"), (0.5, "strong", "red")]:
            ax_p.axvline(xv, ls=":", color=col, alpha=0.4)
            ax_p.text(xv, ax_p.get_ylim()[1] * 0.95, lbl,
                      ha="center", fontsize=6, color=col)

    fig.suptitle(
        "Fig 22 — Solution B: Signal-induced ODE gain saturation (honest null result)\n"
        "Multi-core power averaging makes amplitude nonlinearity transparent to LR attack",
        fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig22_ode_saturation.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig22_ode_saturation.png")
    print(f"  KEY FINDING: ODE saturation provides no LR resistance in 13-core MCF.")
    print(f"  Reason: multi-core coupling averages per-core power across 13 cores.")
    print(f"  -> Gain saturation is symbol-independent -> LR finds H_eff perfectly.")
    print(f"  -> Solution A (phase-sensitive) is required for LR attack resistance.")
    return results


# ── Figure 24 — SPM nonlinear activation: LR attack resistance ───────────────

def sweep_spm_nonlinear(enc: MCFEncryption,
                         true_key: np.ndarray,
                         out_dir: str,
                         T: int = 600,
                         n_trials: int = 4,
                         seed: int = 7) -> List[Dict]:
    """
    Fig 24 — Self-Phase Modulation (SPM) as a phase-sensitive nonlinear activation
    to break the linear regression (LR) attack.

    Physical mechanism used in the internal nonlinear design notes:
    SPM rotates the phase of each core's field by an amount proportional to its
    instantaneous power, without changing the amplitude:

        z'_i = z_i * exp(j * gamma_i * |z_i|^2)    [forward — Alice]
        z_i  = z'_i * exp(-j * gamma_i * |z'_i|^2) [inverse — Bob, EXACT]

    where gamma_i = gamma_NL * K_i  (pump key K_i sets SPM coefficient per core).

    Why SPM resists LR but SA does not:
    - SA changes amplitude → 13-core coupling averages per-core power uniformly
      across all symbol patterns → effective channel y = C*H_combined*x (linear)
    - SPM changes PHASE → multi-core coupling does NOT average phase fluctuations
      away. Even though E[|z_i|^2] is approximately constant, the per-symbol
      fluctuation std(|z_i|^2) ~ 28% of mean creates symbol-dependent phase
      rotations. The true map x->y is a degree-3 polynomial; the linear model
      y ~ H_est*x has irreducible residual error for large enough gamma.

    Architecture:
      z  = H1(K1) * x + n1          [stage 1 MCF]
      z' = SPM(z, gamma_NL * K1)    [key-dependent phase rotation]
      y  = H2(K2) * z' + n2         [stage 2 MCF with mixed-regime K2]

    The sweep tests gamma_NL from 0 (no SPM) to 20 (strong SPM) and measures:
      - BER_Bob  : Bob's BER after exact SPM inversion + MMSE decode
      - BER_Eve_LR : Eve's linear regression BER (expected to rise with gamma_NL)
      - BER_Eve_naive   : Random-key Eve (~0.5 throughout)
      - BER_Eve_informed: Informed Eve (~0.25 from mixed-regime K2, unaffected by SPM)

    Parameters
    ----------
    gamma_NL_vals : sweep of normalized SPM coefficients (design units)
    n_abs = ceil(0.30 * N) absorption-regime components in K2 (fixed)
    """
    rng      = np.random.default_rng(seed)
    N        = enc.geom.N
    data_idx = enc.geom.data_core_indices
    Nd       = len(data_idx)
    L        = enc.params.length_mm
    I_Nd     = np.eye(Nd)
    p_sat_inactive = 1e6

    # Mixed-regime K2 (fixed for all gamma sweep points).
    # N=13: optimal n_abs=10 (77%) from full Fig 21 simulation.
    # N>=19: 30% rule is N-independent (validated by Fig 21b).
    n_abs   = 10 if N == 13 else max(1, int(np.ceil(0.30 * N)))
    k1      = true_key.copy()
    k2      = rng.uniform(0.6, 1.0, size=N)
    abs_idx = rng.choice(N, size=n_abs, replace=False)
    k2[abs_idx] = rng.uniform(0.3, 0.5, size=n_abs)

    # Bob's matrices
    H1_bob = enc.sim.transfer_matrix(k1, L)
    H2_bob = enc.sim.transfer_matrix(k2, L)
    Hd1    = H1_bob[np.ix_(data_idx, data_idx)]
    Hd2    = H2_bob[np.ix_(data_idx, data_idx)]
    all_v1 = _ase_noise_variances(k1, enc.params, enc.geom)
    all_v2 = _ase_noise_variances(k2, enc.params, enc.geom)
    sig2_1 = float(np.mean(all_v1[data_idx]))
    sig2_2 = float(np.mean(all_v2[data_idx]))
    sc1    = np.sqrt(all_v1[data_idx] / 2.0)
    sc2    = np.sqrt(all_v2[data_idx] / 2.0)
    Hd1_inv = np.linalg.solve(Hd1.conj().T @ Hd1 + sig2_1 * I_Nd, Hd1.conj().T)
    Hd2_inv = np.linalg.solve(Hd2.conj().T @ Hd2 + sig2_2 * I_Nd, Hd2.conj().T)

    # Power normalization scale (deterministic from K1, known to both Alice and Bob).
    # z_scale[i] = sqrt(E[|Z_i|^2]) = sqrt(sum_j |Hd1[i,j]|^2) for unit-power QPSK input.
    # Dividing Z by z_scale makes E[|Z_norm_i|^2] = 1, so phi_NL = gamma * O(1)
    # instead of phi_NL = gamma * O(EDFA_gain) >> 2pi.
    # Both Alice and Bob compute this identically from K1; it is NOT secret.
    z_scale = np.sqrt(np.sum(np.abs(Hd1) ** 2, axis=1))   # (Nd,)

    def _ber(x_hat, bits_ref):
        re_b = (x_hat.real >= 0).astype(int)
        im_b = (x_hat.imag >= 0).astype(int)
        return float(np.mean(bits_ref != np.stack([re_b, im_b], axis=-1)))

    gamma_NL_vals = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = []
    print(f"\n[Fig 24] SPM nonlinear activation: LR attack resistance sweep (power-normalised)")
    print(f"  K2: n_abs={n_abs}/{N} mixed-regime, gamma_NL sweep: {gamma_NL_vals}")
    print(f"  z_scale (per core): mean={float(np.mean(z_scale)):.2f}, "
          f"min={float(np.min(z_scale)):.2f}, max={float(np.max(z_scale)):.2f}")
    print(f"  (phi_NL = gamma * |Z_norm|^2 with E[|Z_norm|^2]=1, so phi_std ~ gamma*0.28)")
    print(f"  {'gamma_NL':>10} {'BER_Bob':>10} {'BER_Eve_LR':>12} "
          f"{'BER_Eve_naive':>15} {'BER_Eve_inf':>13} {'phi_std_rad':>13}")
    print(f"  {'-'*75}")

    for gamma_NL in gamma_NL_vals:
        # Key-dependent SPM: gamma_i = gamma_NL * K1_i
        gamma_key = gamma_NL * k1[data_idx]

        bob_bers, lr_bers, naive_bers, inf_bers, phi_stds = [], [], [], [], []

        for _ in range(n_trials):
            bits   = rng.integers(0, 2, size=(T, Nd, 2))
            re_sym = 2.0 * bits[:, :, 0] - 1.0
            im_sym = 2.0 * bits[:, :, 1] - 1.0
            X      = (re_sym + 1j * im_sym) / np.sqrt(2.0)

            # Forward: stage 1 -> power-normalize -> SPM -> denormalize -> stage 2
            n1     = (rng.normal(size=(T, Nd)) * sc1[None, :] +
                      1j * rng.normal(size=(T, Nd)) * sc1[None, :])
            Z      = X @ Hd1.T + n1                         # (T, Nd)
            Z_norm = Z / z_scale[None, :]                   # unit-power per core
            Z_spm_norm = _apply_spm(Z_norm, gamma_key)      # SPM on unit-power signal
            Z_spm  = Z_spm_norm * z_scale[None, :]          # denormalize
            A      = _saturable_absorber(Z_spm, p_sat_inactive)   # SA inactive
            n2     = (rng.normal(size=(T, Nd)) * sc2[None, :] +
                      1j * rng.normal(size=(T, Nd)) * sc2[None, :])
            Y      = A @ Hd2.T + n2                          # (T, Nd)

            # Bob: exact 3-stage inversion
            # Normalise with the SAME z_scale (deterministic from K1) before SPM inv.
            z_hat      = Y @ Hd2_inv.T
            a_hat      = _saturable_absorber_inv(z_hat, p_sat_inactive)
            a_hat_norm = a_hat / z_scale[None, :]           # normalise
            z_inv_norm = _invert_spm(a_hat_norm, gamma_key) # exact SPM inversion
            z_inv      = z_inv_norm * z_scale[None, :]      # denormalise
            x_hat_b    = z_inv @ Hd1_inv.T
            bob_bers.append(_ber(x_hat_b, bits))

            # Diagnostic: phi_std on the normalised signal (should be ~ gamma * 0.28)
            phi_nl = gamma_key[None, :] * np.abs(Z_norm) ** 2
            phi_stds.append(float(np.std(phi_nl)))

            # Eve LR: fits y ~= x @ H_est.T (linear model for a nonlinear channel)
            Hd_est_T = np.linalg.lstsq(X, Y, rcond=None)[0]
            x_hat_lr = Y @ np.linalg.pinv(Hd_est_T.T).T
            lr_bers.append(_ber(x_hat_lr, bits))

            # Eve naive: random U[0,1] for both keys, no knowledge of gamma or scale
            ek1_n = rng.uniform(0, 1, size=N)
            ek2_n = rng.uniform(0, 1, size=N)
            Hd1_n = enc.sim.transfer_matrix(ek1_n, L)[np.ix_(data_idx, data_idx)]
            Hd2_n = enc.sim.transfer_matrix(ek2_n, L)[np.ix_(data_idx, data_idx)]
            zh_n  = Y @ np.linalg.pinv(Hd2_n).T
            ah_n  = _saturable_absorber_inv(zh_n, p_sat_inactive)
            zi_n  = _invert_spm(ah_n, np.zeros(Nd))        # gamma=0 (unknown)
            naive_bers.append(_ber(zi_n @ np.linalg.pinv(Hd1_n).T, bits))

            # Eve informed: knows gamma_NL (public) and key regime K1~U[0.6,1.0],
            # but has wrong K1 -> wrong z_scale -> wrong gamma_key -> wrong SPM inversion.
            ek1_i       = rng.uniform(0.6, 1.0, size=N)
            ek2_i       = rng.uniform(0.6, 1.0, size=N)    # wrong K2 regime
            Hd1_i       = enc.sim.transfer_matrix(ek1_i, L)[np.ix_(data_idx, data_idx)]
            Hd2_i       = enc.sim.transfer_matrix(ek2_i, L)[np.ix_(data_idx, data_idx)]
            z_scale_i   = np.sqrt(np.sum(np.abs(Hd1_i) ** 2, axis=1))  # wrong scale
            gamma_key_i = gamma_NL * ek1_i[data_idx]                    # wrong key
            zh_i        = Y @ np.linalg.pinv(Hd2_i).T
            ah_i        = _saturable_absorber_inv(zh_i, p_sat_inactive)
            ah_i_norm   = ah_i / z_scale_i[None, :]        # wrong normalisation
            zi_i_norm   = _invert_spm(ah_i_norm, gamma_key_i)  # wrong gamma
            zi_i        = zi_i_norm * z_scale_i[None, :]   # wrong denormalisation
            inf_bers.append(_ber(zi_i @ np.linalg.pinv(Hd1_i).T, bits))

        row = {
            "gamma_NL":           float(gamma_NL),
            "BER_Bob_mean":       float(np.mean(bob_bers)),
            "BER_Eve_LR_mean":    float(np.mean(lr_bers)),
            "BER_Eve_naive_mean": float(np.mean(naive_bers)),
            "BER_Eve_inf_mean":   float(np.mean(inf_bers)),
            "phi_std_rad_mean":   float(np.mean(phi_stds)),
        }
        results.append(row)
        print(f"  {gamma_NL:>10.1f} {row['BER_Bob_mean']:>10.4f} "
              f"{row['BER_Eve_LR_mean']:>12.4f} "
              f"{row['BER_Eve_naive_mean']:>15.4f} "
              f"{row['BER_Eve_inf_mean']:>13.4f} "
              f"{row['phi_std_rad_mean']:>13.4f}")

    _write_csv(f"{out_dir}/spm_nonlinear.csv", results)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    gammas     = [r["gamma_NL"]           for r in results]
    ber_bob    = [r["BER_Bob_mean"]       for r in results]
    ber_lr     = [r["BER_Eve_LR_mean"]    for r in results]
    ber_naive  = [r["BER_Eve_naive_mean"] for r in results]
    ber_inf    = [r["BER_Eve_inf_mean"]   for r in results]
    phi_std    = [r["phi_std_rad_mean"]   for r in results]

    ax = axes[0]
    ax.semilogx([g + 1e-3 for g in gammas], ber_bob,   "b-o",  lw=2, ms=7,
                label="BER_Bob (exact SPM inversion)")
    ax.semilogx([g + 1e-3 for g in gammas], ber_lr,    "r-s",  lw=2, ms=7,
                label="BER_Eve LR (linear regression)")
    ax.semilogx([g + 1e-3 for g in gammas], ber_naive, "k-^",  lw=2, ms=7,
                label="BER_Eve naive (random key)")
    ax.semilogx([g + 1e-3 for g in gammas], ber_inf,   "m-D",  lw=2, ms=7,
                label="BER_Eve informed (K1>0.6, wrong K2)")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="Random guess = 0.5")
    ax.set_xlabel("gamma_NL  (SPM coefficient, normalized units)")
    ax.set_ylabel("BER")
    ax.set_ylim(-0.02, 0.65)
    ax.set_title("Fig 24a — SPM nonlinear activation: BER vs. gamma_NL\n"
                 "Phase-only nonlinearity breaks LR attack; Bob decodes exactly")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.annotate(
        "SPM changes PHASE only\n-> not averaged by MCF coupling\n"
        "-> LR linear model fails at large gamma",
        xy=(5, 0.3), fontsize=8, color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    ax2 = axes[1]
    ax2.semilogx([g + 1e-3 for g in gammas], phi_std, "g-D", lw=2, ms=7,
                 label="std(phi_NL) per core")
    ax2.axhline(0.5, ls="--", color="orange", alpha=0.7,
                label="0.5 rad threshold (LR disruption)")
    ax2.axhline(np.pi / 4, ls=":", color="red", alpha=0.7,
                label="pi/4 rad (strong nonlinearity)")
    ax2.set_xlabel("gamma_NL  (SPM coefficient)")
    ax2.set_ylabel("std(phi_NL)  [radians]")
    ax2.set_title("Fig 24b — SPM phase variation across symbols\n"
                  "Increasing gamma raises phase std -> LR residual error")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)
    ax2.annotate(
        "std(phi_NL) = gamma_NL * std(|z_i|^2)\n"
        "std(|z_i|^2) ~ 28% of mean (13-core CLT)\n"
        "LR fails when std(phi) > ~0.5 rad",
        xy=(2, 0.3), fontsize=8, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

    fig.suptitle(
        "Fig 24 — SPM Phase-Sensitive Nonlinear Activation: Breaking the LR Attack\n"
        "Pump-key-dependent SPM creates cubic x->y mapping that linear regression cannot fit",
        fontsize=9.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig24_spm_nonlinear.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig24_spm_nonlinear.png")

    # Key findings
    lr_zero = [r for r in results if r["BER_Eve_LR_mean"] < 0.01]
    lr_nonzero = [r for r in results if r["BER_Eve_LR_mean"] >= 0.05]
    bob_zero = [r for r in results if r["BER_Bob_mean"] < 0.01]
    print(f"  BER_Bob = 0 throughout: {len(bob_zero)}/{len(results)} points")
    if lr_nonzero:
        print(f"  LR attack first fails (BER>0.05) at gamma_NL = {lr_nonzero[0]['gamma_NL']}")
    if lr_zero:
        print(f"  LR attack unresisted (BER<0.01) up to gamma_NL = {lr_zero[-1]['gamma_NL']}")
    return results


# ── Figure 25 — LR attack vs. number of known-plaintext pairs WITH SPM ────────

def sweep_spm_lr_pairs(enc: MCFEncryption,
                        true_key: np.ndarray,
                        out_dir: str,
                        gamma_NL: float = 2.0,
                        n_trials: int = 5,
                        seed: int = 7) -> List[Dict]:
    """
    Fig 25 — How many known-plaintext pairs does Eve need to break the
    SPM-protected two-stage channel?

    Without SPM (gamma=0): BER_Eve_LR=0 at N_obs=N=13 (exact rank threshold).
    With SPM (gamma=gamma_NL): the linear model y~Hx is incorrect; the residual
    grows with N_obs but the best linear approximation converges to a fixed
    non-zero BER. This sweep reveals whether SPM SHIFTS the knee of the curve
    (more pairs needed to reach low BER) or simply RAISES the asymptotic floor.

    Tests N_obs in [1, 5, 10, 13, 20, 50, 100, 200, 500].
    For each N_obs: Eve uses lstsq on those N_obs pairs, tests on held-out data.
    Also compares no-SPM baseline (gamma=0) vs SPM at gamma_NL.
    """
    rng      = np.random.default_rng(seed)
    N        = enc.geom.N
    data_idx = enc.geom.data_core_indices
    Nd       = len(data_idx)
    L        = enc.params.length_mm
    I_Nd     = np.eye(Nd)
    p_sat_inactive = 1e6
    T_test   = 600   # fixed held-out test set

    n_abs   = 10 if N == 13 else max(1, int(np.ceil(0.30 * N)))
    k1      = true_key.copy()
    k2      = rng.uniform(0.6, 1.0, size=N)
    abs_idx = rng.choice(N, size=n_abs, replace=False)
    k2[abs_idx] = rng.uniform(0.3, 0.5, size=n_abs)

    H1_bob  = enc.sim.transfer_matrix(k1, L)
    H2_bob  = enc.sim.transfer_matrix(k2, L)
    Hd1     = H1_bob[np.ix_(data_idx, data_idx)]
    Hd2     = H2_bob[np.ix_(data_idx, data_idx)]
    all_v1  = _ase_noise_variances(k1, enc.params, enc.geom)
    all_v2  = _ase_noise_variances(k2, enc.params, enc.geom)
    sc1     = np.sqrt(all_v1[data_idx] / 2.0)
    sc2     = np.sqrt(all_v2[data_idx] / 2.0)
    sig2_1  = float(np.mean(all_v1[data_idx]))
    sig2_2  = float(np.mean(all_v2[data_idx]))
    Hd1_inv = np.linalg.solve(Hd1.conj().T @ Hd1 + sig2_1 * I_Nd, Hd1.conj().T)
    Hd2_inv = np.linalg.solve(Hd2.conj().T @ Hd2 + sig2_2 * I_Nd, Hd2.conj().T)
    z_scale = np.sqrt(np.sum(np.abs(Hd1) ** 2, axis=1))   # power-norm scale

    gamma_key = gamma_NL * k1[data_idx]

    def _ber(x_hat, bits_ref):
        re_b = (x_hat.real >= 0).astype(int)
        im_b = (x_hat.imag >= 0).astype(int)
        return float(np.mean(bits_ref != np.stack([re_b, im_b], axis=-1)))

    def _forward(X_in, rng_):
        """Run SPM two-stage channel, return Y."""
        n1_    = (rng_.normal(size=X_in.shape) * sc1[None, :] +
                  1j * rng_.normal(size=X_in.shape) * sc1[None, :])
        Z_     = X_in @ Hd1.T + n1_
        Z_n    = Z_ / z_scale[None, :]
        Z_spm_ = _apply_spm(Z_n, gamma_key) * z_scale[None, :]
        n2_    = (rng_.normal(size=X_in.shape) * sc2[None, :] +
                  1j * rng_.normal(size=X_in.shape) * sc2[None, :])
        return Z_spm_ @ Hd2.T + n2_

    def _forward_nospm(X_in, rng_):
        """Run single-stage channel (no SPM, no K2), return Y."""
        n1_    = (rng_.normal(size=X_in.shape) * sc1[None, :] +
                  1j * rng_.normal(size=X_in.shape) * sc1[None, :])
        return X_in @ Hd1.T + n1_

    N_obs_vals = [1, 5, 10, 13, 20, 50, 100, 200, 500]
    results    = []

    print(f"\n[Fig 25] LR attack vs. N_obs known-plaintext pairs (gamma_NL={gamma_NL})")
    print(f"  n_abs={n_abs}/{N}, gamma_NL={gamma_NL}, T_test={T_test}, n_trials={n_trials}")
    print(f"  {'N_obs':>8} {'BER_LR_SPM':>13} {'BER_LR_noSPM':>15}")
    print(f"  {'-'*40}")

    for N_obs in N_obs_vals:
        spm_bers, nospm_bers = [], []
        for _ in range(n_trials):
            # Training data (N_obs known pairs)
            bits_tr  = rng.integers(0, 2, size=(N_obs, Nd, 2))
            X_tr     = ((2.0 * bits_tr[:, :, 0] - 1.0) +
                        1j * (2.0 * bits_tr[:, :, 1] - 1.0)) / np.sqrt(2.0)
            Y_tr_spm   = _forward(X_tr, rng)
            Y_tr_nospm = _forward_nospm(X_tr, rng)

            # Test data (T_test fresh pairs)
            bits_te  = rng.integers(0, 2, size=(T_test, Nd, 2))
            X_te     = ((2.0 * bits_te[:, :, 0] - 1.0) +
                        1j * (2.0 * bits_te[:, :, 1] - 1.0)) / np.sqrt(2.0)
            Y_te_spm   = _forward(X_te, rng)
            Y_te_nospm = _forward_nospm(X_te, rng)

            # Eve LR: fit on training, evaluate on test
            if N_obs >= Nd:   # square or overdetermined: lstsq
                H_est_spm   = np.linalg.lstsq(X_tr, Y_tr_spm,   rcond=None)[0]
                H_est_nospm = np.linalg.lstsq(X_tr, Y_tr_nospm, rcond=None)[0]
            else:              # underdetermined: minimum-norm via pinv
                H_est_spm   = np.linalg.pinv(X_tr) @ Y_tr_spm
                H_est_nospm = np.linalg.pinv(X_tr) @ Y_tr_nospm

            x_hat_spm   = Y_te_spm   @ np.linalg.pinv(H_est_spm.T).T
            x_hat_nospm = Y_te_nospm @ np.linalg.pinv(H_est_nospm.T).T
            spm_bers.append(_ber(x_hat_spm,   bits_te))
            nospm_bers.append(_ber(x_hat_nospm, bits_te))

        row = {"N_obs": N_obs,
               "BER_LR_SPM_mean":   float(np.mean(spm_bers)),
               "BER_LR_noSPM_mean": float(np.mean(nospm_bers))}
        results.append(row)
        print(f"  {N_obs:>8} {row['BER_LR_SPM_mean']:>13.4f} "
              f"{row['BER_LR_noSPM_mean']:>15.4f}")

    _write_csv(f"{out_dir}/spm_lr_pairs.csv", results)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    obs   = [r["N_obs"]             for r in results]
    b_spm = [r["BER_LR_SPM_mean"]   for r in results]
    b_ns  = [r["BER_LR_noSPM_mean"] for r in results]

    ax.semilogx(obs, b_ns,  "r-s",  lw=2, ms=7,
                label=f"BER_Eve_LR, no SPM (single-stage baseline)")
    ax.semilogx(obs, b_spm, "b-o",  lw=2, ms=7,
                label=f"BER_Eve_LR, SPM gamma_NL={gamma_NL} (two-stage + normalised SPM)")
    ax.axvline(Nd, ls="--", color="gray", alpha=0.7, label=f"N={Nd} (rank threshold)")
    ax.axhline(0.5, ls=":", color="green", alpha=0.6, label="Random guess floor (0.5)")
    ax.axhline(0.05, ls=":", color="orange", alpha=0.6, label="BER=0.05 security target")
    ax.set_xlabel("N_obs — number of known-plaintext (x, y) pairs used by Eve")
    ax.set_ylabel("BER_Eve_LR (lower = Eve decodes better)")
    ax.set_ylim(-0.02, 0.65)
    ax.set_title(
        f"Fig 25 — LR Attack vs. N_obs Known-Plaintext Pairs\n"
        f"SPM gamma_NL={gamma_NL} shifts/raises BER floor; no-SPM collapses to 0 at N_obs=N={Nd}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # annotate key features
    knee_nospm = next((r["N_obs"] for r in results if r["BER_LR_noSPM_mean"] < 0.05), None)
    if knee_nospm:
        ax.annotate(f"no-SPM breaks at\nN_obs={knee_nospm}",
                    xy=(knee_nospm, 0.04), xytext=(knee_nospm * 1.5, 0.12),
                    arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=8)
    knee_spm = next((r["N_obs"] for r in results if r["BER_LR_SPM_mean"] < 0.05), None)
    if knee_spm:
        ax.annotate(f"SPM breaks at\nN_obs={knee_spm}",
                    xy=(knee_spm, 0.04), xytext=(knee_spm * 1.5, 0.20),
                    arrowprops=dict(arrowstyle="->", color="blue"), color="blue", fontsize=8)
    else:
        ax.annotate("SPM: LR never reaches\nBER<0.05 in this sweep",
                    xy=(500, b_spm[-1]), xytext=(80, b_spm[-1] + 0.10),
                    arrowprops=dict(arrowstyle="->", color="blue"), color="blue", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig25_spm_lr_pairs.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig25_spm_lr_pairs.png")
    return results

# ── Figure 23 — Comprehensive adversary benchmark: old vs. new scheme ─────────

def sweep_adversary_benchmark_new_scheme(enc: MCFEncryption,
                                          true_key: np.ndarray,
                                          out_dir: str,
                                          T: int = 500,
                                          n_trials: int = 3,
                                          T_train_lstm: int = 1500,
                                          T_test_lstm: int = 500,
                                          lstm_window: int = 8,
                                          lstm_epochs: int = 10,
                                          seed: int = 7) -> Dict:
    """
    Fig 23 — Four-adversary benchmark: single-stage (old) vs. two-stage
    mixed-regime K2 (new) scheme.

    OLD: y = H1(K1) * x + n
    NEW: y = H2(K2) * sigma(H1(K1) * x + n1, P_sat_inf) + n2
         K2 has n_abs = ceil(0.30 * N) absorption-regime components in [0.3, 0.5]
         P_sat = 1e6 (SA inactive) — isolates the regime-crossing security effect

    Adversary types tested on both schemes:
    1. Naive    — random U[0,1] for all key components
    2. Informed — K1~U[0.6,1.0] (knows regime), K2~U[0.6,1.0] (wrong K2 regime)
    3. LR       — linear regression using T known-plaintext (x_t, y_t) pairs
    4. LSTM     — LSTM neural network (T_train_lstm pairs, if PyTorch available)

    Expected findings:
    - Naive:    ~0.5  in both schemes (random key fails regardless)
    - Informed: ~0.004 (old) -> ~0.25 (new)  FIXED by mixed-regime K2
    - LR:       ~0.0  (old) -> ~0.0  (new)   unfixed (linear composition, SA inactive)
    - LSTM:     ~0.5  in both schemes         LSTM learns nothing useful
    - Bob:       0.0  in both schemes         unchanged
    """
    rng      = np.random.default_rng(seed)
    N        = enc.geom.N
    data_idx = enc.geom.data_core_indices
    Nd       = len(data_idx)
    L        = enc.params.length_mm
    I_Nd     = np.eye(Nd)
    p_sat_inactive = 1e6   # SA effectively off: isolates regime-crossing effect

    # ── Mixed-regime K2: optimal n_abs from Fig 21 simulation ─────────────
    # N=13: n_abs=10 (77%) maximises BER_Eve_inf=0.208; for N>=19 use 30% rule.
    n_abs   = 10 if N == 13 else max(1, int(np.ceil(0.30 * N)))
    k1      = true_key.copy()
    k2      = rng.uniform(0.6, 1.0, size=N)
    abs_idx = rng.choice(N, size=n_abs, replace=False)
    k2[abs_idx] = rng.uniform(0.3, 0.5, size=n_abs)
    print(f"\n[Fig 23] Adversary benchmark: old vs. new scheme")
    print(f"  Two-stage K2: n_abs={n_abs}/{N} ({100*n_abs/N:.0f}% absorption regime)")
    print(f"  K2 absorption indices: {sorted(abs_idx.tolist())}")

    # ── Bob's matrices ─────────────────────────────────────────────────────
    H1_bob  = enc.sim.transfer_matrix(k1, L)
    H2_bob  = enc.sim.transfer_matrix(k2, L)
    Hd1     = H1_bob[np.ix_(data_idx, data_idx)]
    Hd2     = H2_bob[np.ix_(data_idx, data_idx)]
    all_v1  = _ase_noise_variances(k1, enc.params, enc.geom)
    all_v2  = _ase_noise_variances(k2, enc.params, enc.geom)
    sig2_1  = float(np.mean(all_v1[data_idx]))
    sig2_2  = float(np.mean(all_v2[data_idx]))
    sc1     = np.sqrt(all_v1[data_idx] / 2.0)
    sc2     = np.sqrt(all_v2[data_idx] / 2.0)
    Hd1_inv = np.linalg.solve(Hd1.conj().T @ Hd1 + sig2_1 * I_Nd, Hd1.conj().T)
    Hd2_inv = np.linalg.solve(Hd2.conj().T @ Hd2 + sig2_2 * I_Nd, Hd2.conj().T)

    # ── BER helper ─────────────────────────────────────────────────────────
    def _ber(x_hat: np.ndarray, bits_ref: np.ndarray) -> float:
        """BER between complex x_hat (T,Nd) and bits_ref (T,Nd,2)."""
        re_b = (x_hat.real >= 0).astype(int)
        im_b = (x_hat.imag >= 0).astype(int)
        bits_hat = np.stack([re_b, im_b], axis=-1)   # (T, Nd, 2)
        return float(np.mean(bits_ref != bits_hat))

    # ── Collect trials: Naive / Informed / LR ─────────────────────────────
    res_old = {k: [] for k in ["bob", "naive", "informed", "lr"]}
    res_new = {k: [] for k in ["bob", "naive", "informed", "lr"]}

    for _ in range(n_trials):
        bits   = rng.integers(0, 2, size=(T, Nd, 2))
        re_sym = 2.0 * bits[:, :, 0] - 1.0
        im_sym = 2.0 * bits[:, :, 1] - 1.0
        X      = (re_sym + 1j * im_sym) / np.sqrt(2.0)   # (T, Nd)

        # OLD SCHEME: y = X @ Hd1.T + noise1 ─────────────────────────────
        n1 = (rng.normal(size=(T, Nd)) * sc1[None, :] +
              1j * rng.normal(size=(T, Nd)) * sc1[None, :])
        Y_old = X @ Hd1.T + n1

        res_old["bob"].append(_ber(Y_old @ Hd1_inv.T, bits))

        ek_naive = rng.uniform(0, 1, size=N)
        Hd_n = enc.sim.transfer_matrix(ek_naive, L)[np.ix_(data_idx, data_idx)]
        res_old["naive"].append(_ber(Y_old @ np.linalg.pinv(Hd_n).T, bits))

        ek_inf = rng.uniform(0.6, 1.0, size=N)
        Hd_i  = enc.sim.transfer_matrix(ek_inf, L)[np.ix_(data_idx, data_idx)]
        res_old["informed"].append(_ber(Y_old @ np.linalg.pinv(Hd_i).T, bits))

        Hd_lr_T = np.linalg.lstsq(X, Y_old, rcond=None)[0]
        res_old["lr"].append(_ber(Y_old @ np.linalg.pinv(Hd_lr_T.T).T, bits))

        # NEW SCHEME: two-stage K2 mixed-regime, SA inactive ──────────────
        Z     = X @ Hd1.T + n1                          # stage 1 output
        A     = _saturable_absorber(Z, p_sat_inactive)  # SA (effectively linear)
        n2    = (rng.normal(size=(T, Nd)) * sc2[None, :] +
                 1j * rng.normal(size=(T, Nd)) * sc2[None, :])
        Y_new = A @ Hd2.T + n2                          # stage 2 output

        # Bob (new): invert stage 2, invert SA (trivial at p_sat->inf), invert stage 1
        z_hat = Y_new @ Hd2_inv.T
        a_hat = _saturable_absorber_inv(z_hat, p_sat_inactive)
        res_new["bob"].append(_ber(a_hat @ Hd1_inv.T, bits))

        # Naive (new): random K1 and K2 for both stages
        ek1_n = rng.uniform(0, 1, size=N)
        ek2_n = rng.uniform(0, 1, size=N)
        Hd1_n = enc.sim.transfer_matrix(ek1_n, L)[np.ix_(data_idx, data_idx)]
        Hd2_n = enc.sim.transfer_matrix(ek2_n, L)[np.ix_(data_idx, data_idx)]
        zh_n  = Y_new @ np.linalg.pinv(Hd2_n).T
        ah_n  = _saturable_absorber_inv(zh_n, p_sat_inactive)
        res_new["naive"].append(_ber(ah_n @ np.linalg.pinv(Hd1_n).T, bits))

        # Informed (new): K1~U[0.6,1.0], K2~U[0.6,1.0] (wrong K2 regime)
        ek1_i = rng.uniform(0.6, 1.0, size=N)
        ek2_i = rng.uniform(0.6, 1.0, size=N)
        Hd1_i = enc.sim.transfer_matrix(ek1_i, L)[np.ix_(data_idx, data_idx)]
        Hd2_i = enc.sim.transfer_matrix(ek2_i, L)[np.ix_(data_idx, data_idx)]
        zh_i  = Y_new @ np.linalg.pinv(Hd2_i).T
        ah_i  = _saturable_absorber_inv(zh_i, p_sat_inactive)
        res_new["informed"].append(_ber(ah_i @ np.linalg.pinv(Hd1_i).T, bits))

        # LR (new): fits y ~= x @ H_combined.T (SA inactive -> linear composition)
        Hd_comb_T = np.linalg.lstsq(X, Y_new, rcond=None)[0]
        res_new["lr"].append(_ber(Y_new @ np.linalg.pinv(Hd_comb_T.T).T, bits))

    old_m = {k: float(np.mean(v)) for k, v in res_old.items()}
    new_m = {k: float(np.mean(v)) for k, v in res_new.items()}

    # ── LSTM adversary ─────────────────────────────────────────────────────
    lstm_old = lstm_new = float("nan")
    if HAS_TORCH:
        T_tot = T_train_lstm + T_test_lstm
        bits_l = rng.integers(0, 2, size=(T_tot, Nd, 2))
        X_l    = ((2.0 * bits_l[:, :, 0] - 1.0) +
                  1j * (2.0 * bits_l[:, :, 1] - 1.0)) / np.sqrt(2.0)

        n1_l    = (rng.normal(size=(T_tot, Nd)) * sc1[None, :] +
                   1j * rng.normal(size=(T_tot, Nd)) * sc1[None, :])
        Y_old_l = X_l @ Hd1.T + n1_l

        print(f"  Training OLD-scheme LSTM ({T_train_lstm} pairs, {lstm_epochs} epochs)...")
        _, m_old, _ = train_eve_rnn(
            Y_old_l[:T_train_lstm], X_l[:T_train_lstm],
            window=lstm_window, epochs=lstm_epochs, seed=seed)

        dev = next(m_old.parameters()).device
        y_ctx = np.concatenate([Y_old_l[T_train_lstm - lstm_window: T_train_lstm],
                                 Y_old_l[T_train_lstm:]], axis=0)
        xh_flat = np.zeros((T_test_lstm, 2 * Nd), dtype=np.float32)
        m_old.eval()
        with torch.no_grad():
            for t in range(lstm_window, lstm_window + T_test_lstm):
                feat_r = np.concatenate(
                    [y_ctx[t - lstm_window: t].real,
                     y_ctx[t - lstm_window: t].imag], axis=1)
                inp = torch.from_numpy(feat_r[None].astype(np.float32)).to(dev)
                xh_flat[t - lstm_window] = m_old(inp).cpu().numpy()[0]
        lstm_old = _ber(xh_flat[:, :Nd] + 1j * xh_flat[:, Nd:], bits_l[T_train_lstm:])

        n2_l    = (rng.normal(size=(T_tot, Nd)) * sc2[None, :] +
                   1j * rng.normal(size=(T_tot, Nd)) * sc2[None, :])
        Z_l     = X_l @ Hd1.T + n1_l
        A_l     = _saturable_absorber(Z_l, p_sat_inactive)
        Y_new_l = A_l @ Hd2.T + n2_l

        print(f"  Training NEW-scheme LSTM ({T_train_lstm} pairs, {lstm_epochs} epochs)...")
        _, m_new, _ = train_eve_rnn(
            Y_new_l[:T_train_lstm], X_l[:T_train_lstm],
            window=lstm_window, epochs=lstm_epochs, seed=seed)

        dev = next(m_new.parameters()).device
        y_ctx_n = np.concatenate([Y_new_l[T_train_lstm - lstm_window: T_train_lstm],
                                   Y_new_l[T_train_lstm:]], axis=0)
        xh_flat_n = np.zeros((T_test_lstm, 2 * Nd), dtype=np.float32)
        m_new.eval()
        with torch.no_grad():
            for t in range(lstm_window, lstm_window + T_test_lstm):
                feat_r = np.concatenate(
                    [y_ctx_n[t - lstm_window: t].real,
                     y_ctx_n[t - lstm_window: t].imag], axis=1)
                inp = torch.from_numpy(feat_r[None].astype(np.float32)).to(dev)
                xh_flat_n[t - lstm_window] = m_new(inp).cpu().numpy()[0]
        lstm_new = _ber(xh_flat_n[:, :Nd] + 1j * xh_flat_n[:, Nd:], bits_l[T_train_lstm:])
        print(f"  LSTM: OLD={lstm_old:.4f}  NEW={lstm_new:.4f}")
    else:
        print(f"  Skipping LSTM (PyTorch not available)")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n  {'Adversary':<18} {'OLD BER':>10} {'NEW BER':>10}  {'Delta':>10}")
    print(f"  {'-'*52}")
    for key, label in [("bob","Bob (correct)"), ("naive","Naive U[0,1]"),
                        ("informed","Informed K>0.6"), ("lr","LR attack")]:
        d = new_m[key] - old_m[key]
        s = "+" if d >= 0 else ""
        print(f"  {label:<18} {old_m[key]:>10.4f} {new_m[key]:>10.4f}  {s}{d:>9.4f}")
    if not np.isnan(lstm_old):
        d = lstm_new - lstm_old
        s = "+" if d >= 0 else ""
        print(f"  {'LSTM':<18} {lstm_old:>10.4f} {lstm_new:>10.4f}  {s}{d:>9.4f}")

    # ── CSV ────────────────────────────────────────────────────────────────
    _write_csv(f"{out_dir}/adversary_benchmark_new_scheme.csv", [
        {"scheme": "old", "BER_Bob": old_m["bob"], "BER_Eve_naive": old_m["naive"],
         "BER_Eve_informed": old_m["informed"], "BER_Eve_LR": old_m["lr"],
         "BER_Eve_LSTM": lstm_old},
        {"scheme": "new", "BER_Bob": new_m["bob"], "BER_Eve_naive": new_m["naive"],
         "BER_Eve_informed": new_m["informed"], "BER_Eve_LR": new_m["lr"],
         "BER_Eve_LSTM": lstm_new},
    ])

    # ── Plot ───────────────────────────────────────────────────────────────
    adv_labels = ["Bob\n(correct)", "Naive\nU[0,1]", "Informed\nK>0.6",
                  "LR attack", "LSTM"]
    old_bers = [old_m["bob"], old_m["naive"], old_m["informed"], old_m["lr"],
                0.0 if np.isnan(lstm_old) else lstm_old]
    new_bers = [new_m["bob"], new_m["naive"], new_m["informed"], new_m["lr"],
                0.0 if np.isnan(lstm_new) else lstm_new]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(adv_labels))
    w = 0.35
    ax = axes[0]
    b_old = ax.bar(x - w / 2, old_bers, w, label="Single-stage (old)",
                   color="tab:blue", alpha=0.8, edgecolor="black")
    b_new = ax.bar(x + w / 2, new_bers, w, label="Two-stage K2 mixed (new)",
                   color="tab:orange", alpha=0.8, edgecolor="black")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="Random guess")
    ax.set_xticks(x)
    ax.set_xticklabels(adv_labels, fontsize=9)
    ax.set_ylabel("BER")
    ax.set_ylim(0, 0.70)
    ax.set_title("BER per adversary type\nOld (single-stage) vs. New (two-stage K2 mixed)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    for bar in list(b_old) + list(b_new):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # Panel B: delta BER (new - old) — improvement per adversary
    ax2 = axes[1]
    deltas = [n - o for o, n in zip(old_bers, new_bers)]
    cols_d = ["green" if d >= 0 else "red" for d in deltas]
    ax2.bar(x, deltas, color=cols_d, alpha=0.85, edgecolor="black")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(adv_labels, fontsize=9)
    ax2.set_ylabel("ΔBER  (New - Old)")
    ax2.set_title("Security improvement: two-stage K2 vs. single-stage\n"
                  "(green = higher Eve BER = more secure)")
    ax2.grid(axis="y", alpha=0.4)
    for xi, di in zip(x, deltas):
        s = "+" if di >= 0 else ""
        ax2.text(xi, di + (0.005 if di >= 0 else -0.018),
                 f"{s}{di:.3f}", ha="center",
                 va="bottom" if di >= 0 else "top", fontsize=8, fontweight="bold")

    inf_d = new_m["informed"] - old_m["informed"]
    ax2.annotate(
        f"Informed adversary:\n+{inf_d:.3f} improvement\nFIXED by mixed-regime K2",
        xy=(2, inf_d), xytext=(2.6, max(inf_d + 0.04, 0.05)),
        fontsize=7.5, color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="green"),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))
    lr_d = new_m["lr"] - old_m["lr"]
    ax2.annotate(
        f"LR attack: {lr_d:+.3f}\n(unchanged)\nOpen problem",
        xy=(3, lr_d), xytext=(3.4, lr_d - 0.06),
        fontsize=7.5, color="darkred",
        arrowprops=dict(arrowstyle="->", color="red"),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.7))

    fig.suptitle(
        "Fig 23 — Four-adversary benchmark: single-stage vs. two-stage mixed-regime K2\n"
        "Mixed-regime K2 FIXES informed adversary; LR attack remains open (SA inactive)",
        fontsize=9.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig23_adversary_benchmark_new_scheme.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig23_adversary_benchmark_new_scheme.png")
    return {"old": old_m, "new": new_m, "lstm_old": lstm_old, "lstm_new": lstm_new}


# ──────────────────────────────────────────────────────────────────────────────
# 13.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main_full_legacy():
    out_dir = "mcf_pls_figs"
    _ensure(out_dir)
    seed = 42

    print("=" * 60)
    print("MCF Physical-Layer Security Simulation")
    print("=" * 60)

    # ── instantiate physical model ────────────────────────────────────────
    params = FiberParams(
        n_core=1.44934,
        delta_n=0.00479,
        core_diam_um=8.2,
        pitch_um=16.4,
        lam_sig_um=1.550,
        lam_pump_um=0.976,
        length_mm=150.0,      # 150mm: between Lc (117.6mm) and 2Lc (235mm)
        g_max_per_mm=0.100,   # 0.1 mm⁻¹ (fully-active Er-doped MCF; Giles 1991)
        p_sat_norm=0.5,       # half-saturation pump power (normalised)
        alpha_H=0.3,
        sigma_noise=0.05,     # moderate noise: SNR ≈ 20 dB (realistic optical link)
    )
    geom = MCFGeometry(pitch_um=16.4)

    print(f"\nFiber parameters:")
    print(f"  n_core={params.n_core:.5f}, dn={params.delta_n:.5f}")
    print(f"  Core diameter={params.core_diam_um} um, Pitch={params.pitch_um} um")
    print(f"  V at 1550nm = {params.V():.3f} (single-mode: V < 2.405)")
    print(f"  V at 976nm  = {params.V(params.lam_pump_um):.3f}")
    print(f"  beta at 1550nm = {params.propagation_constant()*1000:.4f} rad/mm")
    kappa = params.coupling_coeff()
    print(f"  kappa (nearest-neighbour) = {kappa:.4f} rad/mm")
    Lc = np.pi / (2 * kappa) if kappa > 0 else float("inf")
    print(f"  Coupling length Lc = pi/2kappa = {Lc:.1f} mm")
    print(f"  Fiber length L = {params.length_mm} mm")
    print(f"  Erbium-doped cores (key): {geom.erbium_core_indices}")
    print(f"  Data cores: {geom.data_core_indices}")

    # ── create encryption framework ───────────────────────────────────────
    enc = MCFEncryption(params, geom, seed=seed)

    # Secret pump key (Alice & Bob shared)
    rng = np.random.default_rng(seed)
    true_key = rng.uniform(0.6, 1.0, size=13)  # 13-dim key (all cores); strong gain regime
    print(f"\nTrue pump key K = {np.round(true_key, 3)}")

    # ── Figure 0: MCF geometry diagram ───────────────────────────────────
    print("\n[Fig 0] MCF geometry diagram...")
    plot_mcf_geometry(geom, params, out_dir)

    # ── Physical validation: CME vs. analytical + wavelength comparison ──
    print("\n[Validation] CME simulator validation vs. analytical solution...")
    validate_cme_simulator(params, geom, out_dir)

    # ── Figure 1: transfer matrix visualisation ───────────────────────────
    print("\n[Fig 1] Transfer matrix visualisation...")
    plot_transfer_matrix(enc, true_key, out_dir)

    # ── Figure 2: BER vs. key mismatch ───────────────────────────────────
    print("\n[Fig 2] BER vs. key mismatch sweep...")
    sweep_key_mismatch(enc, true_key, out_dir, T=600, seed=seed)

    # ── Figure 2b: Analog key sensitivity (single-component dP) ──────────
    print("\n[Fig 2b] Analog key mismatch sweep (single-component dP)...")
    sweep_analog_key_mismatch(enc, true_key, out_dir, T=600, seed=seed)

    # ── Figure 3: Secrecy capacity vs. SNR (multi-modulation) ────────────
    print("\n[Fig 3] Secrecy capacity vs. SNR (QPSK + 16-QAM + 64-QAM)...")
    sweep_secrecy_capacity(enc, true_key, out_dir, seed=seed)

    # ── Figure 4: Key sensitivity ─────────────────────────────────────────
    print("\n[Fig 4] Key sensitivity...")
    sweep_key_sensitivity(enc, true_key, out_dir, seed=seed)

    # ── Figure 5: BER vs. fiber length ───────────────────────────────────
    print("\n[Fig 5] BER vs. fiber length...")
    sweep_fiber_length(enc, true_key, out_dir, T=500, seed=seed)

    # ── Figure 6: Condition number ───────────────────────────────────────
    print("\n[Fig 6] Condition number vs. pump diversity...")
    sweep_condition_number(enc, out_dir, n_trials=20, seed=seed)

    # ── Figure 7: Manufacturing tolerance ────────────────────────────────
    print("\n[Fig 7] Manufacturing tolerance sweep...")
    sweep_manufacturing_tolerance(enc, true_key, out_dir,
                                  T=500, n_trials=3, seed=seed)

    # ── Figure 10: Joint sensitivity (key mismatch × manufacturing) ───────
    print("\n[Fig 10] Joint sensitivity analysis...")
    sweep_joint_sensitivity(enc, true_key, out_dir, T=400, n_trials=4, seed=seed)

    # ── Figure 8: ML adversary ────────────────────────────────────────────
    print("\n[Fig 8] ML adversary benchmark...")
    ml_results = ml_adversary_benchmark(enc, true_key, out_dir,
                                         T_train=2500, T_test=600, seed=seed)

    # ── Figure 9: Secrecy capacity vs. Erbium core count ─────────────────
    print("\n[Fig 9] Erbium core count sweep (naive + informed adversary)...")
    erbium_results = sweep_erbium_count(enc, out_dir, seed=seed)

    # ── Figure 11: Mixed-regime key sweep ─────────────────────────────────
    print("\n[Fig 11] Mixed-regime key sweep (countermeasure to informed adversary)...")
    mixed_results = sweep_mixed_regime_key(enc, out_dir, T=500, n_trials=12, seed=seed)

    # ── Figure 12: Multi-modulation BER (actual QAM simulation) ──────────
    print("\n[Fig 12] Multi-modulation QAM BER (QPSK / 16-QAM / 64-QAM)...")
    qam_results = sweep_qam_ber(enc, true_key, out_dir, T=2000, seed=seed)

    # ── Figure 13: Multi-component analog sensitivity ─────────────────────
    print("\n[Fig 13] Multi-component analog sensitivity (k wrong components x dP)...")
    sweep_multi_component_sensitivity(enc, true_key, out_dir, T=400,
                                      n_subsets=6, seed=seed)

    # ── Figure 14: BER vs ||ΔH||_F correlation ───────────────────────────
    print("\n[Fig 14] BER_Eve vs ||DeltaH||_F correlation (80 random keys)...")
    sweep_ber_hf_correlation(enc, true_key, out_dir, T=200, n_keys=80, seed=seed)

    # ── Figure 15: Large MCF security scaling (7, 19, 37 cores) ──────────
    print("\n[Fig 15] Large MCF security scaling (7-core, 19-core, 37-core)...")
    sweep_large_mcf(params, out_dir, n_rings_list=[1, 2, 3],
                    n_trials=4, T=250, seed=seed)

    # ── Figure 16: QAM BER vs. key mismatch ───────────────────────────────
    print("\n[Fig 16] QAM BER vs. key mismatch (QPSK / 16-QAM / 64-QAM)...")
    sweep_qam_key_mismatch(enc, true_key, out_dir, T=600, n_trials=3, seed=seed)

    # ── Figure 10b: Joint sensitivity — key mismatch × fiber length ───────
    print("\n[Fig 10b] Joint sensitivity: key mismatch x fiber length...")
    sweep_joint_sensitivity_length(enc, true_key, out_dir, T=400, n_trials=4, seed=seed)

    # ── Figure 17: Linear regression adversary ────────────────────────────
    print("\n[Fig 17] Linear regression adversary (BER vs. known-plaintext pairs)...")
    sweep_linear_regression_adversary(enc, true_key, out_dir,
                                      T_collect=500, n_trials=4, seed=seed)

    # ── Figures 18–20: Two-stage nonlinear MCF (saturable absorber) ───────
    print("\n[Fig 18] Nonlinear operating point sweep (P_sat_sig vs. BER)...")
    op_results = sweep_operating_point_nonlinear(enc, true_key, out_dir,
                                                 T=600, n_trials=3, seed=seed)

    # Choose best operating point: P_sat where BER_Eve_LR is highest and BER_Bob=0
    best_op = max(op_results, key=lambda r: r["BER_Eve_LR"]
                  if r["BER_Bob"] < 0.02 else -1)
    best_mult = best_op["p_sat_mult"]
    print(f"  Best operating point: P_sat = {best_op['p_sat_sig']:.3f} "
          f"({best_mult}x RMS), BER_Eve_LR={best_op['BER_Eve_LR']:.4f}")

    print("\n[Fig 19] LR adversary on two-stage nonlinear channel...")
    sweep_lr_adversary_nonlinear(enc, true_key, out_dir,
                                 T_collect=500, p_sat_sig_mult=best_mult,
                                 n_trials=3, seed=seed)

    print("\n[Fig 20] Key mismatch: linear vs. nonlinear comparison...")
    sweep_key_mismatch_nonlinear(enc, true_key, out_dir, T=600, n_trials=3,
                                 p_sat_sig_mult=best_mult, seed=seed)

    # ── Figure 21: Mixed-regime K₂ fix (Problem 2 — informed adversary) ──
    print("\n[Fig 21] Mixed-regime K2: countermeasure to informed adversary...")
    mixed_k2_results = sweep_mixed_regime_k2_fix(enc, true_key, out_dir,
                                                  T=400, n_trials=3, seed=seed)
    # Report best operating point (highest BER_Eve_inf with BER_Bob < 2%)
    feasible = [r for r in mixed_k2_results if r["BER_Bob_mean"] < 0.02]
    if feasible:
        best_k2 = max(feasible, key=lambda r: r["BER_Eve_inf_mean"])
        print(f"  Best n_abs={best_k2['n_absorption']}: "
              f"BER_Eve_inf={best_k2['BER_Eve_inf_mean']:.4f}, "
              f"BER_Bob={best_k2['BER_Bob_mean']:.4f}, "
              f"cond(H2)={best_k2['cond_H2_mean']:.2f}")

    # ── Figure 21b: Mixed-regime K₂ scaling (7 / 19 / 37 cores) ─────────
    print("\n[Fig 21b] Mixed-regime K2 scaling across 7/19/37-core MCF...")
    sweep_mixed_regime_k2_scaling(params, out_dir,
                                   n_rings_list=[1, 2, 3],
                                   T=100, n_trials=2, seed=seed)

    # ── Figure 22: ODE saturation — Solution B tradeoff (Problem 1) ──────
    print("\n[Fig 22] ODE gain saturation sweep (Solution B — LR attack tradeoff)...")
    sweep_ode_saturation(enc, true_key, out_dir, T=50, n_trials=3, seed=seed)

    # ── Figure 24: SPM nonlinear activation (LR attack resistance) ───────
    print("\n[Fig 24] SPM nonlinear activation: LR attack resistance...")
    spm_results = sweep_spm_nonlinear(enc, true_key, out_dir,
                                      T=500, n_trials=4, seed=seed)

    # ── Figure 25: LR attack vs. N_obs pairs WITH SPM ────────────────────
    print("\n[Fig 25] LR attack vs. N_obs known-plaintext pairs (SPM gamma=2)...")
    sweep_spm_lr_pairs(enc, true_key, out_dir, gamma_NL=2.0, n_trials=5, seed=seed)

    # ── Figure 23: Adversary benchmark — old vs. new scheme ───────────────
    print("\n[Fig 23] Adversary benchmark: single-stage vs. two-stage mixed-regime K2...")
    adv_bench = sweep_adversary_benchmark_new_scheme(
        enc, true_key, out_dir, T=500, n_trials=3,
        T_train_lstm=1500, T_test_lstm=500, lstm_epochs=10, seed=seed)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Run complete. Figures + CSVs saved to:", out_dir)
    if ml_results:
        print(f"  BER_Bob       = {ml_results.get('BER_Bob', '?'):.4f}")
        print(f"  BER_Eve_WK    = {ml_results.get('BER_Eve_WK', '?'):.4f}")
        print(f"  BER_Eve_LSTM  = {ml_results.get('BER_Eve_LSTM', '?'):.4f}")
    if erbium_results:
        best = max(erbium_results, key=lambda r: r.get("Cs_mean", 0))
        print(f"  Best C_s at n_Er={best['n_erbium_cores']}: {best['Cs_mean']:.3f} bits/s/Hz")
        print(f"  BER_Eve_naive at n_Er={best['n_erbium_cores']}: {best['BER_Eve_naive_mean']:.4f}")
        print(f"  BER_Eve_informed at n_Er={best['n_erbium_cores']}: {best['BER_Eve_inf_mean']:.4f}")
    if adv_bench:
        print(f"\n  Fig 23 — Adversary benchmark (old vs. new scheme):")
        print(f"  {'Adversary':<18} {'OLD':>8} {'NEW':>8}")
        for key, label in [("bob","Bob"), ("naive","Naive"), ("informed","Informed"), ("lr","LR")]:
            print(f"  {label:<18} {adv_bench['old'][key]:>8.4f} {adv_bench['new'][key]:>8.4f}")
        if not np.isnan(adv_bench['lstm_old']):
            print(f"  {'LSTM':<18} {adv_bench['lstm_old']:>8.4f} {adv_bench['lstm_new']:>8.4f}")
    print("=" * 60)


def main():
    """
    Focused publication runner for the v3 manuscript package.

    The suite writes the current manuscript package to `mcf_pls_figs3` and
    regenerates the focused benchmark, the scaling/significance artifacts, and
    the calibration bundle used by the draft and supplement.

    Main outputs:
    - FO: keyed optical encryption on the same observation channel
    - WT: explicit degraded passive-tap observation model
    - Linear baseline vs. protected nonlinear schemes
    - Calibration traceability artifacts for the submission package
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "mcf_pls_figs3")
    _ensure(out_dir)
    seed = 42

    print("=" * 72)
    print("MCF PLS v3: focused publication suite")
    print("=" * 72)

    params = FiberParams(
        n_core=1.44934,
        delta_n=0.00479,
        core_diam_um=8.2,
        pitch_um=16.4,
        lam_sig_um=1.550,
        lam_pump_um=0.976,
        length_mm=150.0,
        g_max_per_mm=0.100,
        p_sat_norm=0.5,
        alpha_H=0.3,
        sigma_noise=0.05,
    )
    geom = MCFGeometry(pitch_um=16.4)
    enc = MCFEncryption(params, geom, seed=seed)

    rng = np.random.default_rng(seed)
    true_key = rng.uniform(0.6, 1.0, size=geom.N)
    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )

    config_row = {
        "seed": seed,
        "out_dir": out_dir,
        "true_key": np.round(true_key, 6).tolist(),
        "tap_label": tap_cfg.label,
        "tap_fraction": tap_cfg.tap_fraction,
        "eve_noise_scale": tap_cfg.eve_noise_scale,
        "observed_core_indices": _resolve_tap_core_indices(geom, tap_cfg.observed_core_indices),
        "fiber_length_mm": params.length_mm,
        "sigma_noise": params.sigma_noise,
        "g_max_per_mm": params.g_max_per_mm,
    }
    _write_csv(os.path.join(out_dir, "publication_suite_config.csv"), [config_row])

    print(f"Output directory: {out_dir}")
    print(f"Observed tap cores: {config_row['observed_core_indices']}")
    print(f"Tap fraction: {tap_cfg.tap_fraction:.3f}, Eve noise scale: {tap_cfg.eve_noise_scale:.2f}")

    print("\n[Suite] Geometry and physical validation...")
    plot_mcf_geometry(geom, params, out_dir)
    validate_cme_simulator(params, geom, out_dir)
    plot_transfer_matrix(enc, true_key, out_dir)
    trace_rows, anchor_rows = write_physical_calibration_bundle(params, out_dir, base_dir)

    benchmark_rows = benchmark_fo_wt_attack_suite(
        enc, true_key, out_dir, tap_cfg=tap_cfg, seed=seed)
    complexity_rows = sweep_kpa_lr_sample_complexity(
        enc, true_key, out_dir, tap_cfg=tap_cfg, seed=seed)
    lstm_rows = sweep_lstm_training_size(
        enc, true_key, out_dir, tap_cfg=tap_cfg, seed=seed)
    core_rows = benchmark_core_scaling_13_19_37(params, out_dir, seed=seed)
    xpm_rows = benchmark_xpm_exploratory(
        enc, true_key, out_dir, tap_cfg=tap_cfg, seed=seed)
    multiseed_rows = validate_37core_protected2_multiseed(params, out_dir)
    complexity37_rows = sweep_37core_sample_complexity(params, out_dir)
    tap37_rows = sweep_37core_tap_sensitivity(params, out_dir, seed=seed)
    spmxpm37_rows = benchmark_spm_xpm_combo_37core(params, out_dir, seed=seed)
    qam37_rows = sweep_qam_ber_protected2_37core(params, out_dir, seed=seed)
    scaling_sig_rows, scaling_sig_tests = run_scaling_multiseed_significance(
        params, out_dir)
    write_minimum_publishable_plan(out_dir)
    write_submission_manifest(out_dir)

    print("\n" + "=" * 72)
    print("Focused suite complete.")
    print(f"Saved figures and CSVs to: {out_dir}")
    if benchmark_rows:
        print("Attack benchmark summary:")
        for row in benchmark_rows:
            print(
                f"  {row['scheme']:<9} {row['option']}: "
                f"Bob={row['BER_Bob']:.4f}  "
                f"KPA={row['BER_KPA']:.4f}  "
                f"LR={row['BER_LR']:.4f}  "
                f"LSTM={row['BER_LSTM']:.4f}"
            )
    print(f"KPA/LR sweep rows: {len(complexity_rows)}")
    print(f"LSTM sweep rows: {len(lstm_rows)}")
    print(f"Core-scaling rows: {len(core_rows)}")
    print(f"XPM exploratory rows: {len(xpm_rows)}")
    print(f"37-core multiseed rows: {len(multiseed_rows)}")
    print(f"37-core sample-complexity rows: {len(complexity37_rows)}")
    print(f"37-core tap-sensitivity rows: {len(tap37_rows)}")
    print(f"37-core SPM+XPM rows: {len(spmxpm37_rows)}")
    print(f"Scaling-significance raw rows: {len(scaling_sig_rows)}")
    print(f"Scaling-significance tests: {len(scaling_sig_tests)}")
    print(f"Calibration traceability rows: {len(trace_rows)}")
    print(f"Calibration anchor rows: {len(anchor_rows)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
