"""Legacy exploratory figure functions from v1/v2 development.

These functions were used during exploratory analysis and are NOT called
by the publication suite main(). They are preserved for reference and
reproducibility of earlier results (mcf_pls_figs/ and mcf_pls_figs2/).
Entry point: main_full_legacy()
"""
from __future__ import annotations
import os, csv, json, warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from mcf_pls.physics import FiberParams, MCFGeometry, HexMCFGeometry, CMESimulator, MCFEncryption
from mcf_pls.nonlinear import (_saturable_absorber, _saturable_absorber_inv, _propagate_saturated,
    _apply_spm, _invert_spm, _apply_xpm, _invert_xpm, _secrecy_capacity,
    _mmse_sic_rate_qpsk, _ase_noise_variances, key_sensitivity,
    _int_to_bits, _qam_constellation_gray, _nearest_constellation_indices)
from mcf_pls.adversaries import train_eve_mlp, train_eve_rnn
from mcf_pls.evaluation import (TapObservationConfig, _qpsk_bits_to_syms, _qpsk_demod_array,
    _bit_error_rate, _resolve_tap_core_indices, _complex_awgn, _linear_mmse_inverse,
    _mixed_regime_key, _row_energy_scale, _mmse_inverse_from_key)
from mcf_pls.utils import _ensure, _write_csv, _read_csv_rows

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

# plot_transfer_matrix, plot_mcf_geometry, validate_cme_simulator, _draw_power_map
# live in mcf_pls/utils.py (also called by run.py / main())
from mcf_pls.utils import plot_transfer_matrix, plot_mcf_geometry, validate_cme_simulator, _draw_power_map

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


