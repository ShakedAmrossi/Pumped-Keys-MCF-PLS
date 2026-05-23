"""Publication benchmark suite: all functions called by main()."""
from __future__ import annotations
import os, csv, json, warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcf_pls.physics import FiberParams, MCFGeometry, HexMCFGeometry, MCFEncryption, CMESimulator
from mcf_pls.nonlinear import (_apply_spm, _invert_spm, _apply_xpm, _invert_xpm,
    _qam_constellation_gray, _nearest_constellation_indices, _ase_noise_variances)
from mcf_pls.evaluation import (TapObservationConfig, _qpsk_bits_to_syms, _qpsk_demod_array,
    _bit_error_rate, _resolve_tap_core_indices, _complex_awgn, _linear_mmse_inverse,
    _apply_wt_tap, _mixed_regime_key, _row_energy_scale,
    _evaluate_kpa_attack, _evaluate_lr_attack, _evaluate_mlp_attack,
    _evaluate_transformer_attack, _evaluate_hybrid_attack, _evaluate_lstm_attack,
    _build_dataset_for_scheme, _apply_laser_phase_noise)
from mcf_pls.utils import _ensure, _write_csv, _read_csv_rows


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


def sweep_fab_tolerance_37core(base_params: FiberParams,
                               out_dir: str,
                               fab_tol_fracs: Tuple[float, ...] = (0.0, 0.001, 0.003, 0.01, 0.03),
                               seeds: Optional[List[int]] = None,
                               T: int = 500) -> List[Dict]:
    """
    Supplementary S12: Manufacturing Δn tolerance sweep for 37-core P2.

    Models the realistic mismatch scenario: Bob calibrates his MMSE equalizer
    using the NOMINAL (design-spec) transfer matrix, but the actual fiber has
    per-core delta_n perturbations (manufacturing defects).  Bob decodes with
    the wrong matrix; both his BER and Eve's attack difficulty degrade as
    fab_tol_frac increases.

    Key finding: degradation begins by fab_tol_frac = 0.001 (Bob BER rises to
    ~0.22); Bob collapses to random-guess BER by 0.003.  Eve's best linear
    attack BER simultaneously drops from ~0.436 to ~0.147 at 0.030, because the
    perturbed H1 changes the SPM input distribution and the channel becomes more
    linearly learnable.  Per-fiber calibration is therefore required for BOTH
    reliable decoding and security.

    Scope: only per-core delta_n / beta_0 variation is swept; coupling
    perturbations (which depend on delta_n through V, W, and beta in
    coupling_coeff()) are not swept here.  Only the first P2 stage (H1) uses
    the perturbed matrix; H2 perturbations are left for separate analysis.

    fab_tol_frac = 0.003 (±0.3% Δn variation) is cited as a typical upper-bound
    for commercial MCF in Takenaga et al. (2012).
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    rows = []
    print("\n[Supp S12] Fabrication tolerance sweep (37-core P2, nominal-Bob / perturbed-H1)...")
    for seed in seeds:
        enc, true_key, geom = _make_37core_setup(base_params, seed)
        params = enc.params
        data_idx = geom.data_core_indices
        Nd = len(data_idx)
        I_Nd = np.eye(Nd)
        gamma_nl = 2.0

        # P2 channel uses two keys; k2 is drawn once and held fixed for this seed
        rng_setup = np.random.default_rng(seed + 5500)
        k2 = _mixed_regime_key(rng_setup, geom.N)

        # NOMINAL first-stage matrix — Bob always calibrates to this
        H1_nom = enc.sim.transfer_matrix(true_key, params.length_mm)
        H2_nom = enc.sim.transfer_matrix(k2, params.length_mm)
        Hd1_nom = H1_nom[np.ix_(data_idx, data_idx)]
        Hd2_nom = H2_nom[np.ix_(data_idx, data_idx)]
        all_v1 = _ase_noise_variances(true_key, params, geom)
        all_v2 = _ase_noise_variances(k2, params, geom)
        sigma1 = np.sqrt(all_v1[data_idx])
        sigma2 = np.sqrt(all_v2[data_idx])
        sig2_1 = float(np.mean(all_v1[data_idx]))
        sig2_2 = float(np.mean(all_v2[data_idx]))
        # Bob's MMSE inverses — built from nominal matrices, never updated
        Hd1_nom_inv = np.linalg.solve(
            Hd1_nom.conj().T @ Hd1_nom + sig2_1 * I_Nd, Hd1_nom.conj().T)
        Hd2_nom_inv = np.linalg.solve(
            Hd2_nom.conj().T @ Hd2_nom + sig2_2 * I_Nd, Hd2_nom.conj().T)
        z_scale = _row_energy_scale(Hd1_nom)  # scale factor from nominal H1
        gamma_key = gamma_nl * true_key[data_idx]

        for frac in fab_tol_fracs:
            # ACTUAL first-stage: perturbed by per-core Δn manufacturing defects.
            # beta_0 is updated per core; kappa is NOT re-evaluated here (it
            # depends on delta_n through V, W, and coupling_coeff(), but that
            # perturbation path is left for a separate coupling-deviation study).
            if frac > 0:
                perturbed_sim = CMESimulator(
                    params, geom,
                    rng=np.random.default_rng(seed + 5000 + int(frac * 10000)),
                    delta_n_noise_frac=frac)
                H1_actual = perturbed_sim.transfer_matrix(true_key, params.length_mm)
            else:
                H1_actual = H1_nom
            Hd1_actual = H1_actual[np.ix_(data_idx, data_idx)]

            rng = np.random.default_rng(seed + 6000 + int(frac * 10000))
            bits_train = rng.integers(0, 2, size=(T, Nd, 2))
            bits_test = rng.integers(0, 2, size=(T, Nd, 2))
            x_train = _qpsk_bits_to_syms(bits_train)
            x_test = _qpsk_bits_to_syms(bits_test)

            def _fwd_p2(x: np.ndarray, rng_l: np.random.Generator) -> np.ndarray:
                # Stage 1: ACTUAL (perturbed) first MCF transfer
                n1 = _complex_awgn(rng_l, x.shape, sigma1[None, :])
                z = x @ Hd1_actual.T + n1
                # SPM nonlinear block (key-controlled)
                z_n = z / z_scale[None, :]
                z_spm = _apply_spm(z_n, gamma_key) * z_scale[None, :]
                # Stage 2: nominal second MCF transfer
                n2 = _complex_awgn(rng_l, x.shape, sigma2[None, :])
                return z_spm @ Hd2_nom.T + n2

            y_train = _fwd_p2(x_train, rng)
            y_test = _fwd_p2(x_test, rng)

            # Bob decodes using NOMINAL inverses — mismatch in first stage
            z_hat = y_test @ Hd2_nom_inv.T
            z_inv = _invert_spm(z_hat / z_scale[None, :], gamma_key) * z_scale[None, :]
            x_hat_bob = z_inv @ Hd1_nom_inv.T
            ber_bob = _bit_error_rate(bits_test, _qpsk_demod_array(x_hat_bob))

            # Eve (FO): full observation, best of KPA-CE and KPA-LR linear attacks
            n_obs = min(200, T)
            ber_ce_fo = _evaluate_kpa_attack(
                x_train[:n_obs], y_train[:n_obs], y_test, bits_test, sigma2=sig2_2)
            ber_lr_fo = _evaluate_lr_attack(
                x_train[:n_obs], y_train[:n_obs], y_test, bits_test)
            ber_eve_fo = min(ber_ce_fo, ber_lr_fo)

            # Eve (WT): tapped outer-ring observation
            y_eve_tr_wt, _ = _apply_wt_tap(y_train, geom, params, tap_cfg, rng)
            y_eve_te_wt, _ = _apply_wt_tap(y_test, geom, params, tap_cfg, rng)
            sigma2_wt = (params.sigma_noise * tap_cfg.eve_noise_scale) ** 2
            ber_ce_wt = _evaluate_kpa_attack(
                x_train[:n_obs], y_eve_tr_wt[:n_obs], y_eve_te_wt, bits_test,
                sigma2=sigma2_wt)
            ber_lr_wt = _evaluate_lr_attack(
                x_train[:n_obs], y_eve_tr_wt[:n_obs], y_eve_te_wt, bits_test)
            ber_eve_wt = min(ber_ce_wt, ber_lr_wt)

            row = {
                "seed": seed,
                "fab_tol_frac": frac,
                "BER_Bob": ber_bob,
                "BER_Eve_FO": ber_eve_fo,
                "BER_Eve_WT": ber_eve_wt,
            }
            rows.append(row)
            print(f"  seed={seed} tol={frac:.3f}: Bob={ber_bob:.4f} "
                  f"Eve_FO={ber_eve_fo:.4f} Eve_WT={ber_eve_wt:.4f}")

    _write_csv(f"{out_dir}/fab_tolerance_37core.csv", rows)

    summary = []
    for frac in fab_tol_fracs:
        sub = [r for r in rows if r["fab_tol_frac"] == frac]
        bob_mean, bob_std, bob_ci = _mean_std_ci([r["BER_Bob"] for r in sub])
        efo_mean, efo_std, efo_ci = _mean_std_ci([r["BER_Eve_FO"] for r in sub])
        ewt_mean, ewt_std, ewt_ci = _mean_std_ci([r["BER_Eve_WT"] for r in sub])
        summary.append({
            "fab_tol_frac": frac,
            "BER_Bob_mean": bob_mean, "BER_Bob_std": bob_std, "BER_Bob_ci95": bob_ci,
            "BER_Eve_FO_mean": efo_mean, "BER_Eve_FO_std": efo_std, "BER_Eve_FO_ci95": efo_ci,
            "BER_Eve_WT_mean": ewt_mean, "BER_Eve_WT_std": ewt_std, "BER_Eve_WT_ci95": ewt_ci,
        })
    _write_csv(f"{out_dir}/fab_tolerance_37core_summary.csv", summary)

    # Categorical x-axis: fab_tol_frac=0 is a special "ideal" baseline that
    # cannot be represented on a log scale (log(0)=-inf). Use integer indices
    # with percentage tick labels instead.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frac_vals = [r["fab_tol_frac"] for r in summary]
    x_idx = list(range(len(frac_vals)))
    tick_labels = [f"{f*100:.1f}%" if f > 0 else "0\n(ideal)" for f in frac_vals]
    ax.plot(x_idx, [r["BER_Bob_mean"] for r in summary],
            "o-", color="tab:blue", lw=2, ms=6, label="Bob (nominal decoder)")
    ax.plot(x_idx, [r["BER_Eve_FO_mean"] for r in summary],
            "^-", color="tab:red", lw=2, ms=6, label="Eve best linear (FO)")
    ax.plot(x_idx, [r["BER_Eve_WT_mean"] for r in summary],
            "D--", color="darkred", lw=2, ms=6, label="Eve best linear (WT)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Δn variation (% of nominal)")
    ax.set_ylabel("BER")
    ax.set_title("37-core P2: BER vs. manufacturing Δn tolerance\n"
                 "(Bob uses nominal decoder; actual fiber has per-core Δn defects)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_fab_tolerance_37core.png", dpi=200)
    plt.close(fig)
    return rows


def sweep_phase_noise_37core(base_params: FiberParams,
                             out_dir: str,
                             sigma_pn_values: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.10),
                             seeds: Optional[List[int]] = None,
                             T_train: int = 1200,
                             T_test: int = 400) -> List[Dict]:
    """
    Supplementary / Section 6.8: Laser phase noise sensitivity for 37-core P2.

    Injects Wiener-process phase noise on the transmitted signal before the
    two-stage P2 channel.  Both Bob and Eve observe the phase-noisy output
    with no phase compensation applied to either party.  The sweep is an
    uncompensated stress test that identifies the operating regime constraint:
    the sigma_pn level at which Bob's BER begins to degrade sets the minimum
    coherence requirement for the transmitter laser.

    sigma_pn values and corresponding linewidths at 10 GBd (T_s = 0.1 ns):
      linewidth = sigma_pn^2 / (2*pi*T_s)
      0.00 rad  — ideal coherent (baseline)
      0.01 rad  — ~159 kHz  (narrow-linewidth DFB / ECL class)
      0.05 rad  — ~3.98 MHz (typical telecom DFB)
      0.10 rad  — ~15.9 MHz (wide-linewidth DFB)
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    rows = []
    print("\n[Section 6.8] Phase noise sensitivity sweep (37-core P2)...")
    for seed in seeds:
        enc, true_key, geom = _make_37core_setup(base_params, seed)
        params = enc.params
        data_idx = geom.data_core_indices
        Nd = len(data_idx)
        I_Nd = np.eye(Nd)

        k1 = true_key.copy()
        rng_setup = np.random.default_rng(seed + 8000)
        k2 = _mixed_regime_key(rng_setup, geom.N)

        H1 = enc.sim.transfer_matrix(k1, params.length_mm)
        H2 = enc.sim.transfer_matrix(k2, params.length_mm)
        Hd1 = H1[np.ix_(data_idx, data_idx)]
        Hd2 = H2[np.ix_(data_idx, data_idx)]
        from mcf_pls.nonlinear import _ase_noise_variances, _apply_spm, _invert_spm
        all_v1 = _ase_noise_variances(k1, params, geom)
        all_v2 = _ase_noise_variances(k2, params, geom)
        sigma1 = np.sqrt(all_v1[data_idx])
        sigma2 = np.sqrt(all_v2[data_idx])
        sig2_1 = float(np.mean(all_v1[data_idx]))
        sig2_2 = float(np.mean(all_v2[data_idx]))
        Hd1_inv = np.linalg.solve(Hd1.conj().T @ Hd1 + sig2_1 * I_Nd, Hd1.conj().T)
        Hd2_inv = np.linalg.solve(Hd2.conj().T @ Hd2 + sig2_2 * I_Nd, Hd2.conj().T)
        z_scale = _row_energy_scale(Hd1)
        gamma_key = 2.0 * k1[data_idx]

        for sigma_pn in sigma_pn_values:
            for option in ["FO", "WT"]:
                rng = np.random.default_rng(seed + 8100 + int(sigma_pn * 1000))
                bits_train = rng.integers(0, 2, size=(T_train, Nd, 2))
                bits_test = rng.integers(0, 2, size=(T_test, Nd, 2))
                x_train_clean = _qpsk_bits_to_syms(bits_train)
                x_test_clean = _qpsk_bits_to_syms(bits_test)

                # Apply phase noise to transmitted symbols
                x_train = _apply_laser_phase_noise(x_train_clean, sigma_pn, rng)
                x_test = _apply_laser_phase_noise(x_test_clean, sigma_pn, rng)

                def _fwd(x, rng_l):
                    n1 = _complex_awgn(rng_l, x.shape, sigma1[None, :])
                    z = x @ Hd1.T + n1
                    z_n = z / z_scale[None, :]
                    z_spm = _apply_spm(z_n, gamma_key) * z_scale[None, :]
                    n2 = _complex_awgn(rng_l, x.shape, sigma2[None, :])
                    return z_spm @ Hd2.T + n2

                y_train_full = _fwd(x_train, rng)
                y_test_full = _fwd(x_test, rng)

                # Bob decodes directly (no phase compensation); this is the
                # uncompensated case, establishing the operating regime boundary
                z_hat = y_test_full @ Hd2_inv.T
                z_inv = _invert_spm(z_hat / z_scale[None, :], gamma_key) * z_scale[None, :]
                x_hat_bob = z_inv @ Hd1_inv.T
                from mcf_pls.evaluation import _qpsk_demod_array, _bit_error_rate
                ber_bob = _bit_error_rate(bits_test, _qpsk_demod_array(x_hat_bob))

                if option == "FO":
                    y_eve_train = y_train_full.copy()
                    y_eve_test = y_test_full.copy()
                else:
                    y_eve_train, _ = _apply_wt_tap(y_train_full, geom, params, tap_cfg, rng)
                    y_eve_test, _ = _apply_wt_tap(y_test_full, geom, params, tap_cfg, rng)

                sigma2_eve = (params.sigma_noise * (tap_cfg.eve_noise_scale if option == "WT" else 1.0)) ** 2
                # Eve's known plaintext is the intended clean symbols, NOT the
                # phase-noisy transmitted signal (Eve cannot observe the transmitter
                # laser's random phase trajectory).
                n_obs = min(200, T_train)
                ber_ce = _evaluate_kpa_attack(
                    x_train_clean[:n_obs], y_eve_train[:n_obs], y_eve_test, bits_test, sigma2=sigma2_eve)
                ber_lr = _evaluate_lr_attack(
                    x_train_clean[:n_obs], y_eve_train[:n_obs], y_eve_test, bits_test)
                ber_lstm = _evaluate_lstm_attack(
                    y_eve_train, x_train_clean, y_eve_test, bits_test, window=8, epochs=6, seed=seed + 8200)

                row = {
                    "seed": seed,
                    "sigma_pn": sigma_pn,
                    "option": option,
                    "BER_Bob": ber_bob,
                    "BER_KPA_CE": ber_ce,
                    "BER_KPA_LR": ber_lr,
                    "BER_KPA_LSTM": ber_lstm,
                    "BER_Eve_best": min(ber_ce, ber_lr, ber_lstm),
                }
                rows.append(row)
                print(f"  seed={seed} σ_pn={sigma_pn:.2f} {option}: "
                      f"Bob={ber_bob:.4f} CE={ber_ce:.4f} LR={ber_lr:.4f} LSTM={ber_lstm:.4f}")

    _write_csv(f"{out_dir}/phase_noise_37core.csv", rows)

    summary = []
    for sigma_pn in sigma_pn_values:
        for option in ["FO", "WT"]:
            sub = [r for r in rows if r["sigma_pn"] == sigma_pn and r["option"] == option]
            bob_mean, bob_std, bob_ci = _mean_std_ci([r["BER_Bob"] for r in sub])
            eve_mean, eve_std, eve_ci = _mean_std_ci([r["BER_Eve_best"] for r in sub])
            summary.append({
                "sigma_pn": sigma_pn,
                "option": option,
                "BER_Bob_mean": bob_mean, "BER_Bob_std": bob_std, "BER_Bob_ci95": bob_ci,
                "BER_Eve_best_mean": eve_mean, "BER_Eve_best_std": eve_std, "BER_Eve_best_ci95": eve_ci,
            })
    _write_csv(f"{out_dir}/phase_noise_37core_summary.csv", summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, option in zip(axes, ["FO", "WT"]):
        sub = [r for r in summary if r["option"] == option]
        xs = [r["sigma_pn"] for r in sub]
        ax.errorbar(xs, [r["BER_Bob_mean"] for r in sub],
                    yerr=[r["BER_Bob_ci95"] for r in sub],
                    marker="o", color="tab:blue", lw=2, ms=6, capsize=4, label="Bob")
        ax.errorbar(xs, [r["BER_Eve_best_mean"] for r in sub],
                    yerr=[r["BER_Eve_best_ci95"] for r in sub],
                    marker="^", color="tab:red", lw=2, ms=6, capsize=4, label="Eve (best)")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
        ax.set_xlabel("Phase noise std σ_pn [rad/symbol]")
        ax.set_ylabel("BER")
        ax.set_title(f"{option}")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("37-core P2: BER vs. laser phase noise (Wiener process)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_phase_noise_37core.png", dpi=200)
    plt.close(fig)
    return rows


def sweep_lstm_window_ablation(base_params: FiberParams,
                               out_dir: str,
                               windows: Tuple[int, ...] = (4, 8, 16, 32),
                               seeds: Optional[List[int]] = None,
                               T_train: int = 1200,
                               T_test: int = 400) -> List[Dict]:
    """
    Supplementary S14: LSTM context-window ablation.

    Tests whether the LSTM attacker's BER is sensitive to the sliding-window
    length w ∈ {4, 8, 16, 32}.  Under a fixed-key memoryless channel, each
    output slot y_t is independent of y_{t-1}, y_{t-2}, ...; the window
    provides no temporal information beyond the current slot.  The ablation
    verifies this: BER_LSTM should be flat across w for the 37-core P2 regime.

    Also tests the Linear/FO baseline where a slight w-dependence could appear
    if the attack model overfits to accidental temporal correlations.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    tap_cfg = TapObservationConfig(
        tap_fraction=0.10,
        eve_noise_scale=1.5,
        observed_core_indices=None,
        label="outer-ring passive tap",
    )
    configs = [
        ("37-core P2", "Protected2", "FO"),
        ("37-core P2", "Protected2", "WT"),
        ("13-core Linear", "Linear", "FO"),
    ]
    rows = []
    print("\n[Supp S14] LSTM window ablation (w ∈ {4, 8, 16, 32})...")
    for seed in seeds:
        enc37, true_key37, _ = _make_37core_setup(base_params, seed)
        enc13 = MCFEncryption(base_params, MCFGeometry(pitch_um=base_params.pitch_um), seed=seed)
        true_key13 = np.random.default_rng(seed).uniform(0.6, 1.0, size=enc13.geom.N)

        for cfg_label, scheme, option in configs:
            if "37" in cfg_label:
                enc = enc37
                true_key = true_key37
            else:
                enc = enc13
                true_key = true_key13

            data = _build_dataset_for_scheme(
                enc, true_key, scheme, option, T_train, T_test,
                seed + 9000, tap_cfg)

            for w in windows:
                ber_lstm = _evaluate_lstm_attack(
                    data["y_eve_train"], data["x_train"],
                    data["y_eve_test"], data["bits_test"],
                    window=w, epochs=6, seed=seed + 9100 + w)
                row = {
                    "seed": seed,
                    "config": cfg_label,
                    "scheme": scheme,
                    "option": option,
                    "window": w,
                    "BER_LSTM": ber_lstm,
                }
                rows.append(row)
                print(f"  seed={seed} {cfg_label} {option} w={w:2d}: BER_LSTM={ber_lstm:.4f}")

    _write_csv(f"{out_dir}/lstm_window_ablation.csv", rows)

    summary = []
    for cfg_label, scheme, option in configs:
        for w in windows:
            sub = [r for r in rows
                   if r["config"] == cfg_label and r["option"] == option and r["window"] == w]
            mean, std, ci = _mean_std_ci([r["BER_LSTM"] for r in sub])
            summary.append({
                "config": cfg_label,
                "scheme": scheme,
                "option": option,
                "window": w,
                "BER_LSTM_mean": mean,
                "BER_LSTM_std": std,
                "BER_LSTM_ci95": ci,
            })
    _write_csv(f"{out_dir}/lstm_window_ablation_summary.csv", summary)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    styles = {
        ("37-core P2", "FO"): ("tab:green", "^", "37-core P2 / FO"),
        ("37-core P2", "WT"): ("darkgreen", "D", "37-core P2 / WT"),
        ("13-core Linear", "FO"): ("tab:red", "o", "13-core Linear / FO"),
    }
    for (cfg_label, option), (color, marker, label) in styles.items():
        sub = [r for r in summary if r["config"] == cfg_label and r["option"] == option]
        xs = [r["window"] for r in sub]
        ys = [r["BER_LSTM_mean"] for r in sub]
        errs = [r["BER_LSTM_ci95"] for r in sub]
        ax.errorbar(xs, ys, yerr=errs, marker=marker, linestyle="-", color=color,
                    lw=2, ms=6, capsize=4, label=label)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guess")
    ax.set_xlabel("LSTM context window w [symbols]")
    ax.set_ylabel("BER")
    ax.set_ylim(0, 0.62)
    ax.set_xticks(list(windows))
    ax.set_title("LSTM window-size ablation: BER vs. context window length\n"
                 "(flat curve = channel memoryless; w choice does not affect security conclusion)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_lstm_window_ablation.png", dpi=200)
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

## Supplementary figures

- `fig_qam_ber_protected2_37core.png`
  Tests whether the 37-core Protected2 regime preserves its security margin across QPSK, 16-QAM, and 64-QAM.

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
- `qam_ber_protected2_37core.csv`
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
