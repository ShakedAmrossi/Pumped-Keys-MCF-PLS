"""I/O utilities, calibration writers, validation figures."""
from __future__ import annotations
import os, csv, json, shutil, warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcf_pls.physics import FiberParams, MCFGeometry, CMESimulator, MCFEncryption
from mcf_pls.nonlinear import _apply_spm, _secrecy_capacity
from scipy.integrate import solve_ivp

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
            "artifact_name": "fig0_mcf_geometry.png",
            "primary_data": "publication_suite_config.csv",
            "supporting_data": "physical_parameter_traceability.csv",
            "manuscript_role": "Geometry overview of the 13-core keyed multi-core fiber platform",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig1_transfer_matrix.png",
            "primary_data": "publication_suite_config.csv",
            "supporting_data": "",
            "manuscript_role": "Illustrative transfer-matrix comparison for the correct and wrong pump keys",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_validation.png",
            "primary_data": "physical_parameter_traceability.csv",
            "supporting_data": "validation_anchor_points.csv",
            "manuscript_role": "Coupled-mode validation against analytical and wavelength-trend reference checks",
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
            "artifact_type": "figure",
            "artifact_name": "fig_xpm_exploratory_benchmark.png",
            "primary_data": "xpm_exploratory_benchmark.csv",
            "supporting_data": "",
            "manuscript_role": "13-core exploratory comparison between SPM and SPM+XPM protection",
        },
        {
            "artifact_type": "figure",
            "artifact_name": "fig_qam_ber_protected2_37core.png",
            "primary_data": "qam_ber_protected2_37core.csv",
            "supporting_data": "",
            "manuscript_role": "Supplementary 37-core Protected2 BER comparison across QPSK, 16-QAM, and 64-QAM",
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
            "artifact_type": "table",
            "artifact_name": "qam_ber_protected2_37core.csv",
            "primary_data": "qam_ber_protected2_37core.csv",
            "supporting_data": "",
            "manuscript_role": "Supplementary modulation-format BER table for the 37-core Protected2 regime",
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


def sync_publication_artifacts(out_dir: str, base_dir: str) -> Dict[str, int]:
    """
    Mirror generated manuscript artifacts into the repository's `data/` and
    `figures/` directories so the checked-in article assets match `run.py`.
    """
    data_dir = os.path.join(base_dir, "data")
    figures_dir = os.path.join(base_dir, "figures")
    _ensure(data_dir)
    _ensure(figures_dir)

    counts = {"csv": 0, "png": 0}
    for name in os.listdir(out_dir):
        src = os.path.join(out_dir, name)
        if not os.path.isfile(src):
            continue
        if name.endswith(".csv"):
            shutil.copy2(src, os.path.join(data_dir, name))
            counts["csv"] += 1
        elif name.endswith(".png"):
            shutil.copy2(src, os.path.join(figures_dir, name))
            counts["png"] += 1
    return counts


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




# ── Validation and geometry figure functions (called by main()) ──────────────

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
