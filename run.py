"""Publication suite entry point. Run: python run.py"""
from __future__ import annotations
import os
import numpy as np
from mcf_pls.physics import FiberParams, MCFGeometry, MCFEncryption
from mcf_pls.evaluation import TapObservationConfig, _resolve_tap_core_indices
from mcf_pls.utils import (_ensure, _write_csv, plot_mcf_geometry,
    validate_cme_simulator, plot_transfer_matrix, write_physical_calibration_bundle,
    sync_publication_artifacts)
from mcf_pls.benchmarks import (benchmark_fo_wt_attack_suite, sweep_kpa_lr_sample_complexity,
    sweep_lstm_training_size, benchmark_core_scaling_13_19_37, benchmark_xpm_exploratory,
    validate_37core_protected2_multiseed, sweep_37core_sample_complexity,
    sweep_37core_tap_sensitivity, benchmark_spm_xpm_combo_37core,
    sweep_qam_ber_protected2_37core, run_scaling_multiseed_significance,
    write_minimum_publishable_plan,
    sweep_fab_tolerance_37core, sweep_phase_noise_37core, sweep_lstm_window_ablation)
from mcf_pls.utils import write_submission_manifest

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
    fab_tol_rows = sweep_fab_tolerance_37core(params, out_dir)
    phase_noise_rows = sweep_phase_noise_37core(params, out_dir)
    lstm_window_rows = sweep_lstm_window_ablation(params, out_dir)

    write_minimum_publishable_plan(out_dir)
    write_submission_manifest(out_dir)
    sync_counts = sync_publication_artifacts(out_dir, base_dir)

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
    print(f"Supplementary QAM rows: {len(qam37_rows)}")
    print(f"Fab tolerance rows: {len(fab_tol_rows)}")
    print(f"Phase noise rows: {len(phase_noise_rows)}")
    print(f"LSTM window ablation rows: {len(lstm_window_rows)}")
    print(f"Synced article assets: {sync_counts['csv']} CSVs -> data/, {sync_counts['png']} PNGs -> figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
