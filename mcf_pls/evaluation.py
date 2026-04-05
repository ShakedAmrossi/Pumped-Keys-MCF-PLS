"""Attack evaluators, dataset builders, observation models."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from mcf_pls.physics import FiberParams, MCFGeometry, MCFEncryption
from mcf_pls.nonlinear import _apply_spm, _invert_spm, _apply_xpm, _invert_xpm, _ase_noise_variances, _adjacency_weight_matrix
from mcf_pls.adversaries import train_eve_mlp, train_eve_transformer, train_eve_hybrid_mlp, train_eve_rnn

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
