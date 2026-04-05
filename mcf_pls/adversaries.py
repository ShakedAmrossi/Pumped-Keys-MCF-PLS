"""Neural adversary models: EveRNN, EveMLP, EveTransformer, EveHybridMLP and trainers."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
