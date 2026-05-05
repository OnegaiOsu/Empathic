"""Data augmentation for tabular features and sequence tensors.

Two families are provided:

* :func:`augment_tabular` -- jitter, feature masking and class-balanced
  oversampling of the per-window feature vectors (used by classical models).
* :func:`augment_sequences` -- Um et al. (2017) time-series augmentations for
  the physiological / keystroke sequence tensors (used by the deep model).

All augmentations are applied *only* on the training fold; the test fold is
left untouched so that LOSO metrics remain trustworthy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------
def augment_tabular(
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "balance",
    noise_scale: float = 0.02,
    mask_prob: float = 0.03,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment the tabular training set.

    Modes:
      ``none``     -- return inputs unchanged.
      ``balance``  -- oversample minority classes up to the majority count.
      ``full``     -- balance + gaussian jitter + random feature masking.
    """
    mode = mode.lower().strip()
    if mode == "none":
        return X, y
    if mode not in {"balance", "full"}:
        raise ValueError(f"unknown mode: {mode}")

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    X_aug = [X]
    y_aug = [y]

    for c, n in zip(classes, counts):
        need = target - n
        if need <= 0:
            continue
        idx = np.where(y == c)[0]
        picks = rng.choice(idx, size=need, replace=True)
        X_new = X[picks].copy()
        if mode == "full":
            noise = rng.normal(0.0, noise_scale, size=X_new.shape).astype(X_new.dtype)
            mask = rng.random(X_new.shape) < mask_prob
            X_new = X_new + noise
            X_new[mask] = 0.0
        X_aug.append(X_new)
        y_aug.append(np.full(need, c, dtype=y.dtype))

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def _jitter(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0.0, sigma, size=x.shape).astype(x.dtype)


def _scaling(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    # One scale factor per channel (Um et al. 2017).
    factors = rng.normal(1.0, sigma, size=(1, x.shape[-1])).astype(x.dtype)
    return x * factors


def _channel_dropout(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Zero out each channel independently with probability ``p``. Input ``(L, C)``."""
    if p <= 0.0:
        return x
    keep = (rng.random(x.shape[-1]) >= p).astype(x.dtype)
    return x * keep[None, :]


def _time_warp(x: np.ndarray, sigma: float, knot: int, rng: np.random.Generator) -> np.ndarray:
    length = x.shape[0]
    orig_steps = np.arange(length)
    # Build a smooth random time warp via cubic interpolation of knot anchors.
    knot_x = np.linspace(0, length - 1, knot + 2)
    warps = rng.normal(1.0, sigma, size=knot + 2)
    warps = np.clip(warps, 0.5, 2.0)
    warp_time = np.interp(orig_steps, knot_x, np.cumsum(warps))
    warp_time = warp_time * (length - 1) / max(warp_time[-1], 1e-6)
    out = np.zeros_like(x)
    for c in range(x.shape[-1]):
        out[:, c] = np.interp(orig_steps, warp_time, x[:, c])
    return out


def augment_sequences(
    seq: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "balance",
    jitter_sigma: float = 0.03,
    scaling_sigma: float = 0.1,
    warp_sigma: float = 0.2,
    warp_knots: int = 4,
    channel_dropout: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment (N, L, C) sequence tensor.

    Modes match :func:`augment_tabular`; ``full`` stacks jitter + scaling +
    time-warp in addition to class balancing.
    """
    mode = mode.lower().strip()
    if mode == "none":
        return seq, y
    if mode not in {"balance", "full"}:
        raise ValueError(f"unknown mode: {mode}")

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    seq_aug = [seq]
    y_aug = [y]

    for c, n in zip(classes, counts):
        need = target - n
        if need <= 0:
            continue
        idx = np.where(y == c)[0]
        picks = rng.choice(idx, size=need, replace=True)
        base = seq[picks].copy()
        if mode == "full":
            out = np.empty_like(base)
            for i in range(base.shape[0]):
                x = base[i]
                x = _jitter(x, jitter_sigma, rng)
                x = _scaling(x, scaling_sigma, rng)
                x = _time_warp(x, warp_sigma, warp_knots, rng)
                x = _channel_dropout(x, channel_dropout, rng)
                out[i] = x
            base = out
        seq_aug.append(base)
        y_aug.append(np.full(need, c, dtype=y.dtype))

    return np.concatenate(seq_aug, axis=0), np.concatenate(y_aug, axis=0)


# ---------------------------------------------------------------------------
# MixUp (for deep-model training batches)
# ---------------------------------------------------------------------------
def mixup_batch(x, y, alpha: float = 0.2):
    """Apply MixUp (Zhang et al., 2018) to a torch batch.

    Returns ``(x_mixed, y_a, y_b, lam)``. The training loop combines the two
    cross-entropy losses as ``lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)``.
    MixUp interpolates between subjects/classes at the input level, which is
    especially helpful when minority classes are under-represented per subject.
    """
    import numpy as _np
    import torch as _torch

    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(_np.random.beta(alpha, alpha))
    idx = _torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1.0 - lam) * x[idx]
    return x_m, y, y[idx], lam
