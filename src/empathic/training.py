"""LOSO training harness and deep-model trainer.

Key design points:

* **Leave-One-Subject-Out (LOSO)** is the standard protocol for both
  corpora: each fold holds out one participant so that the reported accuracy
  reflects cross-person generalisation rather than window-level memorisation.
* **Feature cleanup happens inside the fold** (imputation lives inside each
  model pipeline) so training-time statistics never leak to the held-out
  subject.
* Augmentation is applied **only** on the training fold.
* The deep model is re-instantiated per fold, moved to ``DEVICE`` and trained
  with AdamW + cosine LR + early stopping based on training loss plateau.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .augment import augment_sequences, augment_tabular, mixup_batch
from .config import DEVICE, RESULTS_DIR, TrainDefaults
from .data import DatasetBundle, QUADRANTS
from .evaluation import (
    ClassificationMetrics,
    aggregate_fold_metrics,
    build_summary_frame,
    compute_classification_metrics,
    pool_session_predictions,
    save_metrics,
)
from .models import (
    BiLSTM,
    BiLSTMConfig,
    CNN1D,
    CNN1DConfig,
    ClassicalModel,
    Conformer,
    ConformerConfig,
    DANNConformer,
    DANNConformerConfig,
    MLPProbe,
    MultiStream,
    MultiStreamConfig,
    TinyTCN,
    TinyTCNConfig,
    TSTCCConfig,
    TSTCCEncoder,
    count_parameters,
    default_classical_models,
    nt_xent_loss,
    temporal_contrast_loss,
    tstcc_strong_aug,
    tstcc_weak_aug,
)
from .plotting import ensure_results_dir, plot_confusion, plot_model_comparison, plot_subject_f1
from .utils import align_feature_matrix, ensure_dir, log, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unique_subjects(subject_ids: np.ndarray) -> List[str]:
    return sorted({str(s) for s in subject_ids})


def _label_variance_report(
    target_kind: str,
    y_all: np.ndarray,
    subject_ids_all: np.ndarray,
    labels: List[str],
    verbose: bool,
) -> List[str]:
    """Log per-subject label distribution and return list of subjects that are
    *not* eligible as LOSO test folds (single-class test sets break macro-F1
    for binary targets).

    For 2-class targets we exclude rows with y<0 (NEU) before counting.
    Returns the list of single-class subject_ids.
    """
    valid = y_all >= 0
    y = y_all[valid]
    subj = subject_ids_all[valid]
    subjects = _unique_subjects(subj)
    single_class: List[str] = []
    multi_class: List[str] = []
    for s in subjects:
        ys = y[subj == s]
        if len(np.unique(ys)) < 2:
            single_class.append(s)
        else:
            multi_class.append(s)
    if verbose:
        log(f"  [label-variance] target={target_kind} labels={labels}", verbose)
        log(f"  [label-variance] multi-class subjects: {len(multi_class)}/{len(subjects)} "
            f"-> usable for LOSO evaluation", verbose)
        if single_class:
            log(f"  [label-variance] single-class subjects ({len(single_class)}): "
                f"{single_class} -> excluded from per-fold metrics", verbose)
    return single_class


def _label_list(target_kind: str, bundle: DatasetBundle) -> List[str]:
    if target_kind == "quadrant":
        return list(bundle.quadrant_labels)
    if target_kind == "native":
        return list(bundle.native_labels)
    if target_kind == "valence":
        return ["LV", "HV"]
    if target_kind == "arousal":
        return ["LA", "HA"]
    if target_kind == "stress":
        return ["low_stress", "high_stress"]
    raise ValueError(f"unknown target_kind: {target_kind}")


# Map quadrant label -> (valence_bit, arousal_bit). NEU is intentionally
# absent; rows labelled NEU are excluded from binary axis targets.
_QUADRANT_TO_VA_BITS = {
    "HVHA": (1, 1),
    "HVLA": (1, 0),
    "LVHA": (0, 1),
    "LVLA": (0, 0),
}


def _select_target(bundle: DatasetBundle, target_kind: str) -> np.ndarray:
    if target_kind == "quadrant":
        return bundle.quadrant_target
    if target_kind == "native":
        return bundle.native_target
    if target_kind == "stress":
        if bundle.stress is None:
            raise ValueError(f"dataset {bundle.name} has no stress target")
        return bundle.stress.astype(np.int64)
    if target_kind in {"valence", "arousal"}:
        axis = 0 if target_kind == "valence" else 1
        labels = list(bundle.quadrant_labels)
        q = bundle.quadrant_target
        out = np.full(len(q), -1, dtype=np.int64)
        for i, name in enumerate(labels):
            bits = _QUADRANT_TO_VA_BITS.get(name)
            if bits is None:
                continue
            out[q == i] = bits[axis]
        return out
    raise ValueError(f"unknown target_kind: {target_kind}")


# ---------------------------------------------------------------------------
# Deep training loop
# ---------------------------------------------------------------------------
def _build_deep_model(arch: str, in_channels: int, num_classes: int, seq_len: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "conformer":
        cfg = ConformerConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=128,
            num_blocks=4,
            heads=4,
            conv_kernel=15,
            dropout=0.15,
            seq_stride=2 if seq_len >= 64 else 1,
            max_len=seq_len + 8,
        )
        return Conformer(cfg)
    if arch == "tiny_tcn":
        cfg = TinyTCNConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=64,
            num_blocks=4,
            kernel_size=5,
            dropout=0.3,
        )
        return TinyTCN(cfg)
    if arch == "bilstm":
        cfg = BiLSTMConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden=96,
            num_layers=2,
            dropout=0.3,
        )
        return BiLSTM(cfg)
    if arch == "cnn1d":
        cfg = CNN1DConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=64,
            num_blocks=4,
            kernel_size=7,
            dropout=0.2,
        )
        return CNN1D(cfg)
    if arch == "multistream":
        cfg = MultiStreamConfig(
            in_channels=in_channels,
            num_classes=num_classes,
        )
        return MultiStream(cfg)
    raise ValueError(f"unknown deep arch: {arch}")


class _LateFusionDeep(nn.Module):
    """Wraps any deep model with ``forward_features`` to fuse hand-features.

    Concatenates the sequence-encoder embedding with a small MLP-projected
    tabular feature vector, and routes the joint vector to a fresh classifier
    head. Replaces the base model's head so the classifier sees both modalities.
    """

    def __init__(self, base: nn.Module, feat_dim: int, num_classes: int,
                 fuse_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.base = base
        emb_dim = int(getattr(base, "embed_dim"))
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, fuse_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim),
            nn.SiLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(emb_dim + fuse_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim + fuse_dim, num_classes),
        )
        # Disable the base model's head; we route through ``fusion_head``.
        if hasattr(base, "head"):
            base.head = nn.Identity()

    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        emb = self.base.forward_features(x_seq)
        f = self.feat_proj(x_feat)
        return self.fusion_head(torch.cat([emb, f], dim=-1))


def _train_deep_fold(
    seq_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray,
    y_val: np.ndarray,
    *,
    num_classes: int,
    cfg: TrainDefaults,
    arch: str = "conformer",
    mixup_alpha: float = 0.0,
    class_weights: Optional[np.ndarray] = None,
    feat_train: Optional[np.ndarray] = None,
    feat_val: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Train one deep-model LOSO fold.

    Returns ``(preds, probas, param_count, arch_name)``.
    """
    base = _build_deep_model(
        arch, in_channels=seq_train.shape[-1], num_classes=num_classes,
        seq_len=max(seq_train.shape[1], seq_val.shape[1]),
    )
    use_fusion = feat_train is not None and feat_val is not None
    if use_fusion:
        model = _LateFusionDeep(
            base, feat_dim=feat_train.shape[1], num_classes=num_classes
        ).to(DEVICE)
        arch_label = f"{arch}_fusion"
    else:
        model = base.to(DEVICE)
        arch_label = arch
    params = count_parameters(model)

    x_train_t = torch.from_numpy(seq_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(seq_val).float().to(DEVICE)
    if use_fusion:
        f_train_t = torch.from_numpy(feat_train).float()
        f_val_t = torch.from_numpy(feat_val).float().to(DEVICE)
        loader = DataLoader(
            TensorDataset(x_train_t, f_train_t, y_train_t),
            batch_size=cfg.deep_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
        )
    else:
        loader = DataLoader(
            TensorDataset(x_train_t, y_train_t),
            batch_size=cfg.deep_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
        )

    if class_weights is not None:
        w = torch.from_numpy(class_weights).float().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.deep_lr, weight_decay=cfg.deep_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.deep_epochs)

    best_loss = float("inf")
    plateau = 0

    for epoch in range(cfg.deep_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            if use_fusion:
                xb, fb, yb = batch
                xb = xb.to(DEVICE, non_blocking=True)
                fb = fb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
            else:
                xb, yb = batch
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if mixup_alpha > 0.0:
                if use_fusion:
                    # Mix sequence and feature tensors with the same lambda.
                    idx = torch.randperm(xb.size(0), device=xb.device)
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    xb_m = lam * xb + (1.0 - lam) * xb[idx]
                    fb_m = lam * fb + (1.0 - lam) * fb[idx]
                    yb_a, yb_b = yb, yb[idx]
                    logits = model(xb_m, fb_m)
                    loss = lam * criterion(logits, yb_a) + (1.0 - lam) * criterion(logits, yb_b)
                else:
                    xb_m, yb_a, yb_b, lam = mixup_batch(xb, yb, alpha=mixup_alpha)
                    logits = model(xb_m)
                    loss = lam * criterion(logits, yb_a) + (1.0 - lam) * criterion(logits, yb_b)
            else:
                logits = model(xb, fb) if use_fusion else model(xb)
                loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        sched.step()
        avg = epoch_loss / max(n, 1)
        if verbose:
            log(f"  epoch {epoch + 1:02d}/{cfg.deep_epochs}  train_loss={avg:.4f}", verbose)
        if avg + 1e-4 < best_loss:
            best_loss = avg
            plateau = 0
        else:
            plateau += 1
            if plateau >= cfg.early_stop_patience:
                break

    model.eval()
    with torch.no_grad():
        preds_all: List[np.ndarray] = []
        probs_all: List[np.ndarray] = []
        for i in range(0, x_val_t.size(0), cfg.deep_batch_size):
            chunk = x_val_t[i : i + cfg.deep_batch_size]
            if use_fusion:
                fchunk = f_val_t[i : i + cfg.deep_batch_size]
                logits = model(chunk, fchunk)
            else:
                logits = model(chunk)
            probs = torch.softmax(logits, dim=-1)
            preds_all.append(logits.argmax(dim=-1).cpu().numpy())
            probs_all.append(probs.cpu().numpy())
        preds = np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=np.int64)
        probs = (
            np.concatenate(probs_all, axis=0)
            if probs_all
            else np.zeros((0, num_classes), dtype=np.float32)
        )

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return preds, probs, params, arch_label


# Backward-compatible alias used in older scripts.
_train_conformer_fold = _train_deep_fold


# ---------------------------------------------------------------------------
# TS-TCC: self-supervised pretraining + frozen-encoder MLP probe
# ---------------------------------------------------------------------------
def _train_tstcc_fold(
    seq_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray,
    y_val: np.ndarray,
    *,
    num_classes: int,
    cfg: TrainDefaults,
    class_weights: Optional[np.ndarray] = None,
    pretrain_epochs: int = 40,
    probe_epochs: Optional[int] = None,
    temperature_temporal: float = 0.5,
    temperature_context: float = 0.2,
    lambda_temporal: float = 1.0,
    lambda_context: float = 0.7,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Pretrain a TS-TCC encoder with contrastive losses then probe with an MLP."""
    in_channels = seq_train.shape[-1]
    enc_cfg = TSTCCConfig(in_channels=in_channels, num_classes=num_classes)
    encoder = TSTCCEncoder(enc_cfg).to(DEVICE)
    probe = MLPProbe(enc_cfg.feature_dim, num_classes=num_classes).to(DEVICE)
    params = count_parameters(encoder) + count_parameters(probe)

    x_train_t = torch.from_numpy(seq_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(seq_val).float().to(DEVICE)

    pre_loader = DataLoader(
        TensorDataset(x_train_t),
        batch_size=cfg.deep_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    pre_opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.deep_lr, weight_decay=cfg.deep_weight_decay)
    pre_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pre_opt, T_max=pretrain_epochs)

    encoder.train()
    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0
        n = 0
        for (xb,) in pre_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            x_bcl = xb.transpose(1, 2)                                  # (B, C, L)
            x_w = tstcc_weak_aug(x_bcl)
            x_s = tstcc_strong_aug(x_bcl)
            f_w, c_w = encoder(x_w)
            f_s, c_s = encoder(x_s)
            T = min(f_w.size(-1), f_s.size(-1))
            loss_t = temporal_contrast_loss(
                f_w[..., :T], f_s[..., :T], temperature=temperature_temporal
            )
            z_w = encoder.project(c_w)
            z_s = encoder.project(c_s)
            loss_c = nt_xent_loss(z_w, z_s, temperature=temperature_context)
            loss = lambda_temporal * loss_t + lambda_context * loss_c
            pre_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            pre_opt.step()
            epoch_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        pre_sched.step()
        if verbose and (epoch + 1) % 10 == 0:
            log(f"  [tstcc-pretrain] epoch {epoch + 1:02d}/{pretrain_epochs}  loss={epoch_loss / max(n, 1):.4f}", verbose)

    # Freeze encoder, extract context vectors, train probe.
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    def _embed(x_blc: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, ctx = encoder(x_blc.to(DEVICE))
        return ctx

    train_ctx_chunks: List[torch.Tensor] = []
    bs = cfg.deep_batch_size
    for i in range(0, x_train_t.size(0), bs):
        train_ctx_chunks.append(_embed(x_train_t[i : i + bs]))
    train_ctx = torch.cat(train_ctx_chunks, dim=0)
    train_y = y_train_t.to(DEVICE)

    val_ctx_chunks: List[torch.Tensor] = []
    for i in range(0, x_val_t.size(0), bs):
        val_ctx_chunks.append(_embed(x_val_t[i : i + bs]))
    val_ctx = torch.cat(val_ctx_chunks, dim=0) if val_ctx_chunks else x_val_t.new_zeros((0, enc_cfg.feature_dim))

    probe_loader = DataLoader(
        TensorDataset(train_ctx.cpu(), train_y.cpu()),
        batch_size=cfg.deep_batch_size,
        shuffle=True,
        drop_last=False,
    )
    if class_weights is not None:
        w = torch.from_numpy(class_weights).float().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    probe_epochs = probe_epochs or cfg.deep_epochs
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=cfg.deep_lr, weight_decay=cfg.deep_weight_decay)
    probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt, T_max=probe_epochs)
    best_loss = float("inf")
    plateau = 0
    for epoch in range(probe_epochs):
        probe.train()
        epoch_loss = 0.0
        n = 0
        for ctx_b, yb in probe_loader:
            ctx_b = ctx_b.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            probe_opt.zero_grad(set_to_none=True)
            logits = probe(ctx_b)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            probe_opt.step()
            epoch_loss += float(loss.item()) * ctx_b.size(0)
            n += ctx_b.size(0)
        probe_sched.step()
        avg = epoch_loss / max(n, 1)
        if avg + 1e-4 < best_loss:
            best_loss = avg
            plateau = 0
        else:
            plateau += 1
            if plateau >= cfg.early_stop_patience:
                break

    probe.eval()
    with torch.no_grad():
        if val_ctx.size(0) == 0:
            preds = np.array([], dtype=np.int64)
            probs = np.zeros((0, num_classes), dtype=np.float32)
        else:
            logits = probe(val_ctx)
            probs_t = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()
            probs = probs_t.cpu().numpy().astype(np.float32)

    del encoder, probe
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return preds, probs, params, "tstcc"


# ---------------------------------------------------------------------------
# DANN: domain-adversarial training with subject discriminator
# ---------------------------------------------------------------------------
def _train_dann_fold(
    seq_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    seq_val: np.ndarray,
    y_val: np.ndarray,
    *,
    num_classes: int,
    cfg: TrainDefaults,
    mixup_alpha: float = 0.0,
    class_weights: Optional[np.ndarray] = None,
    lambda_max: float = 0.5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Train a DANN-Conformer fold with adversarial subject head."""
    unique_subj = sorted({str(s) for s in subj_train.tolist()})
    subj_to_idx = {s: i for i, s in enumerate(unique_subj)}
    s_train = np.array([subj_to_idx[str(s)] for s in subj_train], dtype=np.int64)
    num_subjects = len(unique_subj)

    seq_len = max(seq_train.shape[1], seq_val.shape[1])
    dcfg = DANNConformerConfig(
        in_channels=seq_train.shape[-1],
        num_classes=num_classes,
        num_subjects=num_subjects,
        seq_stride=2 if seq_len >= 64 else 1,
        max_len=seq_len + 8,
    )
    model = DANNConformer(dcfg).to(DEVICE)
    params = count_parameters(model)

    x_train_t = torch.from_numpy(seq_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    s_train_t = torch.from_numpy(s_train).long()
    x_val_t = torch.from_numpy(seq_val).float().to(DEVICE)

    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t, s_train_t),
        batch_size=cfg.deep_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    if class_weights is not None:
        w = torch.from_numpy(class_weights).float().to(DEVICE)
        cls_criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    else:
        cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    subj_criterion = nn.CrossEntropyLoss()

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.deep_lr, weight_decay=cfg.deep_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.deep_epochs)

    total_steps = max(1, cfg.deep_epochs * max(1, len(loader)))
    step = 0
    best_loss = float("inf")
    plateau = 0
    for epoch in range(cfg.deep_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb, sb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            sb = sb.to(DEVICE, non_blocking=True)
            p = step / total_steps
            lam = lambda_max * (2.0 / (1.0 + float(np.exp(-10.0 * p))) - 1.0)
            model.set_lambda(lam)

            optim.zero_grad(set_to_none=True)
            if mixup_alpha > 0.0:
                xb_m, yb_a, yb_b, mix_lam = mixup_batch(xb, yb, alpha=mixup_alpha)
                cls_logits, subj_logits = model(xb_m)
                loss_cls = mix_lam * cls_criterion(cls_logits, yb_a) + (1.0 - mix_lam) * cls_criterion(cls_logits, yb_b)
            else:
                cls_logits, subj_logits = model(xb)
                loss_cls = cls_criterion(cls_logits, yb)
            loss_subj = subj_criterion(subj_logits, sb)
            loss = loss_cls + loss_subj
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += float(loss_cls.item()) * xb.size(0)
            n += xb.size(0)
            step += 1
        sched.step()
        avg = epoch_loss / max(n, 1)
        if verbose:
            log(f"  [dann] epoch {epoch + 1:02d}/{cfg.deep_epochs}  cls_loss={avg:.4f}  lambda={lam:.3f}", verbose)
        if avg + 1e-4 < best_loss:
            best_loss = avg
            plateau = 0
        else:
            plateau += 1
            if plateau >= cfg.early_stop_patience:
                break

    model.eval()
    preds_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x_val_t.size(0), cfg.deep_batch_size):
            chunk = x_val_t[i : i + cfg.deep_batch_size]
            cls_logits, _ = model(chunk)
            probs = torch.softmax(cls_logits, dim=-1)
            preds_all.append(cls_logits.argmax(dim=-1).cpu().numpy())
            probs_all.append(probs.cpu().numpy())
    preds = np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=np.int64)
    probs = (
        np.concatenate(probs_all, axis=0)
        if probs_all
        else np.zeros((0, num_classes), dtype=np.float32)
    )
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return preds, probs, params, "dann"


# ---------------------------------------------------------------------------
# LOSO orchestrator
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    name: str
    labels: List[str]
    per_fold_metrics: List[ClassificationMetrics]
    subjects: List[str]
    overall: ClassificationMetrics
    aggregate: Dict[str, float]
    extra: Dict[str, object]
    session_overall: Optional[ClassificationMetrics] = None
    session_aggregate: Optional[Dict[str, float]] = None
    # Fold-aligned arrays kept so an ensemble can pool probabilities post-hoc.
    y_true_concat: Optional[np.ndarray] = None
    proba_concat: Optional[np.ndarray] = None
    session_concat: Optional[np.ndarray] = None
    subject_concat: Optional[np.ndarray] = None


def _inverse_freq_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Return per-sample weights = (N / (K * count[class_of_sample]))."""
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w_per_class = counts.sum() / (num_classes * counts)
    return w_per_class[y.astype(int)].astype(np.float32)


def _classical_loso(
    bundle: DatasetBundle,
    target_kind: str,
    model_factory: Callable[[], ClassicalModel],
    augment_mode: str,
    seed: int,
    verbose: bool,
) -> RunResult:
    labels = _label_list(target_kind, bundle)
    y_all = _select_target(bundle, target_kind)
    X_all = align_feature_matrix(bundle.samples, bundle.feature_cols)
    subjects = _unique_subjects(bundle.subject_ids)
    session_keys_all = bundle.session_key
    subject_ids_all = bundle.subject_ids

    # Binary targets exclude rows that have no valence/arousal mapping (NEU).
    valid = y_all >= 0
    if not valid.all():
        X_all = X_all[valid]
        y_all = y_all[valid]
        session_keys_all = session_keys_all[valid]
        subject_ids_all = subject_ids_all[valid]
        subjects = _unique_subjects(subject_ids_all)

    per_fold: List[ClassificationMetrics] = []
    y_true_cat: List[np.ndarray] = []
    y_pred_cat: List[np.ndarray] = []
    proba_cat: List[np.ndarray] = []
    session_cat: List[np.ndarray] = []
    subject_cat: List[np.ndarray] = []

    name = model_factory().name
    is_binary = len(labels) == 2

    for subject in subjects:
        test_mask = subject_ids_all == subject
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if len(np.unique(y_all[train_mask])) < 2:
            continue
        # Single-class test fold: macro-F1 is undefined for the absent class
        # (sklearn returns 0 + division warnings). Skip per-fold scoring for
        # binary targets; pooled metrics still include these rows.
        if is_binary and len(np.unique(y_all[test_mask])) < 2:
            if verbose:
                log(f"  [{model_factory().name}] subject={subject}  SKIP (single-class test)", verbose)
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        X_train_aug, y_train_aug = augment_tabular(
            X_train, y_train, mode=augment_mode, seed=seed
        )

        model = model_factory()
        # Inverse-frequency sample weights for estimators that support it
        # (XGBoost). RandomForest / LogisticRegression already take
        # ``class_weight="balanced"``; the Baseline ignores weights.
        if model.supports_sample_weight:
            sw = _inverse_freq_weights(y_train_aug, num_classes=len(labels))
            model.fit(X_train_aug, y_train_aug, sample_weight=sw)
        else:
            model.fit(X_train_aug, y_train_aug)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        if proba is None:
            # One-hot fallback if the estimator lacks predict_proba.
            proba = np.zeros((len(preds), len(labels)), dtype=np.float32)
            proba[np.arange(len(preds)), preds.astype(int)] = 1.0
        else:
            # Some folds may not see every class during training; sklearn
            # estimators then return a (n, k_seen) matrix. Pad to (n, len(labels))
            # so the downstream ensemble can align with deep-model outputs.
            classes_seen = getattr(model, "classes_", None)
            if classes_seen is not None and proba.shape[1] != len(labels):
                full = np.zeros((proba.shape[0], len(labels)), dtype=np.float32)
                for col, cls in enumerate(classes_seen):
                    cls_int = int(cls)
                    if 0 <= cls_int < len(labels):
                        full[:, cls_int] = proba[:, col]
                proba = full

        m = compute_classification_metrics(y_test, preds, labels)
        per_fold.append(m)
        y_true_cat.append(y_test)
        y_pred_cat.append(preds)
        proba_cat.append(proba.astype(np.float32))
        session_cat.append(session_keys_all[test_mask])
        subject_cat.append(np.full(len(y_test), subject, dtype=object))
        if verbose:
            log(f"  [{name}] subject={subject}  acc={m.accuracy:.3f}  f1={m.macro_f1:.3f}", verbose)

    y_true = np.concatenate(y_true_cat) if y_true_cat else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_cat) if y_pred_cat else np.array([], dtype=int)
    overall = compute_classification_metrics(y_true, y_pred, labels)
    agg = aggregate_fold_metrics(per_fold)

    session_overall = None
    session_aggregate = None
    if proba_cat:
        probs = np.concatenate(proba_cat, axis=0)
        keys = np.concatenate(session_cat, axis=0)
        y_t_s, y_p_s, _ = pool_session_predictions(y_true, probs, keys)
        session_overall = compute_classification_metrics(y_t_s, y_p_s, labels)
        session_aggregate = {
            "session_accuracy": session_overall.accuracy,
            "session_balanced_accuracy": session_overall.balanced_accuracy,
            "session_macro_f1": session_overall.macro_f1,
            "session_cohen_kappa": session_overall.cohen_kappa,
            "n_sessions": int(len(y_t_s)),
        }

    return RunResult(
        name=name,
        labels=labels,
        per_fold_metrics=per_fold,
        subjects=subjects[: len(per_fold)],
        overall=overall,
        aggregate=agg,
        extra={"model_type": "classical"},
        session_overall=session_overall,
        session_aggregate=session_aggregate,
        y_true_concat=y_true,
        proba_concat=np.concatenate(proba_cat, axis=0) if proba_cat else None,
        session_concat=np.concatenate(session_cat, axis=0) if session_cat else None,
        subject_concat=np.concatenate(subject_cat, axis=0) if subject_cat else None,
    )


def _deep_loso(
    bundle: DatasetBundle,
    target_kind: str,
    *,
    augment_mode: str,
    cfg: TrainDefaults,
    arch: str = "conformer",
    mixup_alpha: float = 0.0,
    fusion: bool = False,
    verbose: bool = False,
) -> RunResult:
    labels = _label_list(target_kind, bundle)
    y_all = _select_target(bundle, target_kind)
    seqs = bundle.sequences
    subjects = _unique_subjects(bundle.subject_ids)
    session_keys_all = bundle.session_key
    subject_ids_all = bundle.subject_ids

    # Pre-extract tabular features (already z-scored at load time) for
    # late-fusion. Only enabled for the 4 base archs (conformer, tiny_tcn,
    # bilstm, cnn1d); DANN/TSTCC have specialised forward passes.
    feats_all: Optional[np.ndarray] = None
    arch_lc_check = arch.lower().strip()
    if fusion and arch_lc_check not in {"dann", "tstcc"}:
        feats_all = align_feature_matrix(bundle.samples, bundle.feature_cols).astype(np.float32)
        # Replace any residual NaN/Inf so the deep head doesn't poison.
        feats_all = np.nan_to_num(feats_all, nan=0.0, posinf=0.0, neginf=0.0)

    valid = y_all >= 0
    if not valid.all():
        seqs = seqs[valid]
        y_all = y_all[valid]
        session_keys_all = session_keys_all[valid]
        subject_ids_all = subject_ids_all[valid]
        subjects = _unique_subjects(subject_ids_all)
        if feats_all is not None:
            feats_all = feats_all[valid]

    per_fold: List[ClassificationMetrics] = []
    y_true_cat: List[np.ndarray] = []
    y_pred_cat: List[np.ndarray] = []
    proba_cat: List[np.ndarray] = []
    session_cat: List[np.ndarray] = []
    subject_cat: List[np.ndarray] = []
    param_count = 0
    model_name = arch
    is_binary = len(labels) == 2

    for subject in subjects:
        test_mask = subject_ids_all == subject
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if len(np.unique(y_all[train_mask])) < 2:
            continue
        if is_binary and len(np.unique(y_all[test_mask])) < 2:
            if verbose:
                log(f"  [{arch}] subject={subject}  SKIP (single-class test)", verbose)
            continue

        s_train, y_train = seqs[train_mask], y_all[train_mask]
        s_test, y_test = seqs[test_mask], y_all[test_mask]
        subj_train = subject_ids_all[train_mask]
        if feats_all is not None:
            f_train = feats_all[train_mask]
            f_test = feats_all[test_mask]
        else:
            f_train = None
            f_test = None

        s_train_aug, y_train_aug = augment_sequences(
            s_train, y_train, mode=augment_mode, seed=cfg.seed
        )
        # Aligned subject-id array. ``augment_sequences`` may resample/balance,
        # so for DANN we re-derive subject ids by tiling the originals (the
        # adversary only needs same-distribution domain labels, not exact
        # provenance per augmented copy).
        if len(s_train_aug) == len(subj_train):
            subj_train_aug = np.array(subj_train, dtype=object)
            f_train_aug = f_train
        else:
            reps = int(np.ceil(len(s_train_aug) / max(1, len(subj_train))))
            subj_train_aug = np.tile(np.array(subj_train, dtype=object), reps)[: len(s_train_aug)]
            if f_train is not None:
                f_train_aug = np.tile(f_train, (reps, 1))[: len(s_train_aug)]
            else:
                f_train_aug = None

        classes, counts = np.unique(y_train_aug, return_counts=True)
        weights_full = np.ones(len(labels), dtype=np.float32)
        if len(classes) == len(labels):
            inv = counts.max() / np.maximum(counts, 1)
            for c, w in zip(classes, inv):
                weights_full[int(c)] = float(w)

        arch_lc = arch.lower().strip()
        if arch_lc == "tstcc":
            preds, probs, param_count, model_name = _train_tstcc_fold(
                s_train_aug.astype(np.float32),
                y_train_aug.astype(np.int64),
                s_test.astype(np.float32),
                y_test.astype(np.int64),
                num_classes=len(labels),
                cfg=cfg,
                class_weights=weights_full,
                verbose=verbose,
            )
        elif arch_lc == "dann":
            preds, probs, param_count, model_name = _train_dann_fold(
                s_train_aug.astype(np.float32),
                y_train_aug.astype(np.int64),
                subj_train_aug,
                s_test.astype(np.float32),
                y_test.astype(np.int64),
                num_classes=len(labels),
                cfg=cfg,
                mixup_alpha=mixup_alpha,
                class_weights=weights_full,
                verbose=False,
            )
        else:
            preds, probs, param_count, model_name = _train_deep_fold(
                s_train_aug.astype(np.float32),
                y_train_aug.astype(np.int64),
                s_test.astype(np.float32),
                y_test.astype(np.int64),
                num_classes=len(labels),
                cfg=cfg,
                arch=arch,
                mixup_alpha=mixup_alpha,
                class_weights=weights_full,
                feat_train=(f_train_aug.astype(np.float32) if f_train_aug is not None else None),
                feat_val=(f_test.astype(np.float32) if f_test is not None else None),
                verbose=False,
            )
        m = compute_classification_metrics(y_test, preds, labels)
        per_fold.append(m)
        y_true_cat.append(y_test)
        y_pred_cat.append(preds)
        proba_cat.append(probs.astype(np.float32))
        session_cat.append(session_keys_all[test_mask])
        subject_cat.append(np.full(len(y_test), subject, dtype=object))
        if verbose:
            log(f"  [{model_name}] subject={subject}  acc={m.accuracy:.3f}  f1={m.macro_f1:.3f}", verbose)

    y_true = np.concatenate(y_true_cat) if y_true_cat else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_cat) if y_pred_cat else np.array([], dtype=int)
    overall = compute_classification_metrics(y_true, y_pred, labels)
    agg = aggregate_fold_metrics(per_fold)

    session_overall = None
    session_aggregate = None
    if proba_cat:
        probs = np.concatenate(proba_cat, axis=0)
        keys = np.concatenate(session_cat, axis=0)
        y_t_s, y_p_s, _ = pool_session_predictions(y_true, probs, keys)
        session_overall = compute_classification_metrics(y_t_s, y_p_s, labels)
        session_aggregate = {
            "session_accuracy": session_overall.accuracy,
            "session_balanced_accuracy": session_overall.balanced_accuracy,
            "session_macro_f1": session_overall.macro_f1,
            "session_cohen_kappa": session_overall.cohen_kappa,
            "n_sessions": int(len(y_t_s)),
        }

    nice_name = {
        "conformer": "Conformer",
        "tiny_tcn": "TinyTCN",
        "bilstm": "BiLSTM",
        "cnn1d": "CNN1D",
        "tstcc": "TSTCC",
        "dann": "DANN_Conformer",
    }.get(model_name, model_name)
    return RunResult(
        name=nice_name,
        labels=labels,
        per_fold_metrics=per_fold,
        subjects=subjects[: len(per_fold)],
        overall=overall,
        aggregate=agg,
        extra={"model_type": "deep", "arch": model_name, "parameters": param_count, "device": str(DEVICE), "mixup_alpha": mixup_alpha},
        session_overall=session_overall,
        session_aggregate=session_aggregate,
        y_true_concat=y_true,
        proba_concat=np.concatenate(proba_cat, axis=0) if proba_cat else None,
        session_concat=np.concatenate(session_cat, axis=0) if session_cat else None,
        subject_concat=np.concatenate(subject_cat, axis=0) if subject_cat else None,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_experiment(
    bundle: DatasetBundle,
    *,
    target_kind: str = "quadrant",
    seed: int = 42,
    use_gpu: bool = True,
    augment_tabular_mode: str = "balance",
    augment_sequences_mode: str = "balance",
    include_deep: bool = True,
    include_classical: bool = True,
    deep_arch: "str | List[str]" = "conformer",
    mixup_alpha: float = 0.0,
    fusion: bool = False,
    ensemble_members: Optional[Sequence[str]] = None,
    train_cfg: Optional[TrainDefaults] = None,
    results_root: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, RunResult]:
    """Run LOSO with all classical models + the deep model on one bundle."""
    set_seed(seed)
    cfg = train_cfg or TrainDefaults(seed=seed)
    results_root = results_root or RESULTS_DIR
    out_dir = ensure_results_dir(results_root, bundle.name, target_kind)

    log(f"=== {bundle.name.upper()} / target={target_kind} ===", verbose)
    log(f"  samples={len(bundle.samples)}  subjects={len(_unique_subjects(bundle.subject_ids))}  labels={_label_list(target_kind, bundle)}", verbose)
    log(f"  feature_cols={len(bundle.feature_cols)}  seq_shape={bundle.sequences.shape}", verbose)

    # Per-subject label-variance diagnostic (logs which subjects are usable
    # as LOSO folds for this target).
    _label_variance_report(
        target_kind,
        _select_target(bundle, target_kind),
        bundle.subject_ids,
        _label_list(target_kind, bundle),
        verbose=verbose,
    )

    classical = default_classical_models(seed=seed, use_gpu=use_gpu)
    results: Dict[str, RunResult] = {}

    if include_classical:
        for model_name, proto in classical.items():
            log(f"--- LOSO [{model_name}] ---", verbose)
            factory = lambda p=proto: default_classical_models(seed=seed, use_gpu=use_gpu)[model_name]  # fresh estimator per fold
            results[model_name] = _classical_loso(
                bundle, target_kind, factory, augment_mode=augment_tabular_mode,
                seed=seed, verbose=verbose,
            )

    if include_deep:
        archs = [deep_arch] if isinstance(deep_arch, str) else list(deep_arch)
        for arch in archs:
            log(f"--- LOSO [{arch}] ---", verbose)
            res = _deep_loso(
                bundle, target_kind,
                augment_mode=augment_sequences_mode,
                cfg=cfg,
                arch=arch,
                mixup_alpha=mixup_alpha,
                fusion=fusion,
                verbose=verbose,
            )
            results[res.name] = res

    if ensemble_members:
        ens = _build_ensemble(results, ensemble_members, target_kind, bundle, verbose=verbose)
        if ens is not None:
            results[ens.name] = ens

    _persist_results(results, bundle, target_kind, out_dir, verbose=verbose)
    return results


def _build_ensemble(
    results: Dict[str, RunResult],
    members: Sequence[str],
    target_kind: str,
    bundle: DatasetBundle,
    verbose: bool,
) -> Optional[RunResult]:
    """Soft-voting ensemble: average per-window probas across selected models.

    Members are matched case-insensitively against existing RunResult.name. We
    align rows across members by (subject_id, session_key, target index) which
    is invariant across runs. Models lacking proba_concat are skipped.
    """
    available = {name.lower(): res for name, res in results.items()}
    chosen: List[RunResult] = []
    for m in members:
        res = available.get(m.lower())
        if res is None:
            log(f"[ensemble] skipping '{m}' (not in results)", verbose)
            continue
        if res.proba_concat is None or res.subject_concat is None:
            log(f"[ensemble] skipping '{res.name}' (no fold-aligned probas)", verbose)
            continue
        chosen.append(res)
    if len(chosen) < 2:
        log(f"[ensemble] need >=2 valid members, got {len(chosen)}; skipping", verbose)
        return None

    # Use the first chosen result as the canonical row order; verify others agree.
    ref = chosen[0]
    n = len(ref.y_true_concat)
    n_classes = ref.proba_concat.shape[1]
    proba_sum = np.zeros((n, n_classes), dtype=np.float64)
    used_names: List[str] = []
    for res in chosen:
        if res.proba_concat.shape != (n, n_classes):
            log(f"[ensemble] shape mismatch on '{res.name}' ({res.proba_concat.shape} vs {(n, n_classes)}); skipping", verbose)
            continue
        same_subj = np.array_equal(res.subject_concat, ref.subject_concat)
        same_y = np.array_equal(res.y_true_concat, ref.y_true_concat)
        if not (same_subj and same_y):
            log(f"[ensemble] row-order mismatch on '{res.name}'; skipping", verbose)
            continue
        # Temperature calibration: search T in [0.5, 4] for the lowest NLL on
        # each member's own out-of-fold predictions before pooling. Wraps
        # over-confident classifiers (e.g. RF/XGB) toward better-calibrated
        # probabilities so soft-voting isn't dominated by their hard votes.
        p = np.clip(res.proba_concat.astype(np.float64), 1e-6, 1.0)
        logp = np.log(p)
        y_idx = res.y_true_concat.astype(int)
        best_T = 1.0
        best_nll = float("inf")
        for T in (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
            scaled = logp / T
            scaled -= scaled.max(axis=1, keepdims=True)
            ex = np.exp(scaled)
            q = ex / ex.sum(axis=1, keepdims=True)
            nll = -np.log(np.clip(q[np.arange(n), y_idx], 1e-12, 1.0)).mean()
            if nll < best_nll:
                best_nll = nll
                best_T = T
        scaled = logp / best_T
        scaled -= scaled.max(axis=1, keepdims=True)
        ex = np.exp(scaled)
        q = ex / ex.sum(axis=1, keepdims=True)
        proba_sum += q
        used_names.append(res.name)
        log(f"[ensemble] '{res.name}' calibrated T={best_T:.2f} (NLL={best_nll:.3f})", verbose)
    if len(used_names) < 2:
        return None
    proba_avg = (proba_sum / len(used_names)).astype(np.float32)
    preds = proba_avg.argmax(axis=1)

    labels = ref.labels
    overall = compute_classification_metrics(ref.y_true_concat, preds, labels)

    # Per-subject (per-fold) metrics for compatibility with build_summary_frame.
    per_fold: List[ClassificationMetrics] = []
    subjects: List[str] = []
    for sid in pd.unique(ref.subject_concat):
        mask = ref.subject_concat == sid
        if not mask.any():
            continue
        m = compute_classification_metrics(ref.y_true_concat[mask], preds[mask], labels)
        per_fold.append(m)
        subjects.append(str(sid))
    agg = aggregate_fold_metrics(per_fold)

    session_overall = None
    session_aggregate = None
    if ref.session_concat is not None:
        y_t_s, y_p_s, _ = pool_session_predictions(ref.y_true_concat, proba_avg, ref.session_concat)
        session_overall = compute_classification_metrics(y_t_s, y_p_s, labels)
        session_aggregate = {
            "session_accuracy": session_overall.accuracy,
            "session_balanced_accuracy": session_overall.balanced_accuracy,
            "session_macro_f1": session_overall.macro_f1,
            "session_cohen_kappa": session_overall.cohen_kappa,
            "n_sessions": int(len(y_t_s)),
        }

    members_label = "+".join(used_names)
    name = "Ensemble"
    log(f"[ensemble] built 'Ensemble' [{members_label}] over {len(used_names)} members  acc={overall.accuracy:.3f}  f1={overall.macro_f1:.3f}", verbose)
    return RunResult(
        name=name,
        labels=labels,
        per_fold_metrics=per_fold,
        subjects=subjects,
        overall=overall,
        aggregate=agg,
        extra={"model_type": "ensemble", "members": used_names},
        session_overall=session_overall,
        session_aggregate=session_aggregate,
        y_true_concat=ref.y_true_concat,
        proba_concat=proba_avg,
        session_concat=ref.session_concat,
        subject_concat=ref.subject_concat,
    )


def _persist_results(
    results: Dict[str, RunResult],
    bundle: DatasetBundle,
    target_kind: str,
    out_dir: str,
    verbose: bool,
) -> None:
    ensure_dir(out_dir)
    labels = _label_list(target_kind, bundle)
    summary: Dict[str, Dict[str, float]] = {}
    for name, result in results.items():
        model_dir = ensure_dir(os.path.join(out_dir, name))
        save_metrics(os.path.join(model_dir, "metrics_overall.json"), result.overall, extra=result.extra)
        with open(os.path.join(model_dir, "metrics_aggregate.json"), "w", encoding="utf-8") as fh:
            import json
            json.dump(result.aggregate, fh, indent=2)

        if result.session_overall is not None:
            save_metrics(
                os.path.join(model_dir, "metrics_session.json"),
                result.session_overall,
                extra=result.session_aggregate,
            )

        fold_df = build_summary_frame(result.per_fold_metrics, result.subjects)
        fold_df.to_csv(os.path.join(model_dir, "per_subject.csv"), index=False)

        cm = np.array(result.overall.confusion)
        plot_confusion(
            cm, labels,
            title=f"{bundle.name} / {target_kind} / {name}",
            out_path=os.path.join(model_dir, "confusion_matrix.png"),
        )
        if result.session_overall is not None:
            cm_s = np.array(result.session_overall.confusion)
            plot_confusion(
                cm_s, labels,
                title=f"{bundle.name} / {target_kind} / {name} (session-pooled)",
                out_path=os.path.join(model_dir, "confusion_matrix_session.png"),
            )
        plot_subject_f1(
            fold_df,
            title=f"{bundle.name} / {target_kind} / {name}: per-subject macro F1",
            out_path=os.path.join(model_dir, "per_subject_f1.png"),
        )
        row: Dict[str, float] = dict(result.aggregate)
        if result.session_aggregate is not None:
            row.update(result.session_aggregate)
        summary[name] = row

    # Cross-model comparison plot & table.
    summary_df = pd.DataFrame(summary).T
    summary_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(summary_path):
        try:
            existing = pd.read_csv(summary_path, index_col=0)
            # New rows take precedence over re-run model names.
            merged = pd.concat([existing.drop(index=summary_df.index, errors="ignore"), summary_df], axis=0)
            summary_df = merged
        except Exception:  # noqa: BLE001
            pass
    summary_df.to_csv(summary_path)
    plot_model_comparison(
        {name: row for name, row in summary_df.to_dict(orient="index").items()},
        title=f"{bundle.name} / {target_kind}: LOSO comparison",
        out_path=os.path.join(out_dir, "model_comparison.png"),
    )
    if verbose:
        log(f"Saved results to {out_dir}", True)


__all__ = ["run_experiment", "RunResult"]
