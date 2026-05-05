"""Advanced deep models: TS-TCC self-supervised pretraining and DANN.

Two new architectures are added here, both motivated by the WESAD LOSO
benchmark where the supervised deep models (Conformer, TinyTCN, BiLSTM,
CNN1D) underperform classical tree models that learn from hand-crafted
windowed features.

* **TS-TCC** (Eldele et al., 2021) is a self-supervised representation
  learning method for time series that combines *temporal contrasting*
  (predict the future of one augmented view from the past context of the
  other view) with *contextual contrasting* (NT-Xent over per-window
  context vectors). The encoder is pretrained per fold on the training
  subjects' raw windows without labels, then frozen and probed with a
  small MLP for emotion classification. This addresses the very small
  labelled training set per fold (~1.3k windows).

* **DANN** (Ganin & Lempitsky, 2015) attaches a subject-discriminator
  head fed through a Gradient Reversal Layer to the Conformer encoder.
  The encoder is pushed to produce features that the discriminator
  cannot use to identify the source subject, which directly fights the
  subject distribution shift that hurts LOSO generalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .conformer import Conformer, ConformerConfig


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin & Lempitsky 2015)
# ---------------------------------------------------------------------------
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """Module wrapper around :func:`grad_reverse` with a mutable ``lambda_``."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_)


# ---------------------------------------------------------------------------
# DANN-Conformer
# ---------------------------------------------------------------------------
@dataclass
class DANNConformerConfig:
    in_channels: int
    num_classes: int
    num_subjects: int
    d_model: int = 128
    num_blocks: int = 4
    heads: int = 4
    conv_kernel: int = 15
    dropout: float = 0.15
    seq_stride: int = 2
    max_len: int = 512


class DANNConformer(nn.Module):
    """Conformer encoder with class head + adversarial subject head."""

    def __init__(self, cfg: DANNConformerConfig):
        super().__init__()
        self.cfg = cfg
        enc_cfg = ConformerConfig(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            d_model=cfg.d_model,
            num_blocks=cfg.num_blocks,
            heads=cfg.heads,
            conv_kernel=cfg.conv_kernel,
            dropout=cfg.dropout,
            seq_stride=cfg.seq_stride,
            max_len=cfg.max_len,
        )
        self.encoder = Conformer(enc_cfg)
        # Replace the classification head with our own to keep symmetry.
        self.encoder.head = nn.Identity()
        self.class_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.subject_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_subjects),
        )

    def set_lambda(self, lambda_: float) -> None:
        self.grl.lambda_ = float(lambda_)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(x)

    def forward(self, x: torch.Tensor):
        feats = self.features(x)
        class_logits = self.class_head(feats)
        subj_logits = self.subject_head(self.grl(feats))
        return class_logits, subj_logits


# ---------------------------------------------------------------------------
# TS-TCC encoder + probe
# ---------------------------------------------------------------------------
@dataclass
class TSTCCConfig:
    in_channels: int
    num_classes: int = 2  # only used by the probe; SSL pretraining ignores it
    feature_dim: int = 128
    proj_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.2


class _TSTCCConvEncoder(nn.Module):
    """3-block 1-D conv backbone (kernels 8/8/8, stride 1, MaxPool/2 after blocks)."""

    def __init__(self, in_channels: int, feature_dim: int, dropout: float):
        super().__init__()
        c1, c2 = 32, 64
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(c2, feature_dim, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L) -> (B, F, L')
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        return h


class TSTCCEncoder(nn.Module):
    """TS-TCC encoder: conv backbone + Transformer context summariser.

    Forward returns ``(features, context)`` where ``features`` has shape
    ``(B, F, T)`` (per-timestep representations) and ``context`` has shape
    ``(B, F)`` (global pooled context vector).
    """

    def __init__(self, cfg: TSTCCConfig):
        super().__init__()
        self.cfg = cfg
        self.conv = _TSTCCConvEncoder(cfg.in_channels, cfg.feature_dim, cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.feature_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.feature_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.feature_dim),
            nn.BatchNorm1d(cfg.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.feature_dim, cfg.proj_dim),
        )

    def features_from_input(self, x_bcl_or_blc: torch.Tensor) -> torch.Tensor:
        """Accepts either ``(B, C, L)`` or ``(B, L, C)`` and returns ``(B, F, T)``."""
        if x_bcl_or_blc.dim() != 3:
            raise ValueError(f"expected 3-D tensor, got shape {tuple(x_bcl_or_blc.shape)}")
        # Heuristic: input channels equal cfg.in_channels.
        if x_bcl_or_blc.shape[1] == self.cfg.in_channels:
            x = x_bcl_or_blc
        elif x_bcl_or_blc.shape[2] == self.cfg.in_channels:
            x = x_bcl_or_blc.transpose(1, 2)
        else:
            raise ValueError(
                f"cannot infer layout for shape {tuple(x_bcl_or_blc.shape)} with in_channels={self.cfg.in_channels}"
            )
        return self.conv(x)

    def forward(self, x):
        feats = self.features_from_input(x)              # (B, F, T)
        ctx_in = feats.transpose(1, 2)                   # (B, T, F)
        ctx_seq = self.transformer(ctx_in)               # (B, T, F)
        context = ctx_seq.mean(dim=1)                    # (B, F)
        return feats, context

    def project(self, context: torch.Tensor) -> torch.Tensor:
        return self.proj(context)


class MLPProbe(nn.Module):
    """Small MLP probe that classifies frozen TS-TCC context vectors."""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# TS-TCC augmentations and losses
# ---------------------------------------------------------------------------
def tstcc_weak_aug(x: torch.Tensor, jitter_std: float = 0.05, scale_std: float = 0.1) -> torch.Tensor:
    """Weak augmentation: Gaussian jitter + per-channel scaling. Input ``(B, C, L)``."""
    if x.numel() == 0:
        return x
    noise = torch.randn_like(x) * jitter_std
    scale = 1.0 + torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * scale_std
    return (x + noise) * scale


def tstcc_strong_aug(x: torch.Tensor, n_segments: int = 8, jitter_std: float = 0.1) -> torch.Tensor:
    """Strong augmentation: random segment permutation along time + jitter. ``(B, C, L)``."""
    if x.numel() == 0:
        return x
    B, C, L = x.shape
    n_segments = max(2, min(n_segments, L))
    seg_len = L // n_segments
    if seg_len < 1:
        return x + torch.randn_like(x) * jitter_std
    out = torch.empty_like(x)
    for b in range(B):
        order = torch.randperm(n_segments, device=x.device)
        chunks = [
            x[b, :, i * seg_len : (i + 1) * seg_len if i < n_segments - 1 else L]
            for i in range(n_segments)
        ]
        permuted = [chunks[i] for i in order.tolist()]
        out[b] = torch.cat(permuted, dim=-1)[:, :L]
    return out + torch.randn_like(out) * jitter_std


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric NT-Xent contrastive loss for paired embeddings ``z1, z2`` of shape ``(B, D)``."""
    B = z1.size(0)
    if B < 2:
        return z1.new_zeros(())
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    z = torch.cat([z1n, z2n], dim=0)                              # (2B, D)
    sim = z @ z.t() / temperature                                 # (2B, 2B)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def temporal_contrast_loss(
    feats_weak: torch.Tensor, feats_strong: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """Cross-view per-timestep InfoNCE: every timestep in view A is contrasted against
    all timesteps in the batch from view B (positive = same window, same timestep).

    ``feats_*`` shape: ``(B, F, T)``. Operates on a fixed ``T`` shared by both views.
    """
    B, _, T = feats_weak.shape
    if B < 2 or T < 1:
        return feats_weak.new_zeros(())
    a = F.normalize(feats_weak.transpose(1, 2).reshape(B * T, -1), dim=-1)    # (B*T, F)
    b = F.normalize(feats_strong.transpose(1, 2).reshape(B * T, -1), dim=-1)
    logits = a @ b.t() / temperature                                          # (B*T, B*T)
    targets = torch.arange(B * T, device=feats_weak.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


__all__ = [
    "GradientReversalLayer",
    "grad_reverse",
    "DANNConformer",
    "DANNConformerConfig",
    "TSTCCEncoder",
    "TSTCCConfig",
    "MLPProbe",
    "tstcc_weak_aug",
    "tstcc_strong_aug",
    "nt_xent_loss",
    "temporal_contrast_loss",
]
