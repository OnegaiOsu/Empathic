"""Multi-stream encoder for heterogeneous physiological + EEG inputs.

EmoWork V2 mixes signal types whose physical bandwidths and timescales differ
by orders of magnitude (EDA ~0.1 Hz events, ECG QRS ~10 ms, EEG gamma 30-45 Hz).
Forcing them through a single shared 1-D backbone wastes capacity: the kernels
that resolve EEG do nothing for tonic EDA, and vice-versa.

This module trains a small dedicated encoder per modality group and fuses the
resulting embeddings before the classifier head:

* **EEG branch** uses an EEGNet-style stack (depthwise + separable convolutions)
  which is the published gold standard for short, low-density EEG windows.
* **Cardiac, autonomic, and motion branches** use compact residual 1-D convs.

Channel groups are inferred from the standard EmoWork SEQ_CHANNELS order
defined in :mod:`empathic.data.emowork`. If a different channel layout is fed
in (e.g. WESAD), the model gracefully degrades to a single shared encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# Default channel groups for EmoWork's 12-channel sequence layout.
# (cardiac, autonomic, motion, eeg). Indices match SEQ_CHANNELS in
# empathic.data.emowork.
EMOWORK_GROUPS: Dict[str, Tuple[int, ...]] = {
    "cardiac": (0, 1, 7),     # ECG, BVP, HR
    "autonomic": (2, 3),      # EDA, TEMP
    "motion": (4, 5, 6),      # ACC_x/y/z
    "eeg": (8, 9, 10, 11),    # EEG_TP9, AF7, AF8, TP10
}


# ---------------------------------------------------------------------------
# Branch building blocks
# ---------------------------------------------------------------------------
class _ResConv1D(nn.Module):
    """Two-layer residual 1-D conv block with optional stride-2 downsample."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, stride=stride)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(h + self.proj(x))


class _ConvBranch(nn.Module):
    """Generic compact ResNet1D branch for cardiac / autonomic / motion."""

    def __init__(self, in_channels: int, channels: int, num_blocks: int,
                 kernel: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        c = in_channels
        for i in range(num_blocks):
            stride = 2 if (0 < i < num_blocks - 1) else 1
            layers.append(_ResConv1D(c, channels, kernel, stride, dropout))
            c = channels
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(channels)
        self.embed_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, Cin) -> (B, D)
        h = x.transpose(1, 2)
        h = self.body(h)
        h = self.pool(h).squeeze(-1)
        return self.norm(h)


class _EEGNetBranch(nn.Module):
    """EEGNet-inspired branch (Lawhern et al., 2018).

    Layout (channels-first):
      Conv1d  (1 -> F1, kernel=64, padding=same)         # temporal filters
      DepthwiseConv1d  groups=F1, depth_mult=D            # per-temporal spatial
      AvgPool1d(4)
      SeparableConv1d  (F1*D -> F2)                       # feature mixing
      AvgPool1d(8) -> flatten -> LayerNorm

    We treat the *channel dimension* as a spatial axis and use grouped 1-D
    conv to mix electrodes -- equivalent to the original 2-D EEGNet design
    over (channels x time).
    """

    def __init__(self, num_eeg_channels: int, F1: int = 16, D: int = 2,
                 F2: int = 32, kernel_temporal: int = 64, dropout: float = 0.25):
        super().__init__()
        # Temporal filter bank, applied independently to each electrode via
        # grouped conv with groups=num_eeg_channels.
        self.temporal = nn.Conv1d(
            num_eeg_channels, num_eeg_channels * F1,
            kernel_size=kernel_temporal,
            padding=kernel_temporal // 2,
            groups=num_eeg_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(num_eeg_channels * F1)
        # Spatial filter: 1x1 mix across (channels * F1) -> F1*D.
        self.spatial = nn.Conv1d(num_eeg_channels * F1, F1 * D, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(F1 * D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool1d(4)
        self.drop1 = nn.Dropout(dropout)

        # Separable conv = depthwise (groups=F1*D) + pointwise (1x1).
        self.dw = nn.Conv1d(F1 * D, F1 * D, kernel_size=16, padding=8,
                            groups=F1 * D, bias=False)
        self.pw = nn.Conv1d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)
        self.pool2 = nn.AvgPool1d(8)
        self.drop2 = nn.Dropout(dropout)

        self.norm_out = nn.LayerNorm(F2)
        self.embed_dim = F2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, Cin) -> (B, D)
        h = x.transpose(1, 2)
        h = self.bn1(self.temporal(h))
        h = self.act(self.bn2(self.spatial(h)))
        h = self.drop1(self.pool1(h))
        h = self.dw(h)
        h = self.act(self.bn3(self.pw(h)))
        h = self.drop2(self.pool2(h))
        h = h.mean(dim=-1)
        return self.norm_out(h)


# ---------------------------------------------------------------------------
# Multi-stream model
# ---------------------------------------------------------------------------
@dataclass
class MultiStreamConfig:
    in_channels: int
    num_classes: int
    groups: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: dict(EMOWORK_GROUPS))
    branch_channels: int = 64
    branch_blocks: int = 4
    branch_kernel: int = 7
    branch_dropout: float = 0.2
    eeg_F1: int = 16
    eeg_D: int = 2
    eeg_F2: int = 32
    eeg_kernel: int = 64
    eeg_dropout: float = 0.25
    head_dropout: float = 0.3


class MultiStream(nn.Module):
    """Per-modality encoder with late fusion.

    If the input layout doesn't match the configured groups (e.g. fewer
    channels than expected), the unmatched indices are routed to a generic
    "fallback" branch so the model degrades gracefully across datasets.
    """

    def __init__(self, cfg: MultiStreamConfig):
        super().__init__()
        self.cfg = cfg
        used: List[int] = []
        branches: Dict[str, nn.Module] = {}
        active_groups: Dict[str, Tuple[int, ...]] = {}
        for name, idxs in cfg.groups.items():
            valid = tuple(i for i in idxs if 0 <= i < cfg.in_channels)
            if not valid:
                continue
            active_groups[name] = valid
            used.extend(valid)
            if name == "eeg":
                branches[name] = _EEGNetBranch(
                    num_eeg_channels=len(valid),
                    F1=cfg.eeg_F1, D=cfg.eeg_D, F2=cfg.eeg_F2,
                    kernel_temporal=cfg.eeg_kernel,
                    dropout=cfg.eeg_dropout,
                )
            else:
                branches[name] = _ConvBranch(
                    in_channels=len(valid),
                    channels=cfg.branch_channels,
                    num_blocks=cfg.branch_blocks,
                    kernel=cfg.branch_kernel,
                    dropout=cfg.branch_dropout,
                )

        # Any channel not assigned to a configured group goes through a
        # shared fallback branch -- e.g. WESAD where the layout differs.
        leftover = tuple(sorted(set(range(cfg.in_channels)) - set(used)))
        if leftover:
            active_groups["other"] = leftover
            branches["other"] = _ConvBranch(
                in_channels=len(leftover),
                channels=cfg.branch_channels,
                num_blocks=cfg.branch_blocks,
                kernel=cfg.branch_kernel,
                dropout=cfg.branch_dropout,
            )

        self.branches = nn.ModuleDict(branches)
        self.active_groups = active_groups
        # Register integer index lists as buffers so .to(device) follows the
        # parent module without us tracking indices manually.
        for name, idxs in active_groups.items():
            self.register_buffer(
                f"_idx_{name}",
                torch.tensor(idxs, dtype=torch.long),
                persistent=False,
            )

        total_emb = sum(int(b.embed_dim) for b in self.branches.values())
        self.embed_dim = total_emb
        self.head = nn.Sequential(
            nn.LayerNorm(total_emb),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(total_emb, cfg.num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, D)
        feats: List[torch.Tensor] = []
        for name in self.active_groups:
            idx: torch.Tensor = getattr(self, f"_idx_{name}")
            sub = x.index_select(dim=-1, index=idx)
            feats.append(self.branches[name](sub))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


__all__ = ["MultiStream", "MultiStreamConfig", "EMOWORK_GROUPS"]
