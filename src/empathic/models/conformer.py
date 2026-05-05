"""Cutting-edge deep model: 1-D Conformer for emotional state recognition.

The Conformer (Gulati et al., "Conformer: Convolution-augmented Transformer for
Speech Recognition", Interspeech 2020) alternates a Transformer self-attention
branch with a depthwise convolution branch inside every block, so it captures
both long-range dependencies (important for slow physiological drifts) and
local kernels (important for keystroke bursts or ECG QRS complexes). It has
since been adopted widely for biosignal and time-series tasks and is a strong
"cutting edge" choice for this project.

Our implementation is a compact, self-contained PyTorch module. It accepts an
input tensor shaped ``(B, L, C)`` and returns class logits; the training
harness then plugs it into a standard cross-entropy + AdamW loop with cosine
learning-rate decay and early stopping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Core blocks
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, d * expansion),
            nn.SiLU(),  # Swish, as in the Conformer paper.
            nn.Dropout(dropout),
            nn.Linear(d * expansion, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        return self.drop(y)


class ConvModule(nn.Module):
    """Depthwise-separable convolution branch used inside Conformer blocks."""

    def __init__(self, d: int, kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        padding = (kernel - 1) // 2
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, kernel_size=1)        # -> GLU halves channels
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d, d, kernel_size=kernel, padding=padding, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d, d, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        h = self.norm(x).transpose(1, 2)                  # (B, D, L)
        h = self.pw1(h)
        h = self.glu(h)
        h = self.dw(h)
        h = self.bn(h)
        h = self.act(h)
        h = self.pw2(h)
        h = h.transpose(1, 2)
        return self.drop(h)


class ConformerBlock(nn.Module):
    """Half-step FFN -> Self-attention -> Conv -> Half-step FFN -> LayerNorm."""

    def __init__(self, d: int, heads: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ffn1 = FeedForward(d, dropout=dropout)
        self.attn = MultiHeadSelfAttention(d, heads=heads, dropout=dropout)
        self.conv = ConvModule(d, kernel=conv_kernel, dropout=dropout)
        self.ffn2 = FeedForward(d, dropout=dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
@dataclass
class ConformerConfig:
    in_channels: int
    num_classes: int
    d_model: int = 128
    num_blocks: int = 4
    heads: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    seq_stride: int = 2      # 1-D conv subsampling factor at the input
    max_len: int = 4096


class Conformer(nn.Module):
    """1-D Conformer classifier for tabular-time-series emotion recognition."""

    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        self.cfg = cfg
        # Subsample long windows (especially WESAD at 240 tokens) before the
        # attention stack to keep compute manageable.
        self.input_proj = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.d_model, kernel_size=3, stride=cfg.seq_stride, padding=1),
            nn.SiLU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.pe = SinusoidalPE(cfg.d_model, max_len=cfg.max_len)
        self.blocks = nn.ModuleList([
            ConformerBlock(cfg.d_model, cfg.heads, cfg.conv_kernel, cfg.dropout)
            for _ in range(cfg.num_blocks)
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.embed_dim = cfg.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, D)
        h = x.transpose(1, 2)                     # (B, C, L)
        h = self.input_proj(h)                    # (B, D, L')
        h = h.transpose(1, 2)                     # (B, L', D)
        h = self.pe(h)
        h = self.dropout(h)
        for blk in self.blocks:
            h = blk(h)
        return h.mean(dim=1)                      # temporal average pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, K)
        return self.head(self.forward_features(x))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Tiny TCN alternative deep model
# ---------------------------------------------------------------------------
@dataclass
class TinyTCNConfig:
    """Config for :class:`TinyTCN`.

    A compact Temporal Convolutional Network (Bai et al. 2018) is a better
    match than a 1.6M-parameter Conformer when only ~2k training windows are
    available per fold. Dilated causal convolutions give a receptive field
    that covers the full window with far fewer parameters, and heavy dropout
    keeps the small training set from being memorised.
    """
    in_channels: int
    num_classes: int
    channels: int = 64
    num_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.3


class _TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=0, dilation=dilation)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(c_out)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = F.pad(x, (self.pad, 0))              # causal left-padding
        h = self.act(self.bn1(self.conv1(h)))
        h = self.drop(h)
        h = F.pad(h, (self.pad, 0))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.drop(h)
        return h + self.proj(x)


class TinyTCN(nn.Module):
    """Small dilated-TCN classifier for short emotion windows."""

    def __init__(self, cfg: TinyTCNConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        c_in = cfg.in_channels
        for i in range(cfg.num_blocks):
            layers.append(_TCNBlock(
                c_in=c_in,
                c_out=cfg.channels,
                kernel=cfg.kernel_size,
                dilation=2 ** i,
                dropout=cfg.dropout,
            ))
            c_in = cfg.channels
        self.tcn = nn.Sequential(*layers)
        self.embed_dim = cfg.channels
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(cfg.channels),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.channels, cfg.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, D)
        h = x.transpose(1, 2)                    # (B, C, L)
        h = self.tcn(h)
        return self.feature_extractor(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, K)
        return self.head(self.forward_features(x))


# ---------------------------------------------------------------------------
# BiLSTM with attention pooling
# ---------------------------------------------------------------------------
@dataclass
class BiLSTMConfig:
    in_channels: int
    num_classes: int
    hidden: int = 96
    num_layers: int = 2
    dropout: float = 0.3


class BiLSTM(nn.Module):
    """2-layer bidirectional LSTM with attention pooling.

    A canonical recurrent baseline for biosignal time series; useful as a
    contrast point against the convolutional and attention-only models.
    """

    def __init__(self, cfg: BiLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.in_channels,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        d = 2 * cfg.hidden
        self.embed_dim = d
        self.attn = nn.Linear(d, 1)
        self.feature_norm = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(d, cfg.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, D)
        h, _ = self.lstm(x)                              # (B, L, 2H)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # (B, L)
        pooled = torch.einsum("blh,bl->bh", h, w)        # (B, 2H)
        return self.feature_norm(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, K)
        return self.head(self.forward_features(x))


# ---------------------------------------------------------------------------
# 1-D ResNet-style CNN
# ---------------------------------------------------------------------------
@dataclass
class CNN1DConfig:
    in_channels: int
    num_classes: int
    channels: int = 64
    num_blocks: int = 4
    kernel_size: int = 7
    dropout: float = 0.2


class _ResBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, stride=stride)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, stride=1)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride), nn.BatchNorm1d(c_out))
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(h + self.proj(x))


class CNN1D(nn.Module):
    """Small 1-D ResNet-style classifier for fixed-length sequence windows."""

    def __init__(self, cfg: CNN1DConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        c_in = cfg.in_channels
        for i in range(cfg.num_blocks):
            stride = 2 if (i > 0 and i < cfg.num_blocks - 1) else 1
            layers.append(_ResBlock1D(c_in, cfg.channels, cfg.kernel_size, stride, cfg.dropout))
            c_in = cfg.channels
        self.cnn = nn.Sequential(*layers)
        self.embed_dim = cfg.channels
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(cfg.channels),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.channels, cfg.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, D)
        h = x.transpose(1, 2)                            # (B, C, L)
        h = self.cnn(h)
        return self.feature_extractor(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, K)
        return self.head(self.forward_features(x))
