import math
from typing import Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE1D(nn.Module):
    """Squeeze-Excitation 1D module for channel attention.
    
    Args:
        channels: Number of channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, t = x.size()
        se = x.mean(dim=2)  # (B, C)
        se = self.fc(se)  # (B, C)
        return x * se.view(b, c, 1)


class ECA1D(nn.Module):
    """Efficient Channel Attention 1D module - lightweight without FC layers.
    
    Args:
        channels: Number of channels
        kernel_size: Kernel size for adaptive pool (default: 3)
    """
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = x.mean(dim=2, keepdim=True)  # (B, C, 1) -> (B, 1, 1) after mean? No, use max then mean
        y = torch.mean(x, dim=1, keepdim=True)  # (B, 1, T) - temporal attention
        y = self.conv(y)  # (B, 1, T)
        y = self.sigmoid(y)  # (B, 1, T)
        return x * y


class AddNorm(nn.Module):
    """Add & Norm: residual connection with layer normalization for stability."""
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), residual: (B, C, T)
        x = x + residual
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.ln(x)
        x = x.transpose(1, 2)  # (B, C, T)
        return x


class TemporalSelfAttention1D(nn.Module):
    """Temporal self-attention over the time axis with residual + LayerNorm."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads for temporal attention.")
        self.mha = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_t = x.transpose(1, 2)  # (B, T, C)
        attn_out, _ = self.mha(x_t, x_t, x_t, need_weights=False)
        x_t = self.ln(attn_out + x_t)
        return x_t.transpose(1, 2)


TFC_DEFAULT_CHANNELS = (32, 64, 128)


class SharedQKAttention1D(nn.Module):
    """Self-attention with shared Q/K projections and domain-specific V/Out projections."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        domains: Sequence[str] = ("time", "freq", "tf"),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads for shared Q/K attention.")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout
        self.q_proj = nn.Linear(channels, channels, bias=True)
        self.k_proj = nn.Linear(channels, channels, bias=True)
        self.v_proj = nn.ModuleDict({name: nn.Linear(channels, channels, bias=True) for name in domains})
        self.out_proj = nn.ModuleDict({name: nn.Linear(channels, channels, bias=True) for name in domains})
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        if domain not in self.v_proj:
            raise ValueError(f"Unknown domain for shared attention: {domain}")
        # x: (B, C, T)
        x_t = x.transpose(1, 2)  # (B, T, C)
        q = self.q_proj(x_t)
        k = self.k_proj(x_t)
        v = self.v_proj[domain](x_t)

        b, t, c = q.shape
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = torch.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out_proj[domain](out)
        out = self.ln(out + x_t)
        return out.transpose(1, 2)


def build_shared_qk_attn(
    backbone: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    tfc_channels: Sequence[int] = TFC_DEFAULT_CHANNELS,
) -> Optional[object]:
    tfc_out = tfc_channels[-1] if tfc_channels else hidden_dim
    if backbone == "tfc_resnet":
        return SharedQKAttention1D(tfc_out, num_heads=num_heads, dropout=dropout)
    if backbone == "all":
        return {
            "inception": SharedQKAttention1D(hidden_dim, num_heads=num_heads, dropout=dropout),
            "resnet": SharedQKAttention1D(hidden_dim, num_heads=num_heads, dropout=dropout),
            "inception_resattn": SharedQKAttention1D(hidden_dim, num_heads=num_heads, dropout=dropout),
            "tfc_resnet": SharedQKAttention1D(tfc_out, num_heads=num_heads, dropout=dropout),
        }
    return SharedQKAttention1D(hidden_dim, num_heads=num_heads, dropout=dropout)


class ProjectionSkip(nn.Module):
    """Projection skip connection for shape mismatches (stride, pooling, etc)."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if in_channels == out_channels and stride == 1:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ProjectorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformPredictor(nn.Module):
    """Predict transform parameters from an embedding vector."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int = 32,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4 for InceptionBlock1D.")
        branch_channels = out_channels // 4
        bottleneck = min(in_channels, bottleneck_channels)
        self.use_bottleneck = in_channels > 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1) if self.use_bottleneck else nn.Identity()

        self.conv_1 = nn.Conv1d(bottleneck if self.use_bottleneck else in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(bottleneck if self.use_bottleneck else in_channels, branch_channels, kernel_size=5, padding=2)
        self.conv_3 = nn.Conv1d(bottleneck if self.use_bottleneck else in_channels, branch_channels, kernel_size=7, padding=3)
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm1d(branch_channels * 4)
        self.relu = nn.ReLU()
        
        # SE1D module (disabled by default)
        self.use_se = use_se
        if use_se:
            self.se = SE1D(out_channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.bottleneck(x)
        b1 = self.conv_1(x_in)
        b2 = self.conv_2(x_in)
        b3 = self.conv_3(x_in)
        b4 = self.pool(x)
        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)  # Apply SE1D if enabled
        return x


class InceptionTimeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.block1 = InceptionBlock1D(input_dim, hidden_dim, use_se=use_se, se_reduction=se_reduction)
        self.block2 = InceptionBlock1D(hidden_dim, hidden_dim, use_se=use_se, se_reduction=se_reduction)
        self.block3 = InceptionBlock1D(hidden_dim, hidden_dim, use_se=use_se, se_reduction=se_reduction)
        self.residual = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.relu(x + res)


class ResNetBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
        # SE1D module (disabled by default)
        self.use_se = use_se
        if use_se:
            self.se = SE1D(channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)  # Apply SE1D if enabled
        return self.relu(x + residual)


class InceptionTimeOnlyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_se: bool = False,
        se_reduction: int = 16,
        num_heads: int = 4,
        use_temporal_attn: bool = False,
        shared_qk_attn: Optional[SharedQKAttention1D] = None,
        shared_domain: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.inception = InceptionTimeEncoder(input_dim, hidden_dim, use_se=use_se, se_reduction=se_reduction)
        self.blocks = nn.ModuleList(
            [self.inception.block1, self.inception.block2, self.inception.block3]
        )
        self.temporal_attn = (
            TemporalSelfAttention1D(hidden_dim, num_heads=num_heads) if use_temporal_attn else nn.Identity()
        )
        self.use_temporal_attn = use_temporal_attn
        self.shared_qk_attn = shared_qk_attn
        self.shared_domain = shared_domain
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.inception(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
        else:
            x = self.temporal_attn(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.inception(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
            attn_feat = x
        else:
            x = self.temporal_attn(x)
            attn_feat = x if self.use_temporal_attn else None
        x = self.pool(x).squeeze(-1)
        return self.proj(x), attn_feat


class ResNet1DOnlyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        res_blocks: int = 3,
        use_se: bool = False,
        se_reduction: int = 16,
        num_heads: int = 4,
        use_temporal_attn: bool = False,
        shared_qk_attn: Optional[SharedQKAttention1D] = None,
        shared_domain: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock1D(hidden_dim, use_se=use_se, se_reduction=se_reduction) for _ in range(res_blocks)]
        )
        self.temporal_attn = (
            TemporalSelfAttention1D(hidden_dim, num_heads=num_heads) if use_temporal_attn else nn.Identity()
        )
        self.use_temporal_attn = use_temporal_attn
        self.shared_qk_attn = shared_qk_attn
        self.shared_domain = shared_domain
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
        else:
            x = self.temporal_attn(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
            attn_feat = x
        else:
            x = self.temporal_attn(x)
            attn_feat = x if self.use_temporal_attn else None
        x = self.pool(x).squeeze(-1)
        return self.proj(x), attn_feat


class TFCResNetEncoder(nn.Module):
    """Three-layer 1-D ResNet-style encoder (TF-C Appendix E)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channels=TFC_DEFAULT_CHANNELS,
        kernel_sizes=(8, 8, 8),
        strides=(8, 1, 1),
        pool_kernel: int = 2,
        pool_stride: int = 2,
        num_heads: int = 4,
        use_temporal_attn: bool = False,
        shared_qk_attn: Optional[SharedQKAttention1D] = None,
        shared_domain: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes, and strides must have the same length.")
        self.blocks = nn.ModuleList()
        in_channels = input_dim
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            padding = kernel_size // 2
            block = nn.ModuleDict(
                {
                    "conv": nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    "bn": nn.BatchNorm1d(out_channels),
                    "act": nn.ReLU(),
                }
            )
            self.blocks.append(block)
            in_channels = out_channels
        self.temporal_attn = (
            TemporalSelfAttention1D(in_channels, num_heads=num_heads) if use_temporal_attn else nn.Identity()
        )
        self.use_temporal_attn = use_temporal_attn
        self.shared_qk_attn = shared_qk_attn
        self.shared_domain = shared_domain
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for block in self.blocks:
            x = block["conv"](x)
            x = block["bn"](x)
            x = block["act"](x)
            if x.shape[-1] >= self.pool_kernel and self.pool_stride > 0:
                x = F.max_pool1d(x, kernel_size=self.pool_kernel, stride=self.pool_stride)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
        else:
            x = self.temporal_attn(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for block in self.blocks:
            x = block["conv"](x)
            x = block["bn"](x)
            x = block["act"](x)
            if x.shape[-1] >= self.pool_kernel and self.pool_stride > 0:
                x = F.max_pool1d(x, kernel_size=self.pool_kernel, stride=self.pool_stride)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
            attn_feat = x
        else:
            x = self.temporal_attn(x)
            attn_feat = x if self.use_temporal_attn else None
        x = self.pool(x).squeeze(-1)
        return self.proj(x), attn_feat


class TimesBlock1D(nn.Module):
    """Lightweight period-aware block inspired by TimesNet."""

    def __init__(self, channels: int, top_k: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for TimesBlock1D.")
        self.top_k = max(1, top_k)
        self.conv_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False,
        )
        self.conv_w = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=False,
        )
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(channels)

    def _select_periods(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        b, _, t = x.shape
        if t <= 2:
            one = torch.ones(1, device=x.device, dtype=x.dtype)
            return torch.tensor([max(2, t)], device=x.device, dtype=torch.long), one
        spec = torch.fft.rfft(x.mean(dim=1), dim=-1)  # (B, F)
        amp = spec.abs().mean(dim=0)  # (F,)
        amp[0] = 0.0
        k = min(self.top_k, max(1, int(amp.numel()) - 1))
        vals, idx = torch.topk(amp, k=k)
        periods = []
        for freq_idx in idx.tolist():
            if freq_idx <= 0:
                continue
            period = int(round(float(t) / float(freq_idx)))
            period = max(2, min(t, period))
            periods.append(period)
        if not periods:
            periods = [max(2, t)]
            vals = torch.ones(1, device=x.device, dtype=x.dtype)
        return torch.tensor(periods, device=x.device, dtype=torch.long), vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        periods, raw_w = self._select_periods(x)
        weights = torch.softmax(raw_w.to(dtype=x.dtype), dim=0)
        fused = torch.zeros_like(x)
        for i in range(int(periods.numel())):
            p = int(periods[i].item())
            pad = (p - (t % p)) % p
            x_pad = F.pad(x, (0, pad))
            t_pad = x_pad.shape[-1]
            n = t_pad // p
            x2 = x_pad.reshape(b, c, n, p)
            y = self.conv_h(x2)
            y = self.act(y)
            y = self.conv_w(y)
            y = y.reshape(b, c, t_pad)[..., :t]
            fused = fused + weights[i] * y
        out = fused + x
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class TimesNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        num_heads: int = 4,
        use_temporal_attn: bool = False,
        shared_qk_attn: Optional[SharedQKAttention1D] = None,
        shared_domain: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([TimesBlock1D(hidden_dim) for _ in range(max(1, num_blocks))])
        self.temporal_attn = (
            TemporalSelfAttention1D(hidden_dim, num_heads=num_heads) if use_temporal_attn else nn.Identity()
        )
        self.use_temporal_attn = use_temporal_attn
        self.shared_qk_attn = shared_qk_attn
        self.shared_domain = shared_domain
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
        else:
            x = self.temporal_attn(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
            attn_feat = x
        else:
            x = self.temporal_attn(x)
            attn_feat = x if self.use_temporal_attn else None
        x = self.pool(x).squeeze(-1)
        return self.proj(x), attn_feat


class TSLAFilterBlock1D(nn.Module):
    """Adaptive spectral-gated depthwise block (TSLANet-style)."""

    def __init__(self, channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for TSLAFilterBlock1D.")
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.bn(y)
        time_stat = x.mean(dim=-1)
        spec_stat = torch.fft.rfft(x, dim=-1).abs().mean(dim=-1)
        gate = self.gate(torch.cat([time_stat, spec_stat], dim=-1)).unsqueeze(-1)
        y = y * gate
        return F.gelu(x + y)


class TSLANetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        num_heads: int = 4,
        use_temporal_attn: bool = False,
        shared_qk_attn: Optional[SharedQKAttention1D] = None,
        shared_domain: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([TSLAFilterBlock1D(hidden_dim) for _ in range(max(1, num_blocks))])
        self.temporal_attn = (
            TemporalSelfAttention1D(hidden_dim, num_heads=num_heads) if use_temporal_attn else nn.Identity()
        )
        self.use_temporal_attn = use_temporal_attn
        self.shared_qk_attn = shared_qk_attn
        self.shared_domain = shared_domain
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
        else:
            x = self.temporal_attn(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x = self.shared_qk_attn(x, self.shared_domain)
            attn_feat = x
        else:
            x = self.temporal_attn(x)
            attn_feat = x if self.use_temporal_attn else None
        x = self.pool(x).squeeze(-1)
        return self.proj(x), attn_feat


DEFAULT_ALL_BACKBONES = ("inception", "resnet", "tfc_resnet", "inception_resattn")


class MultiBackboneEncoder(nn.Module):
    """Fuse multiple backbone encoders by concatenating their embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        res_blocks: int = 2,
        backbones: Optional[Sequence[str]] = None,
        fuse_out_dim: Optional[int] = None,
        fuse_dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        attn_type: Literal["none", "mha", "eca"] = "mha",
        use_eca_intra: bool = True,
        use_temporal_attn: bool = False,
        shared_qk_attn: Optional[object] = None,
        shared_domain: Optional[str] = None,
        tf_input: bool = False,
    ) -> None:
        super().__init__()
        self.backbones = list(backbones or DEFAULT_ALL_BACKBONES)
        if not self.backbones:
            raise ValueError("backbones must contain at least one backbone name.")
        if "all" in self.backbones:
            raise ValueError("backbones list must not include 'all' to avoid recursion.")
        self.encoders = nn.ModuleList(
            [
                build_encoder(
                    backbone=backbone,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_heads=num_heads,
                    res_blocks=res_blocks,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    attn_type=attn_type,
                    use_eca_intra=use_eca_intra,
                    use_temporal_attn=use_temporal_attn,
                    shared_qk_attn=shared_qk_attn.get(backbone) if isinstance(shared_qk_attn, dict) else shared_qk_attn,
                    shared_domain=shared_domain,
                    tf_input=tf_input,
                )
                for backbone in self.backbones
            ]
        )
        fuse_out_dim = fuse_out_dim or output_dim
        self.fuse_dropout = nn.Dropout(fuse_dropout) if fuse_dropout > 0.0 else nn.Identity()
        self.fuse = nn.Linear(len(self.encoders) * output_dim, fuse_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = [encoder(x) for encoder in self.encoders]
        fused = torch.cat(embeddings, dim=-1)
        fused = self.fuse_dropout(fused)
        return self.fuse(fused)

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        embeddings = []
        attn_feats = {}
        for backbone, encoder in zip(self.backbones, self.encoders):
            if hasattr(encoder, "forward_with_attn"):
                emb, attn = encoder.forward_with_attn(x)
            else:
                emb = encoder(x)
                attn = None
            embeddings.append(emb)
            attn_feats[backbone] = attn
        fused = torch.cat(embeddings, dim=-1)
        fused = self.fuse_dropout(fused)
        return self.fuse(fused), attn_feats


class InceptionResAttnEncoder(nn.Module):
    """Multi-branch encoder with configurable intra/inter-domain attention.
    
    Args:
        input_dim: Input channels
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_heads: Number of attention heads
        res_blocks: Number of ResNet blocks
        attn_type: Attention type - "none" (no inter-domain attn), "mha" (multi-head attention), 
                   "eca" (efficient channel attention only)
        use_eca_intra: Use ECA for intra-domain channel attention (default: True)
        use_se: Use SE1D in blocks (default: False)
        se_reduction: SE reduction ratio (default: 16)
        tf_input: Whether input is TF representation (applies dim reduction before attn)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        res_blocks: int = 2,
        attn_type: Literal["none", "mha", "eca"] = "mha",
        use_eca_intra: bool = True,
        use_se: bool = False,
        se_reduction: int = 16,
        shared_qk_attn: Optional[SharedQKAttention1D] = None,
        shared_domain: Optional[str] = None,
        tf_input: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for multi-head attention.")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attn_type = attn_type
        self.use_eca_intra = use_eca_intra
        self.tf_input = tf_input
        self.shared_qk_attn = shared_qk_attn
        self.shared_domain = shared_domain
        
        # For TF inputs: projection from C*F to hidden_dim
        if tf_input and input_dim != hidden_dim:
            self.tf_proj = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
            )
        else:
            self.tf_proj = None
        
        # Shared input projection to hidden_dim (for dimension matching)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim if not tf_input else hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Inception encoder (intra-domain branch 1: multi-scale temporal)
        self.inception = InceptionTimeEncoder(hidden_dim, hidden_dim, use_se=use_se, se_reduction=se_reduction)
        
        # ResNet blocks (intra-domain branch 2: residual stacking)
        self.resnet = nn.Sequential(
            *[ResNetBlock1D(hidden_dim, use_se=use_se, se_reduction=se_reduction) for _ in range(res_blocks)]
        )
        
        # Intra-domain channel attention (ECA for lightweight shape-preserving attention)
        if use_eca_intra:
            self.eca_intra = ECA1D(hidden_dim, kernel_size=3)
        else:
            self.eca_intra = nn.Identity()
        
        # Inter-domain attention (temporal self-attention or none)
        if attn_type == "mha":
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
            self.add_norm = AddNorm(hidden_dim)
        elif attn_type == "eca":
            # ECA-only mode: no temporal attention
            self.attn = None
            self.add_norm = None
        else:  # "none"
            self.attn = None
            self.add_norm = None
        
        # Output pooling and projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, is_tf: bool = False) -> torch.Tensor:
        x_combined, _ = self._forward_features(x, is_tf=is_tf)
        x_combined = self.pool(x_combined).squeeze(-1)
        return self.proj(x_combined)

    def forward_with_attn(
        self, x: torch.Tensor, is_tf: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_combined, attn_feat = self._forward_features(x, is_tf=is_tf)
        x_combined = self.pool(x_combined).squeeze(-1)
        return self.proj(x_combined), attn_feat

    def _forward_features(
        self, x: torch.Tensor, is_tf: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # For TF inputs: reduce dimension before processing to avoid computation explosion
        # TF shape: (B, C*F, T) -> (B, hidden_dim, T)
        if (is_tf or self.tf_input) and self.tf_proj is not None:
            x = self.tf_proj(x)
        
        # Project input to hidden_dim
        x = self.input_proj(x)
        
        # Intra-domain: Inception branch (multi-scale temporal features)
        x_inc = self.inception(x)
        
        # Intra-domain: ResNet branch (residual refinement)
        x_res = self.resnet(x)
        
        # Combine branches: element-wise addition
        x_combined = x_inc + x_res
        
        # Intra-domain: lightweight channel attention (ECA)
        x_combined = self.eca_intra(x_combined)
        
        attn_feat = None
        # Inter-domain: temporal self-attention (optional MHA with Add&Norm)
        if self.shared_qk_attn is not None:
            if self.shared_domain is None:
                raise ValueError("shared_domain must be set when shared_qk_attn is provided.")
            x_combined = self.shared_qk_attn(x_combined, self.shared_domain)
            attn_feat = x_combined
        elif self.attn_type == "mha" and self.attn is not None:
            # Transpose for attention: (B, C, T) -> (B, T, C)
            x_t = x_combined.transpose(1, 2)
            
            # Multi-head attention
            attn_out, _ = self.attn(x_t, x_t, x_t, need_weights=False)
            
            # Add & Norm for stability
            x_t = self.add_norm(attn_out.transpose(1, 2), x_combined)
            x_combined = x_t
            attn_feat = x_combined
        
        return x_combined, attn_feat


def build_encoder(
    backbone: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_heads: int = 4,
    res_blocks: int = 2,
    use_se: bool = False,
    se_reduction: int = 16,
    attn_type: Literal["none", "mha", "eca"] = "mha",
    use_eca_intra: bool = True,
    use_temporal_attn: bool = False,
    shared_qk_attn: Optional[SharedQKAttention1D] = None,
    shared_domain: Optional[str] = None,
    tf_input: bool = False,
    fuse_dropout: float = 0.0,
) -> nn.Module:
    if backbone == "inception":
        return InceptionTimeOnlyEncoder(
            input_dim, hidden_dim, output_dim,
            use_se=use_se, se_reduction=se_reduction,
            num_heads=num_heads, use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn, shared_domain=shared_domain
        )
    if backbone == "resnet":
        return ResNet1DOnlyEncoder(
            input_dim, hidden_dim, output_dim, res_blocks=res_blocks,
            use_se=use_se, se_reduction=se_reduction,
            num_heads=num_heads, use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn, shared_domain=shared_domain
        )
    if backbone == "tfc_resnet":
        return TFCResNetEncoder(
            input_dim,
            output_dim,
            num_heads=num_heads,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain=shared_domain,
        )
    if backbone == "timesnet":
        return TimesNetEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_blocks=res_blocks,
            num_heads=num_heads,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain=shared_domain,
        )
    if backbone == "tslanet":
        return TSLANetEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_blocks=res_blocks,
            num_heads=num_heads,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain=shared_domain,
        )
    if backbone == "all":
        return MultiBackboneEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            res_blocks=res_blocks,
            use_se=use_se,
            se_reduction=se_reduction,
            attn_type=attn_type,
            use_eca_intra=use_eca_intra,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain=shared_domain,
            tf_input=tf_input,
            fuse_dropout=fuse_dropout,
        )
    if backbone == "inception_resattn":
        return InceptionResAttnEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            res_blocks=res_blocks,
            attn_type=attn_type,
            use_eca_intra=use_eca_intra,
            use_se=use_se,
            se_reduction=se_reduction,
            shared_qk_attn=shared_qk_attn,
            shared_domain=shared_domain,
            tf_input=tf_input,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


class MultiViewModel(nn.Module):
    def __init__(
        self,
        input_dim_time: int,
        input_dim_freq: int,
        input_dim_tf: Optional[int] = None,
        hidden_dim: int = 64,
        output_dim: int = 128,
        projector_hidden_dim: int = 256,
        projector_out_dim: Optional[int] = None,
        use_projectors: bool = True,
        num_heads: int = 4,
        res_blocks: int = 2,
        backbone: str = "all",
        use_se: bool = False,
        se_reduction: int = 16,
        attn_type: Literal["none", "mha", "eca"] = "mha",
        use_eca_intra: bool = True,
        use_temporal_attn: bool = False,
        use_shared_qk_attn: bool = False,
        shared_qk_heads: int = 4,
        shared_qk_dropout: float = 0.0,
        fuse_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        shared_qk_attn = (
            build_shared_qk_attn(backbone, hidden_dim, shared_qk_heads, shared_qk_dropout)
            if use_shared_qk_attn
            else None
        )
        self.time_encoder = build_encoder(
            backbone=backbone,
            input_dim=input_dim_time,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            res_blocks=res_blocks,
            use_se=use_se,
            se_reduction=se_reduction,
            attn_type=attn_type,
            use_eca_intra=use_eca_intra,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain="time",
            tf_input=False,
            fuse_dropout=fuse_dropout,
        )
        self.freq_encoder = build_encoder(
            backbone=backbone,
            input_dim=input_dim_freq,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            res_blocks=res_blocks,
            use_se=use_se,
            se_reduction=se_reduction,
            attn_type=attn_type,
            use_eca_intra=use_eca_intra,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain="freq",
            tf_input=False,
            fuse_dropout=fuse_dropout,
        )
        if input_dim_tf is None:
            input_dim_tf = input_dim_freq
        self.tf_encoder = build_encoder(
            backbone=backbone,
            input_dim=input_dim_tf,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            res_blocks=res_blocks,
            use_se=use_se,
            se_reduction=se_reduction,
            attn_type=attn_type,
            use_eca_intra=use_eca_intra,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain="tf",
            tf_input=True,  # TF inputs get special handling
            fuse_dropout=fuse_dropout,
        )
        projector_out_dim = projector_out_dim or output_dim
        if use_projectors:
            self.time_projector = ProjectorMLP(output_dim, projector_hidden_dim, projector_out_dim)
            self.freq_projector = ProjectorMLP(output_dim, projector_hidden_dim, projector_out_dim)
            self.tf_projector = ProjectorMLP(output_dim, projector_hidden_dim, projector_out_dim)
        else:
            self.time_projector = nn.Identity()
            self.freq_projector = nn.Identity()
            self.tf_projector = nn.Identity()

    def forward(
        self,
        x_time: Optional[torch.Tensor] = None,
        x_freq: Optional[torch.Tensor] = None,
        x_tf: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_attn: bool = False,
    ):
        if return_intermediates:
            encoded = self._encode_views(
                x_time=x_time,
                x_freq=x_freq,
                x_tf=x_tf,
                with_attn=return_attn,
            )
            out = {
                "h_time": encoded["h_time"],
                "h_freq": encoded["h_freq"],
                "h_tf": encoded["h_tf"],
                "z_time": encoded["z_time"],
                "z_freq": encoded["z_freq"],
                "z_tf": encoded["z_tf"],
                "pred_b": None,
                "pred_rho": None,
                "pred_g": None,
            }
            if return_attn:
                out["attn_time"] = encoded["attn_time"]
                out["attn_freq"] = encoded["attn_freq"]
                out["attn_tf"] = encoded["attn_tf"]
            return out

        h_time = self.time_encoder(x_time) if x_time is not None else None
        h_freq = self.freq_encoder(x_freq) if x_freq is not None else None
        h_tf = self.tf_encoder(x_tf) if x_tf is not None else None
        z_time = self.time_projector(h_time) if h_time is not None else None
        z_freq = self.freq_projector(h_freq) if h_freq is not None else None
        z_tf = self.tf_projector(h_tf) if h_tf is not None else None
        return z_time, z_freq, z_tf

    def forward_with_attn(
        self,
        x_time: Optional[torch.Tensor] = None,
        x_freq: Optional[torch.Tensor] = None,
        x_tf: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        dict,
    ]:
        encoded = self._encode_views(x_time=x_time, x_freq=x_freq, x_tf=x_tf, with_attn=True)
        z_time = encoded["z_time"]
        z_freq = encoded["z_freq"]
        z_tf = encoded["z_tf"]
        attn = {
            "time": encoded["attn_time"],
            "freq": encoded["attn_freq"],
            "tf": encoded["attn_tf"],
        }
        return (z_time, z_freq, z_tf), attn

    def _encode_views(
        self,
        x_time: Optional[torch.Tensor],
        x_freq: Optional[torch.Tensor],
        x_tf: Optional[torch.Tensor],
        with_attn: bool,
    ) -> dict:
        if with_attn:
            h_time, attn_time = self._encode_with_attn(self.time_encoder, x_time)
            h_freq, attn_freq = self._encode_with_attn(self.freq_encoder, x_freq)
            h_tf, attn_tf = self._encode_with_attn(self.tf_encoder, x_tf, is_tf=True)
        else:
            h_time = self.time_encoder(x_time) if x_time is not None else None
            h_freq = self.freq_encoder(x_freq) if x_freq is not None else None
            h_tf = self.tf_encoder(x_tf) if x_tf is not None else None
            attn_time = None
            attn_freq = None
            attn_tf = None
        z_time = self.time_projector(h_time) if h_time is not None else None
        z_freq = self.freq_projector(h_freq) if h_freq is not None else None
        z_tf = self.tf_projector(h_tf) if h_tf is not None else None
        return {
            "h_time": h_time,
            "h_freq": h_freq,
            "h_tf": h_tf,
            "z_time": z_time,
            "z_freq": z_freq,
            "z_tf": z_tf,
            "attn_time": attn_time,
            "attn_freq": attn_freq,
            "attn_tf": attn_tf,
        }

    @staticmethod
    def _encode_with_attn(encoder: nn.Module, x: Optional[torch.Tensor], is_tf: bool = False):
        if x is None:
            return None, None
        if hasattr(encoder, "forward_with_attn"):
            try:
                return encoder.forward_with_attn(x, is_tf=is_tf)
            except TypeError:
                return encoder.forward_with_attn(x)
        return encoder(x), None
