from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Dict, Optional

import torch

try:
    from .transforms import band_shift_stft, frequency_scale_time, spectral_coloring, stft_magnitude
except ImportError:  # pragma: no cover - script mode
    from transforms import band_shift_stft, frequency_scale_time, spectral_coloring, stft_magnitude  # type: ignore

BCT_SHAPE = "(B, C, T)"
_EPS = 1e-6


@dataclass(frozen=True)
class PreprocessConfig:
    n_fft: int = 256
    hop_length: int = 64
    win_length: Optional[int] = None
    window_name: str = "hann"
    center: bool = True
    magnitude_power: float = 1.0
    tf_log1p: bool = True
    tf_flatten: bool = True
    normalize_mode: str = "per_sample_channel"

    def hash(self, digits: int = 10) -> str:
        payload = asdict(self)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:digits]


def ensure_bct_batch(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must follow {BCT_SHAPE}, got shape={tuple(x.shape)}.")


def ensure_ct_sample(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    raise ValueError(f"{name} must follow (C, T), got shape={tuple(x.shape)}.")


def normalize_time_series(
    x: torch.Tensor,
    length: Optional[int] = None,
    mode: str = "per_sample_channel",
) -> torch.Tensor:
    x = ensure_ct_sample(x, "x").clone()
    if length is None:
        length = int(x.shape[-1])
    if length <= 0:
        return x
    if mode == "none":
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if length < x.shape[-1]:
            x[..., length:] = 0.0
        return x
    if mode != "per_sample_channel":
        raise ValueError(f"Unsupported normalize mode: {mode}")
    valid = x[..., :length]
    mask = torch.isfinite(valid)
    if mask.any():
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        masked = torch.where(mask, valid, torch.zeros_like(valid))
        mean = masked.sum(dim=-1, keepdim=True) / count
        var = torch.where(mask, (valid - mean) ** 2, torch.zeros_like(valid)).sum(dim=-1, keepdim=True) / count
        std = torch.sqrt(var + _EPS)
    else:
        mean = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        std = torch.ones_like(mean)
    x = (x - mean) / std
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if length < x.shape[-1]:
        x[..., length:] = 0.0
    return x


def compute_stft_magnitude(x: torch.Tensor, config: PreprocessConfig) -> torch.Tensor:
    x = ensure_ct_sample(x, "x")
    return stft_magnitude(
        x,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window_name=config.window_name,
        center=config.center,
        magnitude_power=config.magnitude_power,
    )


def freq_view_from_mag(stft: torch.Tensor) -> torch.Tensor:
    if stft.dim() == 2:
        return stft.mean(dim=-1, keepdim=False).unsqueeze(0)
    if stft.dim() == 3:
        return stft.mean(dim=-1)
    if stft.dim() == 4:
        return stft.mean(dim=-1)
    raise ValueError("Expected STFT magnitude with 2-4 dims.")


def tf_view_from_mag(stft: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    if stft.dim() == 2:
        return stft
    if stft.dim() == 3:
        if not flatten:
            return stft
        channels, freqs, frames = stft.shape
        return stft.reshape(channels * freqs, frames)
    if stft.dim() == 4:
        if not flatten:
            return stft
        batch, channels, freqs, frames = stft.shape
        return stft.reshape(batch, channels * freqs, frames)
    raise ValueError("Expected STFT magnitude with 2-4 dims.")


def build_triview_from_time(x: torch.Tensor, config: PreprocessConfig) -> Dict[str, torch.Tensor]:
    stft_mag = compute_stft_magnitude(x, config)
    x_freq = freq_view_from_mag(stft_mag)
    x_tf = tf_view_from_mag(stft_mag, flatten=config.tf_flatten)
    if config.tf_log1p:
        x_tf = x_tf.log1p()
    return {
        "x_freq": x_freq,
        "x_tf": x_tf,
        "stft_mag": stft_mag,
    }


def apply_per_channel(x: torch.Tensor, fn, *args, **kwargs) -> torch.Tensor:
    x = ensure_ct_sample(x, "x")
    if x.shape[0] == 1:
        return fn(x.squeeze(0), *args, **kwargs).unsqueeze(0)
    return torch.stack([fn(channel, *args, **kwargs) for channel in x], dim=0)


def build_augmented_triviews(
    x: torch.Tensor,
    config: PreprocessConfig,
    shift_bins: float,
    scale_ratio: float,
    color_gains: torch.Tensor,
    shift_mode: str = "zero",
) -> Dict[str, torch.Tensor]:
    x = ensure_ct_sample(x, "x")
    base = build_triview_from_time(x, config)

    shift_mag = band_shift_stft(base["stft_mag"], shift_bins, shift_mode=shift_mode)
    x_shift_freq = freq_view_from_mag(shift_mag)
    x_shift_tf = tf_view_from_mag(shift_mag, flatten=config.tf_flatten)
    if config.tf_log1p:
        x_shift_tf = x_shift_tf.log1p()

    x_scale = apply_per_channel(x, frequency_scale_time, scale_ratio)
    scale_views = build_triview_from_time(x_scale, config)

    x_color = apply_per_channel(x, spectral_coloring, color_gains)
    color_views = build_triview_from_time(x_color, config)

    return {
        "x_freq": base["x_freq"],
        "x_tf": base["x_tf"],
        "x_shift_freq": x_shift_freq,
        "x_shift_tf": x_shift_tf,
        "x_scale": x_scale,
        "x_scale_freq": scale_views["x_freq"],
        "x_scale_tf": scale_views["x_tf"],
        "x_color": x_color,
        "x_color_freq": color_views["x_freq"],
        "x_color_tf": color_views["x_tf"],
    }
