import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_WINDOW_CACHE: Dict[Tuple[int, int, str, str, torch.dtype], torch.Tensor] = {}


@dataclass(frozen=True)
class ScaleParams:
    ratio: float


@dataclass(frozen=True)
class ShiftParams:
    bins: float


@dataclass(frozen=True)
class ColorParams:
    gains: torch.Tensor  # (bands,) linear gains


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    pad = target_len - x.shape[-1]
    return F.pad(x, (0, pad))


def frequency_scale_time(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """Resample in time to simulate speed/sampling-rate changes, then restore length."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    length = x.shape[-1]
    scaled_len = max(1, int(round(length * ratio)))
    x_in = x.unsqueeze(0).unsqueeze(0)
    x_scaled = F.interpolate(x_in, size=scaled_len, mode="linear", align_corners=False)
    x_scaled = x_scaled.squeeze(0).squeeze(0)
    return _match_length(x_scaled, length)


def _roll_preserve_edges(x: torch.Tensor, shift_bins: int, dim: int) -> torch.Tensor:
    """Circular shift interior bins while keeping DC/Nyquist bins fixed."""
    size = x.shape[dim]
    if size <= 2 or shift_bins == 0:
        return x
    index = [slice(None)] * x.dim()
    index[dim] = slice(1, -1)
    interior = x[tuple(index)]
    rolled = torch.roll(interior, shifts=shift_bins, dims=dim)
    out = x.clone()
    out[tuple(index)] = rolled
    return out


def _is_integral_shift(shift_bins: float, eps: float = 1e-8) -> bool:
    return abs(float(shift_bins) - round(float(shift_bins))) < eps


def _reflect_indices(idx: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(idx)
    period = 2 * (size - 1)
    mod = torch.remainder(idx, period)
    return torch.where(mod <= (size - 1), mod, period - mod)


def _gather_last_dim_with_padding(x: torch.Tensor, idx: torch.Tensor, padding_mode: str) -> torch.Tensor:
    size = x.shape[-1]
    if padding_mode == "border":
        mapped = idx.clamp(0, size - 1)
        return x.index_select(-1, mapped)
    if padding_mode == "reflect":
        mapped = _reflect_indices(idx, size)
        return x.index_select(-1, mapped)
    if padding_mode == "zero":
        mapped = idx.clamp(0, size - 1)
        gathered = x.index_select(-1, mapped)
        valid = ((idx >= 0) & (idx < size)).to(dtype=x.real.dtype if x.is_complex() else x.dtype)
        while valid.dim() < gathered.dim():
            valid = valid.unsqueeze(0)
        return gathered * valid
    raise ValueError(f"Unsupported padding_mode: {padding_mode}")


def _shift_along_dim_interp(
    x: torch.Tensor,
    shift_bins: float,
    dim: int,
    padding_mode: str,
) -> torch.Tensor:
    x_m = x.movedim(dim, -1)
    size = x_m.shape[-1]
    if size == 0 or shift_bins == 0:
        return x
    pos = torch.arange(size, device=x_m.device, dtype=torch.float32) - float(shift_bins)
    idx0 = torch.floor(pos).to(torch.long)
    idx1 = idx0 + 1
    weight = (pos - idx0.to(pos.dtype)).to(dtype=x_m.real.dtype if x_m.is_complex() else x_m.dtype)
    x0 = _gather_last_dim_with_padding(x_m, idx0, padding_mode=padding_mode)
    x1 = _gather_last_dim_with_padding(x_m, idx1, padding_mode=padding_mode)
    while weight.dim() < x0.dim():
        weight = weight.unsqueeze(0)
    shifted = x0 * (1.0 - weight) + x1 * weight
    return shifted.movedim(-1, dim)


def _shift_stft_bins(stft: torch.Tensor, shift_bins: float, dim: int, shift_mode: str = "zero") -> torch.Tensor:
    """Shift STFT bins with either zero padding or circular roll."""
    if shift_bins == 0:
        return stft
    if shift_mode == "circular":
        if not _is_integral_shift(shift_bins):
            raise ValueError("circular shift_mode only supports integer shift bins.")
        return _roll_preserve_edges(stft, shift_bins=int(round(shift_bins)), dim=dim)
    if shift_mode in {"border", "reflect"}:
        return _shift_along_dim_interp(stft, shift_bins=shift_bins, dim=dim, padding_mode=shift_mode)
    if shift_mode != "zero":
        raise ValueError(f"Unsupported shift_mode: {shift_mode}")
    if not _is_integral_shift(shift_bins):
        return _shift_along_dim_interp(stft, shift_bins=shift_bins, dim=dim, padding_mode="zero")
    shifted = torch.zeros_like(stft)
    size = stft.shape[dim]
    k = abs(int(round(shift_bins)))
    if k >= size:
        return shifted
    dst = [slice(None)] * stft.dim()
    src = [slice(None)] * stft.dim()
    if float(shift_bins) > 0:
        dst[dim] = slice(k, None)
        src[dim] = slice(None, -k)
    else:
        dst[dim] = slice(None, -k)
        src[dim] = slice(k, None)
    shifted[tuple(dst)] = stft[tuple(src)]
    return shifted


def band_shift_stft(stft: torch.Tensor, shift_bins: float, shift_mode: str = "zero") -> torch.Tensor:
    """Shift frequency bins in an STFT magnitude tensor."""
    if stft.dim() == 2:
        dim = 0
    elif stft.dim() == 3:
        dim = 1
    elif stft.dim() == 4:
        dim = 2
    else:
        raise ValueError("stft must have 2-4 dimensions")
    return _shift_stft_bins(stft, shift_bins, dim, shift_mode=shift_mode)


def _shift_spectrum_bins(spectrum: torch.Tensor, shift_bins: float, shift_mode: str = "zero") -> torch.Tensor:
    """Shift spectrum bins with zero padding or circular roll."""
    if shift_bins == 0:
        return spectrum
    if shift_mode == "circular":
        if not _is_integral_shift(shift_bins):
            raise ValueError("circular shift_mode only supports integer shift bins.")
        return _roll_preserve_edges(spectrum, shift_bins=int(round(shift_bins)), dim=-1)
    if shift_mode in {"border", "reflect"}:
        return _shift_along_dim_interp(spectrum, shift_bins=shift_bins, dim=-1, padding_mode=shift_mode)
    if shift_mode != "zero":
        raise ValueError(f"Unsupported shift_mode: {shift_mode}")
    if not _is_integral_shift(shift_bins):
        return _shift_along_dim_interp(spectrum, shift_bins=shift_bins, dim=-1, padding_mode="zero")
    shifted = torch.zeros_like(spectrum)
    bins = spectrum.shape[-1]
    k = abs(int(round(shift_bins)))
    if k >= bins:
        return shifted
    if float(shift_bins) > 0:
        shifted[..., k:] = spectrum[..., :-k]
    else:
        shifted[..., :-k] = spectrum[..., k:]
    return shifted


def band_shift_time(x: torch.Tensor, shift_bins: float, shift_mode: str = "zero") -> torch.Tensor:
    """Shift frequency bins in the rFFT spectrum, then invert back to time."""
    spectrum = torch.fft.rfft(x)
    shifted = _shift_spectrum_bins(spectrum, shift_bins, shift_mode=shift_mode)
    return torch.fft.irfft(shifted, n=x.shape[-1])


def _get_window(
    *,
    n_fft: int,
    win_length: Optional[int],
    window_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    resolved_win_length = int(win_length) if win_length is not None else int(n_fft)
    key = (int(n_fft), resolved_win_length, window_name, str(device), dtype)
    window = _WINDOW_CACHE.get(key)
    if window is not None and window.device == device and window.dtype == dtype:
        return window

    if window_name == "hann":
        window = torch.hann_window(resolved_win_length, device=device, dtype=dtype)
    elif window_name == "hamming":
        window = torch.hamming_window(resolved_win_length, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported window_name: {window_name}")
    _WINDOW_CACHE[key] = window
    return window


def band_shift_time_stft(
    x: torch.Tensor,
    shift_bins: float,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    window_name: str = "hann",
    center: bool = True,
    shift_mode: str = "zero",
) -> torch.Tensor:
    """Shift STFT frequency bins then invert to time using the same STFT convention."""
    window = _get_window(
        n_fft=n_fft,
        win_length=win_length,
        window_name=window_name,
        device=x.device,
        dtype=x.dtype,
    )
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    shifted = _shift_stft_bins(stft, shift_bins, dim=-2, shift_mode=shift_mode)
    return torch.istft(
        shifted,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=x.shape[-1],
    )


def spectral_coloring(x: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Apply spectral coloring in the frequency domain using linear gains."""
    spectrum = torch.fft.rfft(x)
    gains = gains.to(spectrum.device)
    if gains.numel() != spectrum.numel():
        gains = F.interpolate(
            gains.view(1, 1, -1), size=spectrum.numel(), mode="linear", align_corners=False
        ).view(-1)
    colored = spectrum * gains
    return torch.fft.irfft(colored, n=x.shape[-1])


def make_coloring_gains(
    num_bins: int,
    bands: int,
    max_gain_db: float,
    return_band_gains: bool = False,
    active_bands: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create a smooth random EQ curve in linear gains.

    If return_band_gains is True, returns (full_gains, band_gains).
    """
    if bands <= 0:
        raise ValueError("bands must be positive")
    random_db = torch.empty(bands).uniform_(-max_gain_db, max_gain_db, generator=generator)
    random_linear = torch.pow(10.0, random_db / 20.0)
    if active_bands is not None:
        if len(active_bands) == 0:
            raise ValueError("active_bands must be non-empty when provided.")
        mask = torch.zeros(bands, dtype=torch.bool)
        for idx in active_bands:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= bands:
                raise ValueError(
                    f"active band index {idx_int} out of range for bands={bands}."
                )
            mask[idx_int] = True
        # Keep inactive bands neutral so we can build explicit band holdout protocols.
        random_linear = torch.where(mask, random_linear, torch.ones_like(random_linear))
    gains = F.interpolate(
        random_linear.view(1, 1, -1),
        size=num_bins,
        mode="linear",
        align_corners=True,
    ).view(-1)
    if return_band_gains:
        return gains, random_linear
    return gains


def stft_magnitude(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: Optional[torch.Tensor] = None,
    win_length: Optional[int] = None,
    window_name: str = "hann",
    center: bool = True,
    magnitude_power: float = 1.0,
) -> torch.Tensor:
    if window is None:
        window = _get_window(
            n_fft=n_fft,
            win_length=win_length,
            window_name=window_name,
            device=x.device,
            dtype=x.dtype,
        )
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    magnitude = stft.abs()
    if magnitude_power != 1.0:
        if magnitude_power <= 0:
            raise ValueError("magnitude_power must be positive.")
        magnitude = magnitude.pow(magnitude_power)
    return magnitude


def spectral_centroid(magnitude: torch.Tensor) -> torch.Tensor:
    """Compute spectral centroid along frequency axis."""
    freq_bins = torch.arange(magnitude.shape[0], device=magnitude.device, dtype=magnitude.dtype)
    weights = magnitude.sum(dim=-1) + 1e-8
    centroid = (freq_bins * magnitude.sum(dim=-1)).sum() / weights.sum()
    return centroid


def spectral_bandwidth(magnitude: torch.Tensor) -> torch.Tensor:
    """Compute spectral bandwidth (std around centroid) along frequency axis."""
    freq_bins = torch.arange(magnitude.shape[0], device=magnitude.device, dtype=magnitude.dtype)
    per_bin_energy = magnitude.sum(dim=-1)
    total_energy = per_bin_energy.sum() + 1e-8
    centroid = (freq_bins * per_bin_energy).sum() / total_energy
    variance = (((freq_bins - centroid) ** 2) * per_bin_energy).sum() / total_energy
    return torch.sqrt(variance + 1e-12)


def build_meta(scale_params: ScaleParams, shift_params: ShiftParams, color_params: ColorParams) -> Dict[str, torch.Tensor]:
    return {
        "scale_ratio": torch.tensor(scale_params.ratio),
        "shift_bins": torch.tensor(shift_params.bins),
        "color_gains": color_params.gains.clone(),
    }
