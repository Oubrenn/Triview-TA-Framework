import torch

from src.transforms import (
    band_shift_time,
    band_shift_time_stft,
    frequency_scale_time,
    make_coloring_gains,
    spectral_centroid,
    stft_magnitude,
)


def test_frequency_scale_reproducible():
    x = torch.linspace(0, 1, 256)
    out1 = frequency_scale_time(x, ratio=1.2)
    out2 = frequency_scale_time(x, ratio=1.2)
    assert torch.allclose(out1, out2)


def test_coloring_gain_shape():
    gains = make_coloring_gains(num_bins=128, bands=4, max_gain_db=3.0)
    assert gains.shape == (128,)


def test_coloring_active_band_mask():
    _, band_gains = make_coloring_gains(
        num_bins=128,
        bands=4,
        max_gain_db=3.0,
        return_band_gains=True,
        active_bands=[1, 3],
    )
    assert band_gains.shape == (4,)
    assert torch.allclose(band_gains[[0, 2]], torch.ones(2))


def test_spectral_centroid_monotonic():
    x = torch.sin(torch.linspace(0, 10 * torch.pi, 512))
    mag = stft_magnitude(x, n_fft=128, hop_length=32)
    centroid_base = spectral_centroid(mag)
    x_scaled = frequency_scale_time(x, ratio=0.8)
    mag_scaled = stft_magnitude(x_scaled, n_fft=128, hop_length=32)
    centroid_scaled = spectral_centroid(mag_scaled)
    assert centroid_scaled > centroid_base


def test_fractional_shift_runs_and_preserves_length():
    x = torch.randn(256)
    y = band_shift_time(x, shift_bins=0.25, shift_mode="border")
    z = band_shift_time_stft(x, shift_bins=-0.5, n_fft=64, hop_length=16, shift_mode="reflect")
    assert y.shape == x.shape
    assert z.shape == x.shape


def test_border_shift_has_no_wraparound_boost():
    x = torch.zeros(256)
    x[0] = 1.0
    shifted = band_shift_time(x, shift_bins=0.5, shift_mode="border")
    assert shifted.abs().max().item() <= 1.0 + 1e-6
