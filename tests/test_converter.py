import math

import pytest
import torch

from foasalsa import FoaConverter
from foasalsa.geometry import direction_vector
from foasalsa.synth import band_noise, circular_array, plane_wave, tetrahedral_array

FS = 24000.0


def _band_ratio(foa: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Mean (X, Y, Z) / W transfer ratio over a frequency band."""
    spec = torch.fft.rfft(foa.double(), dim=-1)
    freqs = torch.fft.rfftfreq(foa.shape[-1], 1.0 / FS)
    band = (freqs > low) & (freqs < high)
    return (spec[1:, band] / spec[0, band]).mean(dim=-1)


def test_output_shape_and_alignment():
    mics = circular_array(6, 0.03)
    conv = FoaConverter(sample_rate=FS)
    # a band-limited pulse hitting all mics simultaneously (a source from above: W only for a planar array)
    t = torch.arange(4000, dtype=torch.float64) - 1000
    pulse = torch.sinc(t * 2000.0 / FS) * torch.exp(-((t / 300.0) ** 2))
    audio = pulse[None, None].expand(2, 6, -1).float()
    foa = conv.convert(audio, mics)
    assert foa.shape == (2, 4, 4000)
    assert foa[0, 0].argmax().item() == 1000  # group delay compensated
    assert foa[0, 0, 1000] > 0.9  # unit W gain at low frequencies
    assert foa[0, 1:].abs().max() < 1e-4  # no directional components
    assert torch.allclose(foa[0], foa[1])


def test_planar_array_azimuth_and_zero_z():
    torch.manual_seed(0)
    mics = circular_array(6, 0.03)
    conv = FoaConverter(sample_rate=FS)
    s = band_noise(int(FS * 2), FS, 300, 3000)
    for az in (0.0, 60.0, -135.0):
        p = plane_wave(s, direction_vector(az, 0.0), mics, FS)
        foa = conv.convert(p[None].float(), mics)[0]
        ratio = _band_ratio(foa, 400, 2500)
        est_az = math.degrees(math.atan2(ratio[1].real, ratio[0].real))
        assert abs((est_az - az + 180) % 360 - 180) < 1.0
        assert ratio[2].abs() < 1e-6  # planar array: no Z


def test_3d_array_recovers_elevation():
    torch.manual_seed(0)
    mics = tetrahedral_array(0.03)
    conv = FoaConverter(sample_rate=FS)
    s = band_noise(int(FS * 2), FS, 300, 3000)
    doa = direction_vector(30.0, 40.0)
    p = plane_wave(s, doa, mics, FS)
    foa = conv.convert(p[None].float(), mics)[0]
    ratio = _band_ratio(foa, 400, 2500).real
    est = ratio / ratio.norm()
    assert torch.allclose(est.double(), doa, atol=0.05)


def test_validation():
    conv = FoaConverter()
    with pytest.raises(ValueError):
        conv.convert(torch.zeros(1, 6, 100), circular_array(4))
    with pytest.raises(ValueError):
        conv.convert(torch.zeros(6, 100), circular_array(6))
