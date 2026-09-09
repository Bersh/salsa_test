import torch

from foasalsa.synth import air_absorption, circular_array, drone_source, flyby, wind_noise

FS = 24000.0


def test_drone_source_has_blade_passing_harmonics():
    g = torch.Generator().manual_seed(0)
    s = drone_source(int(FS), FS, rpm=12000.0, n_blades=2, n_motors=1, motor_spread=0.0, wander=0.0, noise_ratio=0.0, generator=g)
    spec = torch.fft.rfft(s).abs()
    freqs = torch.fft.rfftfreq(s.numel(), 1.0 / FS)
    peak = freqs[spec.argmax()].item()
    assert abs(peak - 400.0) < 5.0  # 12000 rpm * 2 blades / 60 s


def test_air_absorption_is_a_low_pass():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(int(FS), dtype=torch.float64, generator=g)
    y = air_absorption(x, 300.0, FS)
    X, Y = torch.fft.rfft(x).abs(), torch.fft.rfft(y).abs()
    freqs = torch.fft.rfftfreq(x.numel(), 1.0 / FS)
    low = (freqs > 100) & (freqs < 300)
    high = (freqs > 6000) & (freqs < 8000)
    assert (Y[low] / X[low]).mean() > 0.9
    assert (Y[high] / X[high]).mean() < 0.3


def test_wind_noise_is_low_frequency():
    g = torch.Generator().manual_seed(0)
    w = wind_noise(int(FS), FS, generator=g)
    spec = torch.fft.rfft(w).abs()
    freqs = torch.fft.rfftfreq(w.numel(), 1.0 / FS)
    assert spec[(freqs > 50) & (freqs < 200)].mean() > 10 * spec[(freqs > 2000) & (freqs < 4000)].mean()


def test_flyby_doppler_shift():
    g = torch.Generator().manual_seed(0)
    mics = circular_array(8, 0.06)
    n = int(FS * 2)
    t = torch.arange(n, dtype=torch.float64) / FS
    tone = torch.sin(2 * torch.pi * 1000.0 * t)
    # approaching along +x at 34.3 m/s: expect about +10 % frequency shift
    pos = torch.stack([200.0 - 34.3 * t, torch.full_like(t, 1.0), torch.zeros_like(t)], dim=1)
    p = flyby(tone, pos, mics, FS, reference_distance=100.0)
    assert p.shape == (8, n)
    spec = torch.fft.rfft(p[0][n // 4 : -n // 4]).abs()
    freqs = torch.fft.rfftfreq(spec.numel() * 2 - 2, 1.0 / FS)
    peak = freqs[spec.argmax()].item()
    assert 1090 < peak < 1130
