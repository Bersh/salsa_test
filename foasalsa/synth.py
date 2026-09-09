"""Synthetic multichannel signals for testing and for the examples.

All functions return float64 tensors on the CPU. Sample rate ``fs`` in Hz.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from .geometry import direction_vector, fibonacci_sphere

# ----------------------------------------------------------------- arrays


def circular_array(n_mics: int = 6, radius: float = 0.03, z: float = 0.0) -> torch.Tensor:
    """``[n_mics, 3]`` planar circular array in the x-y plane (first mic on +x)."""
    ang = torch.arange(n_mics, dtype=torch.float64) * 2.0 * math.pi / n_mics
    return torch.stack([radius * torch.cos(ang), radius * torch.sin(ang), torch.full_like(ang, z)], dim=1)


def centred_circular_array(n_ring: int = 8, radius: float = 0.06) -> torch.Tensor:
    """``[n_ring + 1, 3]`` planar ring plus a centre microphone.

    The centre microphone gives the FOA encoder a clean W (omni) channel at all
    frequencies: the average over an open ring alone crosses zero at ``k r = 2.4``
    (2.2 kHz for r = 6 cm), which flips the sign of X/W and Y/W above it.
    """
    return torch.cat([circular_array(n_ring, radius), torch.zeros(1, 3, dtype=torch.float64)], dim=0)


def square_array(side: float = 0.08) -> torch.Tensor:
    """``[4, 3]`` planar square array in the x-y plane."""
    h = side / 2.0
    return torch.tensor([[h, h, 0.0], [-h, h, 0.0], [-h, -h, 0.0], [h, -h, 0.0]], dtype=torch.float64)


def tetrahedral_array(radius: float = 0.03) -> torch.Tensor:
    """``[4, 3]`` regular tetrahedron (a 3-D array, used in the tests only)."""
    v = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], dtype=torch.float64
    )
    return v / math.sqrt(3.0) * radius


# --------------------------------------------------------------- propagation


def plane_wave(signal: torch.Tensor, doa: torch.Tensor, mics: torch.Tensor, fs: float, c: float = 343.0) -> torch.Tensor:
    """Static far-field plane wave: ``[C, T]`` microphone signals for a source signal ``[T]``.

    Uses exact (FFT based) fractional delays: ``p_m(t) = s(t + n . r_m / c)``.
    ``doa`` is a unit vector ``[3]``.
    """
    signal = torch.as_tensor(signal, dtype=torch.float64)
    mics = torch.as_tensor(mics, dtype=torch.float64)
    doa = torch.as_tensor(doa, dtype=torch.float64)
    max_delay = int(math.ceil(mics.norm(dim=1).max().item() / c * fs)) + 8
    padded = torch.nn.functional.pad(signal, (max_delay, max_delay))
    spec = torch.fft.rfft(padded)
    freqs = torch.fft.rfftfreq(padded.numel(), 1.0 / fs)
    advance = (mics @ doa) / c  # seconds, positive = earlier
    phase = 2.0 * math.pi * freqs[None] * advance[:, None]
    out = torch.fft.irfft(spec[None] * torch.polar(torch.ones_like(phase), phase), n=padded.numel())
    return out[:, max_delay:-max_delay]


def _read_delayed(signal: torch.Tensor, advance_samples: torch.Tensor, oversample: int = 8) -> torch.Tensor:
    """Reads ``signal[t + advance[k, t]]`` for every row ``k`` with fractional-sample accuracy.

    The signal is band-limited, so it is upsampled ``oversample`` times with an FFT
    and read out with linear interpolation. Returns ``[K, T]``.
    """
    signal = torch.as_tensor(signal, dtype=torch.float64)
    n = signal.numel()
    spec = torch.fft.rfft(signal)
    up = torch.zeros(n * oversample // 2 + 1, dtype=spec.dtype)
    up[: spec.numel()] = spec
    s_up = torch.fft.irfft(up, n=n * oversample) * oversample
    t = torch.arange(n, dtype=torch.float64)
    pos = ((t[None] + advance_samples) * oversample).clamp(0.0, n * oversample - 1.001)
    i0 = pos.floor().long()
    frac = pos - i0
    return s_up[i0] * (1.0 - frac) + s_up[i0 + 1] * frac


def moving_plane_wave(
    signal: torch.Tensor, doas: torch.Tensor, mics: torch.Tensor, fs: float, c: float = 343.0, oversample: int = 8
) -> torch.Tensor:
    """Plane wave with a time-varying DOA ``doas [T, 3]`` (unit vectors per sample). Returns ``[C, T]``."""
    mics = torch.as_tensor(mics, dtype=torch.float64)
    doas = torch.as_tensor(doas, dtype=torch.float64)
    advance = (doas @ mics.T).T / c * fs  # [C, T] samples, positive = earlier
    return _read_delayed(signal, advance, oversample)


def flyby(
    signal: torch.Tensor,
    positions: torch.Tensor,
    mics: torch.Tensor,
    fs: float,
    c: float = 343.0,
    reference_distance: float = 1.0,
    oversample: int = 8,
) -> torch.Tensor:
    """A source moving along ``positions [T, 3]`` (metres, relative to the array centre).

    Includes the propagation delay from the source to the array (hence Doppler
    shift), ``reference_distance / distance`` spherical spreading and the
    per-microphone plane-wave delays. Returns ``[C, T]``.
    """
    positions = torch.as_tensor(positions, dtype=torch.float64)
    dist = positions.norm(dim=1)  # [T]
    doas = positions / dist[:, None]
    # signal as heard at the array centre: emitted at t - dist / c, attenuated by 1 / r
    centre = _read_delayed(signal, (-dist / c * fs)[None], oversample)[0] * (reference_distance / dist)
    return moving_plane_wave(centre, doas, mics, fs, c, oversample)


def diffuse_noise(
    n_samples: int,
    mics: torch.Tensor,
    fs: float,
    c: float = 343.0,
    n_directions: int = 48,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Isotropic diffuse noise ``[C, T]`` (unit variance) as a sum of plane waves from all directions."""
    dirs = fibonacci_sphere(n_directions)
    out = torch.zeros(mics.shape[0], n_samples, dtype=torch.float64)
    for d in dirs:
        s = torch.randn(n_samples, dtype=torch.float64, generator=generator)
        out += plane_wave(s, d, mics, fs, c)
    return out / math.sqrt(n_directions)


# ------------------------------------------------------------------ sources


def burst_envelope(
    n_samples: int,
    fs: float,
    on_range: Sequence[float] = (0.08, 0.25),
    off_range: Sequence[float] = (0.05, 0.15),
    start: float = 0.0,
    stop: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Syllable-like on/off envelope ``[T]`` (raised-cosine ramps) between ``start`` and ``stop`` seconds."""
    env = torch.zeros(n_samples, dtype=torch.float64)
    stop = n_samples / fs if stop is None else stop
    t = start
    ramp = 0.02
    while t < stop:
        on = on_range[0] + (on_range[1] - on_range[0]) * torch.rand(1, generator=generator).item()
        on = min(on, stop - t)
        a, b = int(t * fs), int((t + on) * fs)
        seg = torch.ones(b - a, dtype=torch.float64)
        nr = min(int(ramp * fs), (b - a) // 2)
        if nr > 0:
            r = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, nr, dtype=torch.float64))
            seg[:nr] *= r
            seg[-nr:] *= r.flip(0)
        env[a:b] = seg
        off = off_range[0] + (off_range[1] - off_range[0]) * torch.rand(1, generator=generator).item()
        t += on + off
    return env


def harmonic_source(
    n_samples: int,
    fs: float,
    f0: float = 140.0,
    n_harmonics: int = 30,
    start: float = 0.0,
    stop: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Speech-like signal ``[T]``: harmonic complex with a wandering f0 and syllable bursts."""
    t = torch.arange(n_samples, dtype=torch.float64) / fs
    phi = 2.0 * math.pi * torch.rand(2, generator=generator, dtype=torch.float64)
    f0_t = f0 * (1.0 + 0.12 * torch.sin(2 * math.pi * 0.7 * t + phi[0]) + 0.05 * torch.sin(2 * math.pi * 2.3 * t + phi[1]))
    phase = 2.0 * math.pi * torch.cumsum(f0_t, 0) / fs
    sig = torch.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        if h * f0 * 1.2 >= 0.45 * fs:
            break
        # gentle formant-like weighting: emphasise 300-800 Hz and 1.5-2.5 kHz
        fh = h * f0
        weight = 1.0 / h + 0.6 * math.exp(-((fh - 550) / 250) ** 2) + 0.3 * math.exp(-((fh - 2000) / 400) ** 2)
        sig += weight * torch.sin(h * phase)
    env = burst_envelope(n_samples, fs, start=start, stop=stop, generator=generator)
    sig = sig * env
    return sig / sig.abs().max()


def band_noise(
    n_samples: int,
    fs: float,
    low: float,
    high: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Stationary Gaussian noise ``[T]`` band-limited to ``[low, high]`` Hz (unit RMS)."""
    x = torch.randn(n_samples, dtype=torch.float64, generator=generator)
    spec = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n_samples, 1.0 / fs)
    spec[(freqs < low) | (freqs > high)] = 0.0
    y = torch.fft.irfft(spec, n=n_samples)
    return y / y.pow(2).mean().sqrt()


def noise_bursts(
    n_samples: int,
    fs: float,
    low: float = 200.0,
    high: float = 4000.0,
    start: float = 0.0,
    stop: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Band-limited noise bursts ``[T]`` (peak-normalised)."""
    x = band_noise(n_samples, fs, low, high, generator) * burst_envelope(
        n_samples, fs, on_range=(0.05, 0.15), off_range=(0.08, 0.2), start=start, stop=stop, generator=generator
    )
    return x / x.abs().max()


def drone_source(
    n_samples: int,
    fs: float,
    rpm: float = 12000.0,
    n_blades: int = 2,
    n_motors: int = 4,
    motor_spread: float = 0.04,
    wander: float = 0.03,
    n_harmonics: int = 25,
    noise_ratio: float = 0.2,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Multirotor propeller sound ``[T]`` (peak-normalised).

    Each motor produces harmonics of its blade-passing frequency
    (``rpm / 60 * n_blades``) plus weaker harmonics of the shaft rotation. The
    motors run at slightly different speeds (``motor_spread``) that wander slowly
    in time (``wander``), which produces the beating clusters of lines typical of
    a quadcopter. A broadband turbulence floor is added at ``noise_ratio`` times
    the RMS of the harmonic part.
    """
    t = torch.arange(n_samples, dtype=torch.float64) / fs
    sig = torch.zeros_like(t)
    for _ in range(n_motors):
        r = torch.rand(6, generator=generator, dtype=torch.float64)
        offset = (r[0] - 0.5) * 2.0 * motor_spread
        f_mod = 0.2 + 1.3 * r[1:3]
        phi = 2.0 * math.pi * r[3:5]
        modulation = 1.0 + wander * (0.6 * torch.sin(2 * math.pi * f_mod[0] * t + phi[0]) + 0.4 * torch.sin(2 * math.pi * f_mod[1] * t + phi[1]))
        f_rot = rpm / 60.0 * (1.0 + offset) * modulation
        phase = 2.0 * math.pi * torch.cumsum(f_rot, 0) / fs
        f_max = f_rot.max().item()
        for h in range(1, n_harmonics + 1):
            if h * n_blades * f_max >= 0.45 * fs:
                break
            sig += (1.0 / h ** 1.1) * torch.sin(h * n_blades * phase + 2 * math.pi * r[5] * h)
        for k in range(1, 2 * n_blades * 4):
            if k % n_blades == 0 or k * f_max >= 0.45 * fs:
                continue
            sig += (0.12 / k) * torch.sin(k * phase + 2 * math.pi * r[5] * k)
    turbulence = band_noise(n_samples, fs, 200.0, min(8000.0, 0.45 * fs), generator)
    sig = sig + noise_ratio * sig.pow(2).mean().sqrt() * turbulence
    return sig / sig.abs().max()


def air_absorption(signal: torch.Tensor, distance_m: float, fs: float, db_per_m_at_1khz: float = 0.01, exponent: float = 1.3) -> torch.Tensor:
    """Atmospheric absorption over ``distance_m``: ``db_per_m_at_1khz * (f / 1 kHz) ** exponent`` dB per metre.

    A rough fit to ISO 9613 at 20 °C and 50 % humidity (about 1 dB per 100 m at
    1 kHz and 6 dB per 100 m at 4 kHz). Spherical spreading is not included.
    """
    signal = torch.as_tensor(signal, dtype=torch.float64)
    spec = torch.fft.rfft(signal)
    freqs = torch.fft.rfftfreq(signal.numel(), 1.0 / fs)
    att_db = db_per_m_at_1khz * distance_m * (freqs / 1000.0) ** exponent
    return torch.fft.irfft(spec * 10 ** (-att_db / 20.0), n=signal.numel())


def wind_noise(n_samples: int, fs: float, cutoff: float = 300.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Low-frequency-heavy noise ``[T]`` (unit RMS) falling 12 dB per octave above ``cutoff``.

    Wind noise is generated by turbulence at each microphone and is therefore
    uncorrelated between microphones: call this once per microphone.
    """
    x = torch.randn(n_samples, dtype=torch.float64, generator=generator)
    spec = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n_samples, 1.0 / fs)
    spec = spec / (1.0 + (freqs / cutoff) ** 2)
    spec[0] = 0.0
    y = torch.fft.irfft(spec, n=n_samples)
    return y / y.pow(2).mean().sqrt()


def source_scene(
    mics: torch.Tensor,
    fs: float,
    sources: Sequence[tuple[torch.Tensor, float, float]],
    c: float = 343.0,
) -> torch.Tensor:
    """Mixes static plane-wave sources: ``sources`` is a list of ``(signal [T], azimuth_deg, elevation_deg)``."""
    out = None
    for signal, az, el in sources:
        p = plane_wave(signal, direction_vector(az, el), mics, fs, c)
        out = p if out is None else out + p
    return out
