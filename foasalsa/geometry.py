"""Direction / steering-vector helpers shared by the converter and the synthesizer.

Conventions
-----------
* Coordinates are right-handed Cartesian (x forward, y left, z up), in meters.
* A direction of arrival (DOA) is the unit vector pointing *from* the array
  *towards* the source: ``n = (cos(az) cos(el), sin(az) cos(el), sin(el))``.
* A plane wave from DOA ``n`` reaches microphone ``m`` at ``r_m`` as
  ``p_m(t) = s(t + n . r_m / c)`` (microphones closer to the source hear it
  earlier), i.e. ``P_m(f) = S(f) exp(+j 2 pi f n . r_m / c)``.
"""

from __future__ import annotations

import math

import torch


def direction_vector(azimuth_deg, elevation_deg=0.0) -> torch.Tensor:
    """Unit DOA vector(s) ``[..., 3]`` from azimuth / elevation in degrees."""
    az = torch.deg2rad(torch.as_tensor(azimuth_deg, dtype=torch.float64))
    el = torch.deg2rad(torch.as_tensor(elevation_deg, dtype=torch.float64))
    az, el = torch.broadcast_tensors(az, el)
    return torch.stack(
        [torch.cos(az) * torch.cos(el), torch.sin(az) * torch.cos(el), torch.sin(el)], dim=-1
    )


def azimuth_elevation(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuth and elevation in degrees of (not necessarily unit) vectors ``[..., 3]``."""
    az = torch.rad2deg(torch.atan2(vec[..., 1], vec[..., 0]))
    horiz = torch.sqrt(vec[..., 0] ** 2 + vec[..., 1] ** 2)
    el = torch.rad2deg(torch.atan2(vec[..., 2], horiz))
    return az, el


def wrap_degrees(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles (degrees) into ``[-180, 180)``."""
    return (x + 180.0) % 360.0 - 180.0


def fibonacci_sphere(n: int) -> torch.Tensor:
    """``n`` approximately uniformly distributed unit vectors ``[n, 3]`` (float64)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def symmetric_sphere_grid(n_base: int) -> torch.Tensor:
    """A sphere grid that is exactly symmetric under x, y and z sign flips.

    Symmetry matters for the least-squares FOA encoder: for a planar array the
    grid symmetry guarantees that the Z channel is exactly zero instead of
    picking up quadrature noise from an asymmetric grid.
    """
    base = fibonacci_sphere(n_base)
    out = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                out.append(base * torch.tensor([sx, sy, sz], dtype=torch.float64))
    return torch.cat(out, dim=0)


def steering_vectors(
    freqs: torch.Tensor, directions: torch.Tensor, mics: torch.Tensor, speed_of_sound: float
) -> torch.Tensor:
    """Far-field steering matrices ``[K, C, D]`` (complex128).

    ``H[k, m, d] = exp(+j 2 pi f_k (n_d . r_m) / c)`` for frequencies ``freqs [K]``,
    unit directions ``directions [D, 3]`` and microphone positions ``mics [C, 3]``.
    """
    freqs = torch.as_tensor(freqs, dtype=torch.float64)
    directions = torch.as_tensor(directions, dtype=torch.float64)
    mics = torch.as_tensor(mics, dtype=torch.float64)
    delay = (mics @ directions.T) / speed_of_sound  # [C, D] seconds (negative = arrives later)
    phase = 2.0 * math.pi * freqs[:, None, None] * delay[None]  # [K, C, D]
    return torch.polar(torch.ones_like(phase), phase)
