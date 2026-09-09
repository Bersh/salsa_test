"""A-format (raw microphone array) to first-order ambisonics (B-format) conversion.

The converter treats the array as a set of omnidirectional pressure sensors at
known relative positions and designs, for every frequency, the linear encoder
``E(f)`` [4 x C] that best maps far-field plane waves from all directions onto the
SN3D first-order spherical harmonics ``(W, X, Y, Z) = (1, n_x, n_y, n_z)``:

    E(f) = argmin_E || E H(f) - Y ||_F^2 + lambda || E ||_F^2

where ``H(f)`` [C x D] holds the array steering vectors for a dense grid of
directions and ``Y`` [4 x D] the corresponding harmonics. The Tikhonov term
``lambda`` bounds the amplification of the pressure-gradient (X, Y, Z) channels at
low frequencies, where the array is small compared to the wavelength.

The frequency responses are turned into linear-phase FIR filters and applied with
a single grouped convolution, so the conversion is a plain time-domain filter bank
with no STFT parameters.

Two physical limits follow from the geometry and show up in the SALSA examples:

* **Spatial aliasing.** Above roughly ``c / (2 d)`` (``d`` = microphone spacing)
  different directions produce indistinguishable phase patterns and the encoder
  (and hence any DOA cue) becomes unreliable. Choose ``FoaSalsa(fmax=...)``
  accordingly.
* **Planar arrays.** If all microphones lie in a plane, the array carries no
  information about the normal component of the DOA. The symmetric direction grid
  makes the corresponding harmonic (Z for the x-y plane) exactly zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .geometry import steering_vectors, symmetric_sphere_grid


class FoaConverter:
    """Converts an A-format microphone-array batch into B-format (FOA, SN3D).

    Output channel order is ``W, X, Y, Z`` (the order used in the SALSA paper,
    Eq. 6). For ACN ordering (``W, Y, Z, X``) index the result with
    ``foa[:, [0, 2, 3, 1]]``.

    Args:
        sample_rate: sampling rate of the audio in Hz.
        n_taps: length of the FIR encoding filters (even). Longer filters give a
            finer frequency resolution of the encoder; 512 taps at 24 kHz is ample.
        speed_of_sound: in m/s.
        max_gain_db: cap on the amplification of the regularised least-squares
            encoder. Larger values track the ideal harmonics further down in
            frequency at the cost of amplifying sensor noise in the X/Y/Z channels.
        n_directions: number of base directions of the (8-fold symmetric) fitting
            grid; the encoder is fitted on ``8 * n_directions`` directions.
    """

    def __init__(
        self,
        sample_rate: float = 24000.0,
        n_taps: int = 512,
        speed_of_sound: float = 343.0,
        max_gain_db: float = 20.0,
        n_directions: int = 96,
    ) -> None:
        if n_taps % 2 or n_taps < 4:
            raise ValueError("n_taps must be an even integer >= 4")
        self.sample_rate = float(sample_rate)
        self.n_taps = int(n_taps)
        self.speed_of_sound = float(speed_of_sound)
        self.max_gain_db = float(max_gain_db)
        self.n_directions = int(n_directions)
        self._cache: dict = {}

    # ------------------------------------------------------------------ design
    @property
    def delay(self) -> int:
        """Group delay (samples) of the linear-phase encoding filters."""
        return self.n_taps // 2

    @property
    def frequencies(self) -> torch.Tensor:
        """Frequency grid ``[K]`` on which the encoder is designed (K = n_taps/2 + 1)."""
        k = torch.arange(self.n_taps // 2 + 1, dtype=torch.float64)
        return k * self.sample_rate / self.n_taps

    @staticmethod
    def _check_mics(mics: torch.Tensor) -> torch.Tensor:
        mics = torch.as_tensor(mics, dtype=torch.float64).detach().cpu()
        if mics.ndim != 2 or mics.shape[1] != 3:
            raise ValueError(f"mics must have shape [C, 3], got {tuple(mics.shape)}")
        if mics.shape[0] < 4:
            raise ValueError("at least 4 microphones are needed to encode first-order ambisonics")
        return mics

    def encoding_matrices(self, mics: torch.Tensor) -> torch.Tensor:
        """Regularised least-squares encoders ``E`` of shape ``[K, 4, C]`` (complex128).

        ``E[k] @ p(f_k)`` maps the microphone spectra at frequency ``f_k`` to
        ``(W, X, Y, Z)``.
        """
        mics = self._check_mics(mics)
        dirs = symmetric_sphere_grid(self.n_directions)  # [D, 3]
        n_dirs = dirs.shape[0]
        harmonics = torch.cat([torch.ones(n_dirs, 1, dtype=torch.float64), dirs], dim=1).T  # [4, D]
        H = steering_vectors(self.frequencies, dirs, mics, self.speed_of_sound)  # [K, C, D]

        scale = 1.0 / math.sqrt(n_dirs)
        Ht = H * scale
        Yt = (harmonics * scale).to(Ht.dtype)  # [4, D]
        gain = 10.0 ** (self.max_gain_db / 20.0)
        lam = 1.0 / (4.0 * gain * gain)  # max singular gain of s / (s^2 + lam) is 1 / (2 sqrt(lam))

        A = Ht @ Ht.mH + lam * torch.eye(mics.shape[0], dtype=Ht.dtype)  # [K, C, C], Hermitian
        B = Yt[None] @ Ht.mH  # [K, 4, C]
        # E = B A^-1  <=>  E^H = A^-1 B^H (A is Hermitian)
        return torch.linalg.solve(A, B.mH).mH

    def filters(self, mics: torch.Tensor) -> torch.Tensor:
        """Linear-phase FIR encoding filters ``[4, C, n_taps]`` (float64)."""
        E = self.encoding_matrices(mics)  # [K, 4, C]
        k = torch.arange(E.shape[0], dtype=torch.float64)
        phase = torch.polar(torch.ones_like(k), -2.0 * math.pi * k * self.delay / self.n_taps)
        spectrum = (E * phase[:, None, None]).permute(1, 2, 0)  # [4, C, K]
        impulse = torch.fft.irfft(spectrum, n=self.n_taps, dim=-1)
        window = torch.hann_window(self.n_taps, periodic=True, dtype=torch.float64)
        return impulse * window

    # ----------------------------------------------------------------- convert
    def convert(self, audio: torch.Tensor, mics: torch.Tensor) -> torch.Tensor:
        """Converts audio from MIC format (A-format) into First-Order Ambisonics (FOA)
        a.k.a. B-format using the SN3D convention

        Params:
            audio - an input multichannel audio [N, C, T] batch
            mics - a [C, 3] tensor of relative microphone (x, y, z) coordinates, in meters
        Returns:
            foa - output [N, 4, T] batch in FOA format (channels W, X, Y, Z).
        """
        if audio.ndim != 3:
            raise ValueError(f"audio must have shape [N, C, T], got {tuple(audio.shape)}")
        mics = self._check_mics(mics)
        if mics.shape[0] != audio.shape[1]:
            raise ValueError(
                f"audio has {audio.shape[1]} channels but {mics.shape[0]} microphone positions were given"
            )
        key = (tuple(mics.flatten().tolist()), audio.dtype, str(audio.device))
        weight = self._cache.get(key)
        if weight is None:
            # conv1d is a cross-correlation: flip the impulse responses.
            weight = self.filters(mics).flip(-1).to(dtype=audio.dtype, device=audio.device)
            self._cache[key] = weight
        # Compensate the filter group delay so that the output is time-aligned with the input.
        padded = F.pad(audio, (self.n_taps - 1 - self.delay, self.delay))
        return F.conv1d(padded, weight)
