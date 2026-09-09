"""SALSA features for first-order ambisonics.

Reference: T. N. T. Nguyen, K. N. Watcharasupat, N. K. Nguyen, D. L. Jones and
W.-S. Gan, "SALSA: Spatial Cue-Augmented Log-Spectrogram Features for Polyphonic
Sound Event Localization and Detection", IEEE/ACM TASLP 2022
(arXiv:2110.00275). Constants follow the paper and its public implementation.

Per time-frequency (TF) bin ``(t, f)`` of the STFT ``X(t, f)`` [4 channels]:

1. **Log-linear spectrograms** ``LinSpec = log(|X|^2)`` for each of the 4 channels.
2. **Spatial covariance** ``R = 1/(2 T_r + 1) sum_{tau=-T_r}^{T_r} X(t+tau, f) X^H(t+tau, f)``
   with ``T_r = 3`` (Eq. 5).
3. **Eigenvector-based intensity vector (EIV)**: with ``U`` the principal
   eigenvector of ``R``, ``U~ = U / U_W`` (normalise by the omni channel), discard
   the first element, take the real part and normalise to unit norm. At a
   single-source bin this approximates the SN3D steering vector
   ``(cos az cos el, sin az cos el, sin el)`` (Eq. 6), i.e. the DOA.
4. **Single-source bin selection**: the EIV is only kept where the bin passes
   * the *magnitude test* (Eq. 9): the 3-frame running RMS of ``|X_W|`` exceeds
     ``alpha_SNR = 1.5`` times an adaptive per-frequency noise floor, and
   * the *coherence test* (Eq. 11): the direct-to-reverberant ratio
     ``sigma_1 / sigma_2`` of the covariance eigenvalues exceeds ``beta_DRR = 5``.
   Elsewhere - and outside the ``[fmin, fmax]`` band (50 Hz - 9 kHz by default,
   which should be lowered to the spatial-aliasing limit of small arrays) - the
   EIV is zero.
5. The 4 spectrogram channels and the 3 EIV channels are stacked: ``[N, 7, F, T]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SalsaResult:
    """Everything computed on the way to the SALSA features (see :meth:`FoaSalsa.analyze`).

    Shapes use N = batch, F = frequency bins, T = frames, Fb = bins inside the band.
    """

    features: torch.Tensor  #: [N, 7, F, T] SALSA features (4 log-spectrograms + 3 EIV channels)
    linspec: torch.Tensor  #: [N, 4, F, T] log-linear spectrograms
    eiv: torch.Tensor  #: [N, 3, F, T] masked eigenvector-based intensity vectors (x, y, z)
    eiv_raw: torch.Tensor  #: [N, 3, F, T] EIV before single-source bin selection (zero outside band)
    stft: torch.Tensor  #: [N, 4, F, T] complex STFT
    eigenvalues: torch.Tensor  #: [N, F, T, 4] covariance eigenvalues, descending (zero outside band)
    drr: torch.Tensor  #: [N, F, T] direct-to-reverberant ratio sigma_1 / sigma_2 (zero outside band)
    noise_floor: torch.Tensor  #: [N, F, T] tracked noise floor of |X_W| (zero outside band)
    magnitude_mask: torch.Tensor  #: [N, F, T] bool, magnitude test passed
    coherence_mask: torch.Tensor  #: [N, F, T] bool, coherence test passed
    band_mask: torch.Tensor  #: [F] bool, bins for which the EIV is computed
    mask: torch.Tensor  #: [N, F, T] bool, bins whose EIV is kept in ``features``
    frequencies: torch.Tensor  #: [F] bin centre frequencies in Hz
    times: torch.Tensor  #: [T] frame centre times in seconds


def _moving_average(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Centred box average of width ``2 radius + 1`` along the last dim (edges use fewer frames)."""
    if radius <= 0:
        return x
    if x.is_complex():
        return torch.complex(_moving_average(x.real, radius), _moving_average(x.imag, radius))
    shape = x.shape
    y = F.avg_pool1d(
        x.reshape(-1, 1, shape[-1]),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
        count_include_pad=False,
    )
    return y.reshape(shape)


class FoaSalsa:
    """Computes SALSA features from FOA (B-format, channels W, X, Y, Z) audio.

    Args:
        sample_rate: sampling rate in Hz (paper: 24 kHz).
        n_fft: FFT size (paper: 512). ``F = n_fft // 2 + 1`` bins are returned.
        hop_length: STFT hop in samples (paper: 300, i.e. 80 frames/s at 24 kHz).
        win_length: Hann window length (defaults to ``n_fft``).
        fmin, fmax: band (Hz) inside which the EIV is computed; outside it the EIV
            channels are zero. ``fmax`` should not exceed the spatial-aliasing
            frequency of the array that produced the FOA signal.
        n_hop_frames: ``T_r``; the covariance is averaged over ``2 T_r + 1`` frames.
        coherence_threshold: ``beta_DRR``, minimum ``sigma_1 / sigma_2`` of a single-source bin.
        snr_threshold: ``alpha_SNR``, minimum ratio of the running magnitude to the noise floor.
        use_magnitude_test, use_coherence_test: disable either test to keep every bin.
        rms_frames: frames of the causal running RMS used by the magnitude test.
        noise_floor_init_frames: frames assumed to contain noise only when initialising the floor.
        noise_floor_alpha: relative step of the noise-floor tracker (``alpha``).
        noise_floor_slow_scale: slow-rise scale applied after ``noise_floor_hold_frames``
            consecutive frames above the floor (a sustained sound is not noise).
        noise_floor_min: absolute floor of the tracked noise level.
        log_eps: added to ``|X|^2`` before the logarithm.
    """

    def __init__(
        self,
        sample_rate: float = 24000.0,
        n_fft: int = 512,
        hop_length: int = 300,
        win_length: Optional[int] = None,
        fmin: float = 50.0,
        fmax: float = 9000.0,
        n_hop_frames: int = 3,
        coherence_threshold: float = 5.0,
        snr_threshold: float = 1.5,
        use_magnitude_test: bool = True,
        use_coherence_test: bool = True,
        rms_frames: int = 3,
        noise_floor_init_frames: int = 5,
        noise_floor_alpha: float = 0.02,
        noise_floor_slow_scale: float = 0.1,
        noise_floor_hold_frames: int = 3,
        noise_floor_min: float = 1e-6,
        log_eps: float = 1e-10,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.n_hop_frames = int(n_hop_frames)
        self.coherence_threshold = float(coherence_threshold)
        self.snr_threshold = float(snr_threshold)
        self.use_magnitude_test = bool(use_magnitude_test)
        self.use_coherence_test = bool(use_coherence_test)
        self.rms_frames = int(rms_frames)
        self.noise_floor_init_frames = int(noise_floor_init_frames)
        self.noise_floor_alpha = float(noise_floor_alpha)
        self.noise_floor_slow_scale = float(noise_floor_slow_scale)
        self.noise_floor_hold_frames = int(noise_floor_hold_frames)
        self.noise_floor_min = float(noise_floor_min)
        self.log_eps = float(log_eps)

    # --------------------------------------------------------------- presets
    @classmethod
    def for_sustained_sources(cls, fmax: float, fmin: float = 150.0, **kwargs) -> "FoaSalsa":
        """Settings for continuous sources such as drone propellers, engines or fans.

        The paper's noise-floor tracker was designed for event-like sounds with
        onsets and pauses: after 3 frames above the floor it keeps rising by 0.2 %
        per frame (17 % per second), and every dip of the signal below the floor
        re-arms three fast 2 % steps. A steady sound is therefore absorbed into the
        "background" within a couple of seconds and its bins fail the magnitude test.

        This preset keeps the tracker but makes it behave like a minimum
        statistic: no fast re-arming after dips (``noise_floor_hold_frames = 0``) and
        a rise of 0.004 % per frame (``noise_floor_slow_scale = 0.002``), so the floor
        settles at the quietest moments of the sound and only slowly follows
        changes of the background. A perfectly steady tone would take about
        90 s to be absorbed. The 2 % per frame fall is unchanged. ``fmin`` defaults to
        150 Hz to stay above wind noise; ``fmax`` must be chosen for the array
        (below its spatial-aliasing frequency). Any other constructor argument can
        be overridden through ``kwargs``.
        """
        kwargs.setdefault("noise_floor_hold_frames", 0)
        kwargs.setdefault("noise_floor_slow_scale", 0.002)
        return cls(fmin=fmin, fmax=fmax, **kwargs)

    # --------------------------------------------------------------- helpers
    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frequencies(self) -> torch.Tensor:
        return torch.arange(self.n_bins, dtype=torch.float64) * self.sample_rate / self.n_fft

    @property
    def band(self) -> tuple[int, int]:
        """``(lower_bin, upper_bin)``: the EIV is computed for bins ``lower_bin <= k < upper_bin``."""
        lower = max(1, int(math.floor(self.fmin * self.n_fft / self.sample_rate)))
        upper = min(self.n_bins, int(math.floor(self.fmax * self.n_fft / self.sample_rate)))
        if upper <= lower:
            raise ValueError("fmax must be higher than fmin (and above one bin width)")
        return lower, upper

    def stft(self, foa: torch.Tensor) -> torch.Tensor:
        """Complex STFT ``[N, 4, F, T]`` of an FOA batch ``[N, 4, T]``."""
        if foa.ndim != 3 or foa.shape[1] != 4:
            raise ValueError(f"foa must have shape [N, 4, T], got {tuple(foa.shape)}")
        n, c, t = foa.shape
        window = torch.hann_window(self.win_length, periodic=True, dtype=foa.dtype, device=foa.device)
        spec = torch.stft(
            foa.reshape(n * c, t),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        return spec.reshape(n, c, spec.shape[-2], spec.shape[-1])

    def _magnitude_test(self, mag_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Adaptive noise-floor tracking on ``|X_W|`` ``[N, Fb, T]``.

        Returns ``(mask, noise_floor)`` both ``[N, Fb, T]``.
        """
        n, fb, t = mag_w.shape
        # causal running RMS over ``rms_frames`` frames (current + previous)
        power = mag_w.pow(2).reshape(-1, 1, t)
        power = F.pad(power, (self.rms_frames - 1, 0), mode="replicate")
        rms = F.avg_pool1d(power, kernel_size=self.rms_frames, stride=1).sqrt().reshape(n, fb, t)

        floor = 0.5 * rms[..., : self.noise_floor_init_frames].mean(dim=-1)
        countdown = torch.full((n, fb), self.noise_floor_hold_frames, dtype=torch.long, device=mag_w.device)
        up = 1.0 + self.noise_floor_alpha
        up_slow = 1.0 + self.noise_floor_slow_scale * self.noise_floor_alpha
        down = 1.0 - self.noise_floor_alpha
        hold = torch.full_like(countdown, self.noise_floor_hold_frames)

        masks = []
        floors = []
        for i in range(t):
            x = rms[..., i]
            above = x > floor
            countdown = torch.where(above, countdown - 1, hold)
            factor = torch.where(above, torch.where(countdown < 0, up_slow, up), down)
            floor = torch.clamp(floor * factor, min=self.noise_floor_min)
            masks.append(x > self.snr_threshold * floor)
            floors.append(floor)
        return torch.stack(masks, dim=-1), torch.stack(floors, dim=-1)

    # --------------------------------------------------------------- compute
    def analyze(self, foa: torch.Tensor) -> SalsaResult:
        """Computes the SALSA features together with all intermediate quantities."""
        spec = self.stft(foa)  # [N, 4, F, T]
        n, _, n_bins, n_frames = spec.shape
        real_dtype = spec.real.dtype
        device = spec.device
        linspec = torch.log(spec.real.pow(2) + spec.imag.pow(2) + self.log_eps)

        lo, hi = self.band
        xb = spec[:, :, lo:hi]  # [N, 4, Fb, T]

        # spatial covariance averaged over 2 T_r + 1 frames
        outer = xb.unsqueeze(1) * xb.unsqueeze(2).conj()  # [N, 4(i), 4(j), Fb, T] = X_i X_j^*
        cov = _moving_average(outer, self.n_hop_frames).permute(0, 3, 4, 1, 2)  # [N, Fb, T, 4, 4]
        cov = 0.5 * (cov + cov.mH)
        evals, evecs = torch.linalg.eigh(cov)  # ascending eigenvalues
        evals = evals.flip(-1)  # descending
        tiny = torch.finfo(real_dtype).tiny
        drr = evals[..., 0] / torch.clamp(evals[..., 1], min=tiny)

        # eigenvector-based intensity vector
        u = evecs[..., :, -1]  # principal eigenvector [N, Fb, T, 4]
        u0 = u[..., :1]
        safe_u0 = torch.where(u0.abs() > tiny, u0, torch.ones_like(u0))
        v = torch.real(u[..., 1:] / safe_u0)
        v = torch.where(u0.abs() > tiny, v, torch.zeros_like(v))
        eiv_band = v / torch.clamp(v.norm(dim=-1, keepdim=True), min=tiny)  # [N, Fb, T, 3]

        # single-source bin selection
        mag_mask, noise_floor = self._magnitude_test(xb[:, 0].abs())
        coh_mask = drr > self.coherence_threshold
        keep = torch.ones_like(coh_mask)
        if self.use_magnitude_test:
            keep = keep & mag_mask
        if self.use_coherence_test:
            keep = keep & coh_mask

        def embed(x: torch.Tensor, channels_last: bool = False) -> torch.Tensor:
            """Place a band tensor into a zero tensor spanning all bins."""
            if channels_last:  # [N, Fb, T, C] -> [N, C, F, T]
                full = torch.zeros(n, x.shape[-1], n_bins, n_frames, dtype=x.dtype, device=device)
                full[:, :, lo:hi] = x.permute(0, 3, 1, 2)
            else:  # [N, Fb, T] -> [N, F, T]
                full = torch.zeros(n, n_bins, n_frames, dtype=x.dtype, device=device)
                full[:, lo:hi] = x
            return full

        eiv_raw = embed(eiv_band, channels_last=True)
        eiv = embed(eiv_band * keep.unsqueeze(-1), channels_last=True)
        band_mask = torch.zeros(n_bins, dtype=torch.bool, device=device)
        band_mask[lo:hi] = True
        features = torch.cat([linspec, eiv.to(linspec.dtype)], dim=1)

        return SalsaResult(
            features=features,
            linspec=linspec,
            eiv=eiv,
            eiv_raw=eiv_raw,
            stft=spec,
            eigenvalues=embed(evals, channels_last=True).permute(0, 2, 3, 1),
            drr=embed(drr),
            noise_floor=embed(noise_floor),
            magnitude_mask=embed(mag_mask),
            coherence_mask=embed(coh_mask),
            band_mask=band_mask,
            mask=embed(keep),
            frequencies=self.frequencies.to(device),
            times=torch.arange(n_frames, dtype=torch.float64, device=device) * self.hop_length / self.sample_rate,
        )

    def compute(self, foa: torch.Tensor) -> torch.Tensor:
        """Computes SALSA features from a batch of FOA-encoded audio

        Params:
            foa - [N, 4, T] FOA batch
        Returns:
            features - a [N, 7, F, T] tensor of features
        """
        return self.analyze(foa).features
