"""Synthetic examples illustrating the SALSA features for drone (multirotor) sounds on planar arrays.

Run with::

    python -m foasalsa.examples <output_dir>

Every example builds a scene from plane waves hitting a planar (x-y) microphone
array, converts it to FOA with :class:`FoaConverter`, computes SALSA features with
:class:`FoaSalsa` and saves one PNG figure. The sound source in most scenes is a
synthetic quadcopter: four motors at slightly different, slowly wandering speeds,
each producing harmonics of its blade-passing frequency, plus a turbulence floor.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

import numpy as np
import torch

from .converter import FoaConverter
from .geometry import azimuth_elevation, direction_vector, steering_vectors, wrap_degrees
from .salsa import FoaSalsa, SalsaResult
from .synth import (
    air_absorption,
    band_noise,
    centred_circular_array,
    circular_array,
    diffuse_noise,
    drone_source,
    flyby,
    plane_wave,
    source_scene,
    wind_noise,
)

FS = 24000.0
C_SOUND = 343.0

#: a compact ring for reference, a plain 6 cm ring, and the same ring with a centre microphone (used for the drone scenes)
SMALL_ARRAY = circular_array(6, 0.03)
RING_ARRAY = circular_array(8, 0.06)
DRONE_ARRAY = centred_circular_array(8, 0.06)
DRONE_FMAX = 3000.0  # below the ~3.7 kHz aliasing frequency of the 6 cm ring

# ------------------------------------------------------------------ styling
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
MASKED = "#d9d8d4"
SEQ_STEPS = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_STEPS = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a7a6", "#e34948", "#7f1d1d"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.edgecolor": TEXT_2,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "lines.linewidth": 1.6,
        }
    )
    seq = LinearSegmentedColormap.from_list("salsa_seq", SEQ_STEPS)
    div = LinearSegmentedColormap.from_list("salsa_div", DIV_STEPS)
    cyc = matplotlib.colormaps["twilight"].copy()
    for cm in (seq, div, cyc):
        cm.set_bad(MASKED)
    return plt, {"seq": seq, "div": div, "cyc": cyc}


def _imshow(ax, data, res: SalsaResult, cmap, vmin=None, vmax=None, mask=None, fmax_khz=None):
    """Draws a [F, T] map with time on x (s) and frequency on y (kHz); masked bins in gray."""
    arr = np.asarray(data.detach().cpu(), dtype=float)
    if mask is not None:
        arr = np.ma.masked_where(~np.asarray(mask.detach().cpu(), dtype=bool), arr)
    t = res.times.cpu().numpy()
    f = res.frequencies.cpu().numpy() / 1000.0
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[t[0], t[-1], f[0], f[-1]],
        interpolation="nearest",
    )
    if fmax_khz is not None:
        ax.set_ylim(0, fmax_khz)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    return im


def _spec_range(linspec: torch.Tensor) -> tuple[float, float]:
    """Display range of a log-spectrogram: from just below the noise floor to the maximum."""
    lo = torch.quantile(linspec.flatten()[:: max(1, linspec.numel() // 200000)], 0.2).item()
    return lo, linspec.max().item()


def _eiv_azimuth(res: SalsaResult) -> torch.Tensor:
    """Azimuth (degrees) of the *unmasked* EIV at every bin, [N, F, T]."""
    az, _ = azimuth_elevation(res.eiv_raw.permute(0, 2, 3, 1))
    return az


def _active_rms(p: torch.Tensor) -> torch.Tensor:
    active = p[0].abs() > 0.05 * p[0].abs().max()
    return p[0][active].pow(2).mean().sqrt() if active.any() else p[0].pow(2).mean().sqrt()


def _add_noise(
    p: torch.Tensor,
    mics: torch.Tensor,
    snr_sensor_db: Optional[float],
    snr_diffuse_db: Optional[float],
    gen,
    snr_wind_db: Optional[float] = None,
) -> torch.Tensor:
    """Adds sensor, diffuse and (per-microphone, uncorrelated) wind noise at SNRs relative to the active signal RMS."""
    rms = _active_rms(p)
    out = p.clone()
    if snr_sensor_db is not None:
        out += rms * 10 ** (-snr_sensor_db / 20) * torch.randn(p.shape, dtype=p.dtype, generator=gen)
    if snr_diffuse_db is not None:
        out += rms * 10 ** (-snr_diffuse_db / 20) * diffuse_noise(p.shape[-1], mics, FS, C_SOUND, generator=gen)
    if snr_wind_db is not None:
        for m in range(p.shape[0]):
            out[m] += rms * 10 ** (-snr_wind_db / 20) * wind_noise(p.shape[-1], FS, generator=gen)
    return out


def _to_foa(p: torch.Tensor, mics: torch.Tensor) -> torch.Tensor:
    return FoaConverter(sample_rate=FS, speed_of_sound=C_SOUND).convert(p[None].float(), mics)


def _aliasing_frequency(mics: torch.Tensor) -> float:
    d = torch.cdist(mics, mics)
    spacing = d[d > 1e-9].min().item()
    return C_SOUND / (2.0 * spacing)


def _drone_at(n: int, distance_m: float, gen, rpm: float = 12000.0, **kwargs) -> torch.Tensor:
    """A drone sound as heard after ``distance_m`` of air (absorption only; level is normalised by the caller)."""
    return air_absorption(drone_source(n, FS, rpm=rpm, generator=gen, **kwargs), distance_m, FS)


def _fade(n: int, start: float, stop: float, ramp: float = 0.25) -> torch.Tensor:
    """Envelope [T] that is 1 between ``start`` and ``stop`` seconds with raised-cosine ramps."""
    t = torch.arange(n, dtype=torch.float64) / FS
    up = 0.5 - 0.5 * torch.cos(math.pi * ((t - start) / ramp).clamp(0, 1))
    down = 0.5 + 0.5 * torch.cos(math.pi * ((t - (stop - ramp)) / ramp).clamp(0, 1))
    return up * down


# ------------------------------------------------------------- example 1
def example_converter_response(out_dir: str) -> str:
    """Encoder accuracy and its physical limits for the two planar arrays used in the examples."""
    plt, cmaps = _mpl()
    arrays = {"6 mics, r = 3 cm": SMALL_ARRAY, "8-mic ring, r = 6 cm": RING_ARRAY, "8 + centre mic, r = 6 cm (drone array)": DRONE_ARRAY}
    conv = FoaConverter(sample_rate=FS, speed_of_sound=C_SOUND)
    azimuths = torch.arange(0.0, 360.0, 5.0)
    dirs = direction_vector(azimuths, 0.0)
    freqs = conv.frequencies

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    ax_geo, ax_gain, ax_err, ax_noise = axes.flat
    for i, (name, mics) in enumerate(arrays.items()):
        col = SERIES[i]
        ax_geo.scatter(mics[:, 0] * 100, mics[:, 1] * 100, s=[70, 28, 12][i], color=col, label=name, zorder=3 + i)
        E = conv.encoding_matrices(mics)  # [K, 4, C]
        H = steering_vectors(freqs, dirs, mics, C_SOUND)  # [K, C, D]
        resp = torch.einsum("kac,kcd->kad", E, H)  # [K, 4, D]
        w_gain = resp[:, 0].real.median(dim=-1).values  # signed: a sign flip reverses the EIV
        h_gain = torch.sqrt(resp[:, 1].real ** 2 + resp[:, 2].real ** 2).median(dim=-1).values
        est_az = torch.rad2deg(torch.atan2(resp[:, 2].real, resp[:, 1].real))
        az_err = wrap_degrees(est_az - azimuths[None]).abs().median(dim=-1).values
        noise_gain = 20 * torch.log10(E[:, 1].norm(dim=-1) / E[:, 0].norm(dim=-1))
        f = freqs[1:].numpy()
        ax_gain.plot(f, w_gain[1:], color=col, label=f"{name}: W")
        ax_gain.plot(f, h_gain[1:], color=col, ls="--", label=f"{name}: |(X, Y)|")
        ax_err.plot(f, az_err[1:], color=col, label=name)
        ax_noise.plot(f, noise_gain[1:], color=col, label=name)
        if i != 1:  # the plain ring shares the drone array's aliasing frequency
            fa = _aliasing_frequency(mics)
            for ax in (ax_gain, ax_err, ax_noise):
                ax.axvline(fa, color=col, ls=":", lw=1)
            ax_err.text(fa, 100, f" c/2d = {fa/1000:.1f} kHz", color=col, fontsize=7, rotation=90, va="top")
    ax_gain.axhline(0, color=TEXT_2, lw=0.8)

    ax_geo.set_aspect("equal")
    ax_geo.set_xlabel("x (cm)")
    ax_geo.set_ylabel("y (cm)")
    ax_geo.set_title("(a) planar arrays (z = 0)")
    ax_geo.legend(loc="upper right", fontsize=7)
    ax_geo.grid(True, color=GRID)

    ax_gain.set_xscale("log")
    ax_gain.set_ylim(-0.6, 1.3)
    ax_gain.set_xlabel("frequency (Hz)")
    ax_gain.set_ylabel("encoded gain (ideal = 1)")
    ax_gain.set_title("(b) W (signed) and horizontal (X, Y) gain of an encoded plane wave")
    ax_gain.legend(fontsize=6, ncol=2, loc="lower left")

    ax_err.set_xscale("log")
    ax_err.set_ylim(0, 100)
    ax_err.set_xlabel("frequency (Hz)")
    ax_err.set_ylabel("median azimuth error (deg)")
    ax_err.set_title("(c) azimuth error of atan2(Y, X) - spatial aliasing")
    ax_err.legend(loc="upper left", fontsize=7)

    ax_noise.set_xscale("log")
    ax_noise.set_xlabel("frequency (Hz)")
    ax_noise.set_ylabel("||E_X|| / ||E_W||  (dB)")
    ax_noise.set_title(f"(d) white-noise gain of X relative to W (cap {conv.max_gain_db:.0f} dB)")
    ax_noise.legend(loc="upper right", fontsize=7)
    for ax in (ax_gain, ax_err, ax_noise):
        ax.grid(True, which="both", color=GRID)
        ax.set_xlim(40, FS / 2)

    fig.suptitle(
        "FoaConverter: least-squares FOA encoder for open planar arrays. Below ~c/2d the X/Y channels follow the ideal cos/sin\n"
        "pattern; above it directions alias. The W channel of a plain ring crosses zero at kr = 2.4 (2.2 kHz for r = 6 cm), which\n"
        "reverses every EIV above it; a centre microphone keeps W positive, so the 6 cm ring can be used up to 3 kHz for drone harmonics.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, "01_converter_response.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------- example 2
def example_drone_hover(out_dir: str) -> str:
    """All seven SALSA channels for a hovering drone; elevation is invisible to a planar array."""
    plt, cmaps = _mpl()
    mics = DRONE_ARRAY
    n = int(3.0 * FS)
    az_true = 60.0
    sig = _drone_at(n, 40.0, torch.Generator().manual_seed(2))
    salsa = FoaSalsa.for_sustained_sources(fmax=DRONE_FMAX, sample_rate=FS)
    results = []
    for el in (0.0, 30.0):
        p = plane_wave(sig, direction_vector(az_true, el), mics, FS, C_SOUND)
        p = _add_noise(p, mics, snr_sensor_db=40.0, snr_diffuse_db=25.0, gen=torch.Generator().manual_seed(3))
        results.append(salsa.analyze(_to_foa(p, mics)))
    res0, res30 = results

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(7, 2, width_ratios=[1, 1], hspace=0.55, wspace=0.32)
    names = ["LinSpec W", "LinSpec X", "LinSpec Y", "LinSpec Z", "EIV x", "EIV y", "EIV z"]
    feats = res0.features[0]
    lmin, lmax = _spec_range(feats[:3])
    for i in range(7):
        ax = fig.add_subplot(gs[i, 0])
        if i < 4:
            im = _imshow(ax, feats[i], res0, cmaps["seq"], vmin=lmin, vmax=lmax, fmax_khz=4)
            cb_label = "log |X|²"
        else:
            im = _imshow(ax, feats[i], res0, cmaps["div"], vmin=-1, vmax=1, mask=res0.mask[0], fmax_khz=4)
            cb_label = "EIV"
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label(cb_label, fontsize=7)
        ax.set_title(f"{names[i]}  (drone at azimuth {az_true:.0f}°, elevation 0°)", loc="left")
        if i < 6:
            ax.set_xlabel("")

    for i in range(4, 7):
        ax = fig.add_subplot(gs[i, 1])
        im = _imshow(ax, res30.features[0, i], res30, cmaps["div"], vmin=-1, vmax=1, mask=res30.mask[0], fmax_khz=4)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label("EIV", fontsize=7)
        ax.set_title(f"{names[i]}  (same drone at elevation 30°)", loc="left")
        ax.set_ylabel("")
        if i < 6:
            ax.set_xlabel("")

    ax_h = fig.add_subplot(gs[0:3, 1])
    bins = np.arange(-180, 181, 2)
    for res, label, col in ((res0, "elevation 0°", SERIES[0]), (res30, "elevation 30°", SERIES[1])):
        az = _eiv_azimuth(res)[0][res.mask[0]].cpu().numpy()
        ax_h.hist(az, bins=bins, histtype="step", color=col, label=f"{label}: {az.size} bins", lw=1.6)
    ax_h.axvline(az_true, color=TEXT_2, ls=":", lw=1)
    ax_h.set_xlabel("EIV azimuth atan2(y, x) (deg)")
    ax_h.set_ylabel("single-source TF bins")
    ax_h.set_title("EIV azimuth of the selected bins (dotted: true azimuth)", loc="left")
    ax_h.legend(loc="upper left")
    ax_h.set_xlim(-180, 180)

    ax_t = fig.add_subplot(gs[3, 1])
    ax_t.axis("off")
    ax_t.text(
        0,
        1,
        "The spectrogram shows the blade-passing harmonics of four motors as\n"
        "clusters of close, slowly drifting lines. Gray = bins that fail a test or lie\n"
        f"outside [{salsa.fmin:.0f} Hz, {salsa.fmax:.0f} Hz]: their EIV is zero. Every kept bin holds\n"
        "the unit vector (cos az, sin az, 0). A planar array has no Z channel, so\n"
        "LinSpec Z is constant and EIV z is 0: the drone at 30° elevation gives the\n"
        "same EIV as one at the horizon. Elevation needs a 3-D array.",
        va="top",
        fontsize=8,
        color=TEXT,
    )
    fig.suptitle(
        "SALSA features of a hovering quadcopter (40 m) on the 8 + centre mic, r = 6 cm planar array, sustained-source preset (fmax = 3 kHz)",
        fontsize=10,
        y=0.995,
    )
    path = os.path.join(out_dir, "02_drone_hover.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------- example 3
def example_two_drones(out_dir: str) -> str:
    """Two drones with different motor speeds: TF-wise directional cues and the coherence test."""
    plt, cmaps = _mpl()
    gen = torch.Generator().manual_seed(5)
    mics = DRONE_ARRAY
    n = int(4.0 * FS)
    az_a, az_b = 30.0, -100.0
    sig_a = _drone_at(n, 60.0, gen, rpm=12000.0) * _fade(n, 0.0, 2.5)
    sig_b = 0.8 * _drone_at(n, 60.0, gen, rpm=15500.0) * _fade(n, 1.5, 4.0)
    p = source_scene(mics, FS, [(sig_a, az_a, 0.0), (sig_b, az_b, 0.0)], C_SOUND)
    p = _add_noise(p, mics, snr_sensor_db=40.0, snr_diffuse_db=25.0, gen=gen)
    foa = _to_foa(p, mics)
    common = dict(fmax=DRONE_FMAX, sample_rate=FS)
    res_off = FoaSalsa.for_sustained_sources(use_coherence_test=False, **common).analyze(foa)
    res_on = FoaSalsa.for_sustained_sources(**common).analyze(foa)
    az_map = _eiv_azimuth(res_off)[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (ax_spec, ax_hist), (ax_raw, ax_sel) = axes
    lmin, lmax = _spec_range(res_off.linspec[0, 0])
    im = _imshow(ax_spec, res_off.linspec[0, 0], res_off, cmaps["seq"], vmin=lmin, vmax=lmax, fmax_khz=3.5)
    fig.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.04).set_label("log |W|²", fontsize=7)
    ax_spec.set_title("(a) LinSpec W (bars: blue = drone A active, orange = drone B active)", loc="left")
    ax_spec.axvspan(0.0, 2.5, ymin=0.97, color=SERIES[0], lw=0)
    ax_spec.axvspan(1.5, 4.0, ymin=0.94, ymax=0.97, color=SERIES[1], lw=0)

    im = _imshow(ax_raw, az_map, res_off, cmaps["cyc"], vmin=-180, vmax=180, mask=res_off.magnitude_mask[0], fmax_khz=3.5)
    cb = fig.colorbar(im, ax=ax_raw, pad=0.01, fraction=0.04, ticks=[-180, -100, 0, 30, 180])
    cb.set_label("EIV azimuth (deg)", fontsize=7)
    ax_raw.set_title("(b) EIV azimuth per TF bin, magnitude test only", loc="left")

    im = _imshow(ax_sel, az_map, res_on, cmaps["cyc"], vmin=-180, vmax=180, mask=res_on.mask[0], fmax_khz=3.5)
    cb = fig.colorbar(im, ax=ax_sel, pad=0.01, fraction=0.04, ticks=[-180, -100, 0, 30, 180])
    cb.set_label("EIV azimuth (deg)", fontsize=7)
    ax_sel.set_title("(c) EIV azimuth per TF bin, magnitude + coherence tests", loc="left")

    overlap = (res_off.times >= 1.7) & (res_off.times <= 2.3)
    bins = np.arange(-180, 181, 3)
    for res, mask, label, col in (
        (res_off, res_off.magnitude_mask, "magnitude test only", SERIES[2]),
        (res_on, res_on.mask, "magnitude + coherence tests", SERIES[0]),
    ):
        sel = mask[0] & overlap[None, :] & res.band_mask[:, None]
        az = az_map[sel].cpu().numpy()
        ax_hist.hist(az, bins=bins, histtype="stepfilled", alpha=0.35, color=col, lw=0)
        ax_hist.hist(az, bins=bins, histtype="step", color=col, label=f"{label} ({az.size} bins)", lw=1.5)
    for az_s, lab in ((az_a, "A"), (az_b, "B")):
        ax_hist.axvline(az_s, color=TEXT_2, ls=":", lw=1)
        ax_hist.text(az_s, ax_hist.get_ylim()[1] * 0.98, f" {lab} ({az_s:.0f}°)", va="top", fontsize=8, color=TEXT_2)
    ax_hist.set_xlim(-180, 180)
    ax_hist.set_xlabel("EIV azimuth (deg)")
    ax_hist.set_ylabel("TF bins in 1.7-2.3 s (both drones active)")
    ax_hist.set_title("(d) azimuth histogram while the drones overlap", loc="left")
    ax_hist.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))

    fig.suptitle(
        f"Two quadcopters: A (12 000 rpm) at {az_a:.0f}° until 2.5 s, B (15 500 rpm) at {az_b:.0f}° from 1.5 s, both 60 m away.\n"
        "Their harmonic combs interleave, so most TF bins belong to one drone and the EIV points at it. Where a harmonic of A\n"
        "lands on one of B the principal eigenvector mixes them; the coherence test (σ1/σ2 > 5) discards those bins.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out_dir, "03_two_drones.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------- example 4
def example_bin_selection(out_dir: str) -> str:
    """A distant drone in wind and diffuse noise with a wall reflection: the two tests and the DRR threshold."""
    plt, cmaps = _mpl()
    gen = torch.Generator().manual_seed(7)
    mics = DRONE_ARRAY
    n = int(4.0 * FS)
    az_true, az_refl = -45.0, 120.0
    sig = _drone_at(n, 150.0, gen)
    delay = int(0.006 * FS)
    refl = torch.zeros_like(sig)
    refl[delay:] = 0.5 * sig[:-delay]
    p = source_scene(mics, FS, [(sig, az_true, 0.0), (refl, az_refl, 0.0)], C_SOUND)
    p = _add_noise(p, mics, snr_sensor_db=35.0, snr_diffuse_db=10.0, gen=gen, snr_wind_db=0.0)
    foa = _to_foa(p, mics)
    base = dict(fmax=DRONE_FMAX, sample_rate=FS)
    res = FoaSalsa.for_sustained_sources(**base).analyze(foa)
    err = wrap_degrees(_eiv_azimuth(res)[0] - az_true).abs()
    band = res.band_mask[:, None].expand_as(res.mask[0])

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    lmin, lmax = _spec_range(res.linspec[0, 0])
    im = _imshow(axes[0, 0], res.linspec[0, 0], res, cmaps["seq"], vmin=lmin, vmax=lmax, fmax_khz=3.5)
    fig.colorbar(im, ax=axes[0, 0], pad=0.01, fraction=0.04).set_label("log |W|²", fontsize=7)
    axes[0, 0].set_title("(a) LinSpec W: drone + reflection + wind + diffuse noise", loc="left")

    masks = [
        ("(b) magnitude test", res.magnitude_mask[0]),
        ("(c) coherence test (σ1/σ2 > 5)", res.coherence_mask[0]),
        ("(d) both tests -> EIV kept", res.mask[0]),
    ]
    for ax, (title, m) in zip(axes[0, 1:], masks):
        frac = (m & band).sum().item() / band.sum().item()
        im = _imshow(ax, (m & band).float(), res, cmaps["seq"], vmin=0, vmax=1.4, fmax_khz=3.5)
        ax.set_title(f"{title}: {100 * frac:.0f}% of band bins", loc="left")
        ax.set_ylabel("")

    im = _imshow(axes[1, 0], err, res, cmaps["seq"], vmin=0, vmax=90, mask=band & res.magnitude_mask[0], fmax_khz=3.5)
    fig.colorbar(im, ax=axes[1, 0], pad=0.01, fraction=0.04).set_label("|azimuth error| (deg)", fontsize=7)
    axes[1, 0].set_title("(e) EIV azimuth error, bins passing the magnitude test", loc="left")

    ax = axes[1, 1]
    sets = [
        ("all band bins", band, SERIES[3]),
        ("magnitude test", band & res.magnitude_mask[0], SERIES[2]),
        ("coherence test", band & res.coherence_mask[0], SERIES[1]),
        ("both tests", band & res.mask[0], SERIES[0]),
    ]
    for label, m, col in sets:
        e = np.sort(err[m].cpu().numpy())
        ax.plot(e, np.linspace(0, 1, e.size), color=col, label=f"{label} (median {np.median(e):.1f}°)")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)
    ax.set_xlabel("|azimuth error| (deg)")
    ax.set_ylabel("fraction of bins (CDF)")
    ax.set_title("(f) error distribution of the EIV per bin set", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, color=GRID)

    thresholds = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0]
    medians, fracs = [], []
    for beta in thresholds:
        r = FoaSalsa.for_sustained_sources(coherence_threshold=beta, **base).analyze(foa)
        m = r.mask[0] & band
        medians.append(err[m].median().item() if m.any() else float("nan"))
        fracs.append(m.sum().item() / (band & r.magnitude_mask[0]).sum().item())
    for ax, y, ylabel, title in (
        (axes[1, 2], medians, "median |azimuth error| (deg)", "(g) DRR threshold β vs. EIV error"),
        (axes[1, 3], fracs, "kept fraction of magnitude-test bins", "(h) DRR threshold β vs. kept bins"),
    ):
        ax.plot(thresholds, y, color=SERIES[0], marker="o", ms=4)
        ax.axvline(5.0, color=TEXT_2, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("coherence threshold β_DRR (paper: 5)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(True, which="both", color=GRID)
    fig.suptitle(
        f"Single-source bin selection: a drone 150 m away at {az_true:.0f}° (air absorption dulls its high harmonics), a wall reflection\n"
        f"(-6 dB, 6 ms, from {az_refl:.0f}°), wind noise at each microphone (0 dB SNR, mostly below 300 Hz) and diffuse noise (10 dB SNR).\n"
        "The magnitude test removes noise-dominated bins; the coherence test removes bins where reflection or noise break the rank-1 structure.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = os.path.join(out_dir, "04_bin_selection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------- example 5
def example_sustained_source(out_dir: str) -> str:
    """A long hover: the paper's noise-floor tracker versus the sustained-source preset."""
    plt, cmaps = _mpl()
    gen = torch.Generator().manual_seed(9)
    mics = DRONE_ARRAY
    n = int(8.0 * FS)
    az_true = 45.0
    sig = _drone_at(n, 40.0, gen, wander=0.01)
    p = plane_wave(sig, direction_vector(az_true, 0.0), mics, FS, C_SOUND)
    p = _add_noise(p, mics, snr_sensor_db=40.0, snr_diffuse_db=20.0, gen=gen)
    foa = _to_foa(p, mics)
    res_default = FoaSalsa(sample_rate=FS, fmin=150.0, fmax=DRONE_FMAX).analyze(foa)
    res_preset = FoaSalsa.for_sustained_sources(fmax=DRONE_FMAX, sample_rate=FS).analyze(foa)
    band = res_default.band_mask
    times = res_default.times.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    lmin, lmax = _spec_range(res_default.linspec[0, 0])
    im = _imshow(axes[0, 0], res_default.linspec[0, 0], res_default, cmaps["seq"], vmin=lmin, vmax=lmax, fmax_khz=3.5)
    fig.colorbar(im, ax=axes[0, 0], pad=0.01, fraction=0.04).set_label("log |W|²", fontsize=7)
    axes[0, 0].set_title("(a) LinSpec W of an 8 s hover", loc="left")

    # strongest harmonic bin in the band
    power = res_default.stft[0, 0].abs().pow(2).mean(dim=-1)
    power[~band] = 0
    k = int(power.argmax().item())
    mag = res_default.stft[0, 0, k].abs().numpy()
    ax = axes[0, 1]
    ax.plot(times, mag, color=TEXT_2, lw=1, label="|W| at the bin")
    ax.plot(times, 1.5 * res_default.noise_floor[0, k].numpy(), color=SERIES[0], label="threshold, paper tracker")
    ax.plot(times, 1.5 * res_preset.noise_floor[0, k].numpy(), color=SERIES[1], label="threshold, sustained preset")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("magnitude")
    ax.set_title(f"(b) magnitude test at {res_default.frequencies[k]:.0f} Hz: |W| vs 1.5 × noise floor", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, which="both", color=GRID)

    ax = axes[0, 2]
    for res, label, col in ((res_default, "paper tracker", SERIES[0]), (res_preset, "sustained preset", SERIES[1])):
        frac = res.magnitude_mask[0][band].float().mean(dim=0).numpy()
        ax.plot(times, frac, color=col, label=label, lw=1.3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fraction of band bins")
    ax.set_ylim(0, 1)
    ax.set_title("(c) bins passing the magnitude test over time", loc="left")
    ax.legend(loc="center right")
    ax.grid(True, color=GRID)

    for ax, res, label in ((axes[1, 0], res_default, "paper tracker"), (axes[1, 1], res_preset, "sustained preset")):
        az = _eiv_azimuth(res)[0]
        im = _imshow(ax, az, res, cmaps["cyc"], vmin=-180, vmax=180, mask=res.mask[0], fmax_khz=3.5)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04, ticks=[-180, -90, 0, 45, 90, 180])
        cb.set_label("EIV azimuth (deg)", fontsize=7)
        kept = res.mask[0][band].float().mean().item()
        ax.set_title(f"({'de'[ax is axes[1, 1]]}) EIV kept with the {label}: {100 * kept:.0f}% of band bins", loc="left")

    ax = axes[1, 2]
    bins = np.arange(-180, 181, 2)
    for res, label, col in ((res_default, "paper tracker", SERIES[0]), (res_preset, "sustained preset", SERIES[1])):
        az = _eiv_azimuth(res)[0][res.mask[0]].numpy()
        ax.hist(az, bins=bins, histtype="step", color=col, lw=1.5, label=f"{label}: {az.size} bins")
    ax.axvline(az_true, color=TEXT_2, ls=":", lw=1)
    ax.set_xlim(-180, 180)
    ax.set_xlabel("EIV azimuth (deg)")
    ax.set_ylabel("kept TF bins")
    ax.set_title("(f) azimuth of the kept bins over the whole clip", loc="left")
    ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "A continuous sound defeats the paper's magnitude test. Its noise floor is designed for sounds with onsets and pauses: it\n"
        "rises 0.2 % per frame while a sound persists and re-arms fast 2 % steps after every dip, so a steady drone becomes 'background'\n"
        "within seconds. FoaSalsa.for_sustained_sources() removes the re-arming and slows the rise 1000x, so the floor settles at the sound's quietest moments.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = os.path.join(out_dir, "05_sustained_source.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------- example 6
def example_frequency_limits(out_dir: str) -> str:
    """EIV accuracy versus frequency: low-frequency noise gain and spatial aliasing."""
    plt, cmaps = _mpl()
    arrays = {"6 mics, r = 3 cm": SMALL_ARRAY, "8 + centre mic, r = 6 cm (drone array)": DRONE_ARRAY}
    n = int(2.0 * FS)
    az_true = 45.0
    gen = torch.Generator().manual_seed(11)
    sig = band_noise(n, FS, 30.0, FS / 2 - 100, generator=gen)
    full = FoaSalsa(sample_rate=FS, fmin=30.0, fmax=FS / 2, use_magnitude_test=False, use_coherence_test=False)
    curves = {}
    results = {}
    for name, mics in arrays.items():
        p = plane_wave(sig, direction_vector(az_true, 0.0), mics, FS, C_SOUND)
        p = _add_noise(p, mics, snr_sensor_db=30.0, snr_diffuse_db=None, gen=torch.Generator().manual_seed(12))
        res = full.analyze(_to_foa(p, mics))
        err = wrap_degrees(_eiv_azimuth(res)[0] - az_true).abs()
        curves[name] = err.median(dim=-1).values.cpu().numpy()
        results[name] = (mics, res, err)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    freqs = full.frequencies.numpy()
    for i, (name, med) in enumerate(curves.items()):
        ax.plot(freqs[1:], med[1:], color=SERIES[i], label=name)
        fa = _aliasing_frequency(arrays[name])
        ax.axvline(fa, color=SERIES[i], ls=":", lw=1)
        ax.text(fa, 88, f" c/2d = {fa/1000:.1f} kHz", color=SERIES[i], fontsize=7, rotation=90, va="top")
    ax.set_xscale("log")
    ax.set_xlim(40, FS / 2)
    ax.set_ylim(0, 90)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("median |azimuth error| over frames (deg)")
    ax.set_title("(a) EIV error vs frequency (white source, sensor SNR 30 dB)", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, which="both", color=GRID)

    name = "8 + centre mic, r = 6 cm (drone array)"
    mics, res, err = results[name]
    fa = _aliasing_frequency(mics)
    im = _imshow(axes[1], err, res, cmaps["seq"], vmin=0, vmax=90, mask=res.band_mask[:, None].expand_as(err), fmax_khz=12)
    fig.colorbar(im, ax=axes[1], pad=0.01, fraction=0.04).set_label("|azimuth error| (deg)", fontsize=7)
    axes[1].axhline(fa / 1000, color=SERIES[1], ls=":", lw=1)
    axes[1].set_title("(b) drone array: error map, fmax = Nyquist, tests off", loc="left")

    p = plane_wave(sig, direction_vector(az_true, 0.0), mics, FS, C_SOUND)
    p = _add_noise(p, mics, snr_sensor_db=30.0, snr_diffuse_db=None, gen=torch.Generator().manual_seed(12))
    limited = FoaSalsa(sample_rate=FS, fmin=50.0, fmax=DRONE_FMAX).analyze(_to_foa(p, mics))
    err_limited = wrap_degrees(_eiv_azimuth(limited)[0] - az_true).abs()
    im = _imshow(axes[2], err_limited, limited, cmaps["seq"], vmin=0, vmax=90, mask=limited.mask[0], fmax_khz=12)
    fig.colorbar(im, ax=axes[2], pad=0.01, fraction=0.04).set_label("|azimuth error| (deg)", fontsize=7)
    axes[2].set_title(f"(c) drone array: error map, fmax = {DRONE_FMAX / 1000:.0f} kHz (c/2d = {fa / 1000:.1f} kHz), tests on", loc="left")
    for a in axes[1:]:
        a.set_ylabel("")
    axes[1].set_ylabel("frequency (kHz)")
    fig.suptitle(
        "Frequency limits of the directional cue. The small ring is usable up to 5.7 kHz but its tiny phase differences at low\n"
        "frequencies are swamped by sensor noise; the 6 cm drone ring is cleaner below 300 Hz and aliases above 3.7 kHz.\n"
        "Drone harmonics carry most energy below 3 kHz, so fmax = 3 kHz keeps the useful band and zeroes the rest (gray in c).",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = os.path.join(out_dir, "06_frequency_limits.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------- example 7
def example_flyby(out_dir: str) -> str:
    """A fast, low fly-past with Doppler shift versus the covariance window T_r."""
    plt, cmaps = _mpl()
    mics = DRONE_ARRAY
    seconds = 6.0
    n = int(seconds * FS)
    gen = torch.Generator().manual_seed(21)
    speed, lateral, height = 30.0, 15.0, 5.0  # m/s, m, m
    t = torch.arange(n, dtype=torch.float64) / FS
    positions = torch.stack([-90.0 + speed * t, torch.full_like(t, lateral), torch.full_like(t, height)], dim=1)
    sig = _drone_at(n, 50.0, gen)
    p = flyby(sig, positions, mics, FS, C_SOUND, reference_distance=lateral)
    p = _add_noise(p, mics, snr_sensor_db=40.0, snr_diffuse_db=12.0, gen=gen)
    foa = _to_foa(p, mics)

    def apparent_azimuth(frame_times: np.ndarray) -> np.ndarray:
        """Azimuth of the position the drone had when the sound now arriving left it."""
        te = frame_times.copy()
        for _ in range(4):
            x = -90.0 + speed * te
            te = frame_times - np.sqrt(x**2 + lateral**2 + height**2) / C_SOUND
        return np.degrees(np.arctan2(lateral, -90.0 + speed * te))

    windows = [1, 3, 12]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    res0 = FoaSalsa.for_sustained_sources(fmax=DRONE_FMAX, sample_rate=FS).analyze(foa)
    lmin, lmax = _spec_range(res0.linspec[0, 0])
    im = _imshow(axes[0, 0], res0.linspec[0, 0], res0, cmaps["seq"], vmin=lmin, vmax=lmax, fmax_khz=3.5)
    fig.colorbar(im, ax=axes[0, 0], pad=0.01, fraction=0.04).set_label("log |W|²", fontsize=7)
    axes[0, 0].set_title("(a) LinSpec W: harmonics slide down as the drone passes (Doppler)", loc="left")

    strip_axes = [axes[0, 1], axes[0, 2], axes[1, 0]]
    az_bins = np.arange(-180, 181, 3)
    frames = res0.times.numpy()
    az_true_frames = apparent_azimuth(frames)
    for i, tr in enumerate(windows):
        res = FoaSalsa.for_sustained_sources(fmax=DRONE_FMAX, sample_rate=FS, n_hop_frames=tr).analyze(foa)
        az = _eiv_azimuth(res)[0]
        m = res.mask[0] & res.band_mask[:, None]
        tt = np.broadcast_to(frames[None, :], az.shape)[m.numpy()]
        aa = az[m].numpy()
        edges_t = np.concatenate([frames - (frames[1] - frames[0]) / 2, [frames[-1] + (frames[1] - frames[0]) / 2]])
        hist, _, _ = np.histogram2d(tt, aa, bins=[edges_t, az_bins])
        ax = strip_axes[i]
        im = ax.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            cmap=cmaps["seq"],
            extent=[edges_t[0], edges_t[-1], az_bins[0], az_bins[-1]],
            vmin=0,
            vmax=max(1.0, np.percentile(hist, 99.5)),
            interpolation="nearest",
        )
        ax.plot(frames, az_true_frames, color=SERIES[1], lw=1, ls="--", label="apparent azimuth")
        ax.set_title(f"({'bcd'[i]}) T_r = {tr}: covariance over {2 * tr + 1} frames = {(2 * tr + 1) * 300 / FS * 1000:.0f} ms", loc="left")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("EIV azimuth (deg)")
        ax.set_ylim(0, 180)
        if i == 0:
            ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04).set_label("kept bins per frame & 3°", fontsize=7)

        err = wrap_degrees(az - torch.as_tensor(az_true_frames)[None, :])
        spread = np.array(
            [np.percentile(np.abs(err[:, k][m[:, k]].numpy()), 75) if m[:, k].any() else np.nan for k in range(len(frames))]
        )
        frac = m.sum(dim=0).numpy() / res.band_mask.sum().item()
        axes[1, 1].plot(frames, spread, color=SERIES[i], label=f"T_r = {tr}", lw=1.3)
        axes[1, 2].plot(frames, frac, color=SERIES[i], label=f"T_r = {tr}", lw=1.3)
    axes[1, 1].set_title("(e) 75th percentile of |azimuth error| of kept bins per frame", loc="left")
    axes[1, 1].set_ylabel("deg")
    axes[1, 1].set_ylim(0, 45)
    axes[1, 2].set_title("(f) fraction of band bins that pass both tests", loc="left")
    axes[1, 2].set_ylabel("fraction")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].legend(loc="upper left")
    for ax in axes[1, 1:]:
        ax.set_xlabel("time (s)")
        ax.grid(True, color=GRID)
    closest = (90.0 / speed)
    for ax in list(axes[1, 1:]):
        ax.axvline(closest, color=TEXT_2, ls=":", lw=1)
    fig.suptitle(
        f"Fly-past at {speed:.0f} m/s ({speed * 3.6:.0f} km/h), {lateral:.0f} m to the side and {height:.0f} m up, from 90 m ahead to 90 m behind (closest at {closest:.0f} s).\n"
        "Near the closest point the azimuth sweeps at ~115°/s and the level peaks; far away it barely moves and the SNR is poor.\n"
        "The dashed line is the apparent direction: the sound now arriving left the drone up to 0.26 s earlier. T_r trades noise against smear.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = os.path.join(out_dir, "07_flyby.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


EXAMPLES = [
    example_converter_response,
    example_drone_hover,
    example_two_drones,
    example_bin_selection,
    example_sustained_source,
    example_frequency_limits,
    example_flyby,
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m foasalsa.examples", description=__doc__)
    parser.add_argument("output_dir", help="directory where the PNG figures are written")
    parser.add_argument("--only", type=int, nargs="*", help="1-based indices of the examples to run")
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(0)
    selected = [EXAMPLES[i - 1] for i in args.only] if args.only else EXAMPLES
    for fn in selected:
        path = fn(args.output_dir)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
