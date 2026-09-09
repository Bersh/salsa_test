import torch

from foasalsa import FoaConverter, FoaSalsa
from foasalsa.geometry import azimuth_elevation, direction_vector
from foasalsa.synth import circular_array, noise_bursts, plane_wave

FS = 24000.0


def _foa(az: float, seconds: float = 2.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    mics = circular_array(6, 0.03)
    s = noise_bursts(int(FS * seconds), FS, 200, 3500, generator=g)
    p = plane_wave(s, direction_vector(az, 0.0), mics, FS)
    return FoaConverter(sample_rate=FS).convert(p[None].float(), mics)


def test_shapes_and_zero_regions():
    foa = torch.cat([_foa(30.0), _foa(-90.0)])
    salsa = FoaSalsa(sample_rate=FS, fmax=4000.0)
    feats = salsa.compute(foa)
    n_frames = foa.shape[-1] // salsa.hop_length + 1
    assert feats.shape == (2, 7, salsa.n_bins, n_frames)
    assert feats.dtype == torch.float32
    assert torch.isfinite(feats).all()
    lo, hi = salsa.band
    assert (feats[:, 4:, :lo] == 0).all() and (feats[:, 4:, hi:] == 0).all()
    # EIV rows are either zero or unit vectors
    norms = feats[:, 4:].norm(dim=1)
    assert torch.all((norms < 1e-5) | ((norms - 1).abs() < 1e-4))


def test_eiv_points_at_the_source():
    salsa = FoaSalsa(sample_rate=FS, fmax=4000.0)
    for az in (30.0, -90.0):
        res = salsa.analyze(_foa(az))
        est_az, est_el = azimuth_elevation(res.eiv.permute(0, 2, 3, 1))
        m = res.mask
        assert m.float().mean() > 0.05
        err = ((est_az[m] - az + 180) % 360 - 180).abs()
        assert err.median() < 1.0
        assert est_el[m].abs().max() < 1e-3  # planar array => z = 0


def test_tests_can_be_disabled():
    foa = _foa(30.0)
    res_all = FoaSalsa(sample_rate=FS, fmax=4000.0, use_magnitude_test=False, use_coherence_test=False).analyze(foa)
    res_sel = FoaSalsa(sample_rate=FS, fmax=4000.0).analyze(foa)
    assert res_all.mask[:, res_all.band_mask].all()
    assert res_sel.mask.sum() < res_all.mask.sum()
    assert torch.equal(res_all.eiv, res_all.eiv_raw)


def test_sustained_source_preset_keeps_bins():
    """A perfectly steady sound: the paper's tracker masks it after ~2 s, the preset keeps it."""
    from foasalsa.synth import drone_source

    g = torch.Generator().manual_seed(3)
    mics = circular_array(8, 0.06)
    s = drone_source(int(FS * 6.0), FS, n_motors=1, wander=0.0, noise_ratio=0.0, generator=g)
    p = plane_wave(s, direction_vector(45.0, 0.0), mics, FS)
    foa = FoaConverter(sample_rate=FS).convert(p[None].float(), mics)
    default = FoaSalsa(sample_rate=FS, fmin=150.0, fmax=3000.0).analyze(foa)
    preset = FoaSalsa.for_sustained_sources(fmax=3000.0, sample_rate=FS).analyze(foa)
    assert preset.band_mask.equal(default.band_mask)
    band = default.band_mask
    early = default.times < 1.0
    late = default.times > 4.0
    kept = lambda r, sel: r.magnitude_mask[0][band][:, sel].float().mean().item()
    assert kept(default, early) > 0.5 * kept(preset, early)  # both trackers accept the sound at the onset
    assert kept(default, late) < 0.25 * kept(preset, late)  # the default floor has crept above the sound
    assert kept(preset, late) > 0.8 * kept(preset, early)  # the preset keeps accepting it
    est_az, _ = azimuth_elevation(preset.eiv.permute(0, 2, 3, 1))
    err = ((est_az[preset.mask] - 45.0 + 180) % 360 - 180).abs()
    assert err.median() < 1.0
