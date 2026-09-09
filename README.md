# foasalsa

PyTorch implementation of

* **`FoaConverter`** – converts raw microphone-array recordings (A-format, arbitrary
  geometry given as microphone coordinates) into first-order ambisonics (B-format,
  SN3D normalisation, channel order `W, X, Y, Z`).
* **`FoaSalsa`** – computes **SALSA** features (Spatial Cue-Augmented Log-Spectrogram)
  from FOA audio, following Nguyen et al., *SALSA: Spatial Cue-Augmented Log-Spectrogram
  Features for Polyphonic Sound Event Localization and Detection*, IEEE/ACM TASLP 2022
  ([arXiv:2110.00275](https://arxiv.org/abs/2110.00275)).

```bash
pip install .
python -m foasalsa.examples <output_dir>   # renders the example figures
python -m pytest                            # after: pip install .[test]
```

## Usage

```python
import torch
from foasalsa import FoaConverter, FoaSalsa

mics = torch.tensor([[ 0.03, 0.0, 0.0], [0.0,  0.03, 0.0],
                     [-0.03, 0.0, 0.0], [0.0, -0.03, 0.0]])    # [C, 3] metres
audio = torch.randn(2, 4, 24000)                              # [N, C, T] at 24 kHz

foa = FoaConverter(sample_rate=24000).convert(audio, mics)     # [N, 4, T]  (W, X, Y, Z)
features = FoaSalsa(sample_rate=24000, fmax=4000).compute(foa) # [N, 7, F, T]
```

`features[:, :4]` are the log-linear spectrograms of W, X, Y, Z and `features[:, 4:]`
is the eigenvector-based intensity vector (EIV): at every single-source
time-frequency bin a unit vector `(x, y, z)` pointing at the source; zero elsewhere.
`FoaSalsa.analyze` returns all intermediate quantities (STFT, eigenvalues,
direct-to-reverberant ratio, noise floor, test masks).

## What the classes do

**`FoaConverter`** models the array as omnidirectional sensors at the given
positions and solves, per frequency, a regularised least-squares fit of the
first-order spherical harmonics on a dense grid of plane-wave directions. The
resulting frequency responses are applied as linear-phase FIR filters (one grouped
`conv1d`), so the output is time-aligned with the input. Consequences of the
physics, all illustrated by the examples:

* the directional channels (X, Y, Z) can only be recovered where the array is
  larger than a fraction of a wavelength; below that the regularisation
  (`max_gain_db`) limits noise amplification;
* above the spatial-aliasing frequency (roughly `c / (2 * spacing)`) the encoder
  is unreliable, so `FoaSalsa(fmax=...)` should be set below it;
* a **planar** array carries no information about the normal component of the
  direction of arrival: for an x-y array the Z channel is exactly zero and the EIV
  keeps the azimuth but loses the elevation.

**`FoaSalsa`** follows the paper: 24 kHz, 512-point Hann STFT with hop 300;
`log(|X|^2)` spectrograms; a spatial covariance averaged over `2*T_r+1 = 7`
frames; the principal eigenvector normalised by its W component, real part, unit
norm; the EIV is kept only for bins that pass the magnitude test (running RMS of
`|W|` above `1.5 x` an adaptive noise floor) and the coherence test
(`sigma_1 / sigma_2 > 5`), inside `[fmin, fmax]`. All constants are constructor
arguments.

## Drone (multirotor) sounds

The examples target the propeller sound of a fibre-optic-controlled quadcopter:
continuous, made of harmonic clusters (one comb per motor at its blade-passing
frequency), usually distant, and moving. The classes are unchanged; what differs
is the configuration:

* `FoaSalsa.for_sustained_sources(fmax=3000)` – the paper's noise-floor tracker
  slowly absorbs any continuous sound into the "background" and then fails the
  magnitude test for it. The preset turns the tracker into a slow minimum
  tracker (no fast re-arming after dips, 1000x slower rise) and sets
  `fmin = 150 Hz` to stay above wind noise.
* Array: a ring of 8 microphones (r = 6 cm) **plus a centre microphone**
  (`foasalsa.synth.centred_circular_array`). The average over a plain open ring
  crosses zero at `kr = 2.4` (2.2 kHz for r = 6 cm) and reverses every EIV above
  it; the centre microphone gives the encoder a clean W channel.
* Band: 150 Hz to 3 kHz (strongest propeller harmonics, below the 3.7 kHz
  aliasing limit of the ring).
* `foasalsa.synth` provides `drone_source`, `air_absorption`, `wind_noise` and
  `flyby` (Doppler, 1/r and time-varying direction) for building scenes.

## Examples

`python -m foasalsa.examples out/` writes seven figures built from synthetic
scenes on planar arrays; `docs/EXAMPLES.md` explains each one in plain language.

1. `01_converter_response.png` – encoder accuracy, the sign flip of a plain
   ring's W channel and how a centre microphone fixes it, aliasing limits and
   low-frequency noise gain.
2. `02_drone_hover.png` – the seven SALSA channels for a hovering drone, and the
   same drone at 30° elevation (identical EIV: a planar array cannot see elevation).
3. `03_two_drones.png` – two drones with different motor speeds: their harmonic
   combs interleave and every bin's EIV points at the dominant one; the
   coherence test removes the bins where the combs collide.
4. `04_bin_selection.png` – a distant drone in wind and diffuse noise with a wall
   reflection: magnitude and coherence tests, and the DRR threshold sweep.
5. `05_sustained_source.png` – an 8 s hover: the paper's noise-floor tracker
   versus the sustained-source preset.
6. `06_frequency_limits.png` – EIV error versus frequency for the small ring and
   the drone array: sensor-noise amplification at low frequencies, aliasing at
   high ones, and the chosen band.
7. `07_flyby.png` – a 30 m/s fly-past with Doppler shift and propagation delay,
   for three covariance window lengths.

## Design decisions

Choices that are not fixed by the paper, and why they were made:

* **Encoder = regularised least squares on a direction grid.** The array
  geometry is arbitrary (no baffle, no spherical sampling), so the encoder is
  fitted numerically: for every frequency, `E = argmin ||E H - Y||² + λ||E||²`
  over 768 plane-wave directions. Nothing else is assumed about the array.
* **Regularisation is specified as a gain cap.** `λ = 1 / (4 g²)` with
  `g = 10^(max_gain_db / 20)`, which makes the worst-case amplification of any
  channel exactly `max_gain_db` (default 20 dB). A cap in dB is easier to relate
  to microphone self-noise than a bare λ.
* **8-fold symmetric direction grid** (96 Fibonacci points mirrored in x, y, z).
  Guarantees that a planar array yields an exactly zero Z channel and that the
  fit has no directional bias from the grid.
* **Linear-phase FIR filters (512 taps, Hann) applied with one grouped `conv1d`.**
  The group delay is compensated, so `foa` is sample-aligned with `audio`; the
  filter bank is cached per (geometry, dtype, device). This keeps the converter
  free of STFT parameters and independent of `FoaSalsa`.
* **Channel order `W, X, Y, Z` (paper Eq. 6), SN3D.** For ACN order
  (`W, Y, Z, X`) use `foa[:, [0, 2, 3, 1]]`.
* **Edge-aware covariance averaging.** The 7-frame box average uses fewer frames
  at the clip edges instead of the reference implementation's wrap-around
  padding, which would mix the end of a clip with its start.
* **The default magnitude test replicates the reference tracker exactly**,
  including its drift on sustained sounds, so the default `FoaSalsa()`
  reproduces the paper. Behaviour for continuous sounds is opt-in through
  `for_sustained_sources`. Band edges follow the reference too
  (`lower_bin = max(1, floor(fmin·n_fft/fs))`, `upper_bin = min(n_bins, floor(fmax·n_fft/fs))`).
* **Preset tuning.** A first version that only zeroed the slow rise still lost
  the drone, because every dip below the floor re-armed three fast 2 % steps and
  the floor ratcheted upwards. The final preset removes that re-arming
  (`noise_floor_hold_frames = 0`) and keeps a small slow rise
  (`noise_floor_slow_scale = 0.002`, 0.004 % per frame, about 90 s to absorb a
  perfectly steady tone) so the floor still follows slow changes of the weather.
* **Synthetic drone model.** Blade-passing frequency `rpm / 60 × blades`
  (12 000 rpm, 2 blades → 400 Hz), four motors with ±4 % speed spread and 3 %
  slow wander, 25 harmonics, 20 % broadband turbulence; air absorption
  `0.01 dB/m × (f / 1 kHz)^1.3`; wind noise generated independently per
  microphone, falling 12 dB per octave above 300 Hz; fly-bys use the exact
  propagation delay per microphone, so Doppler shift, 1/r decay and the lag of
  the apparent direction emerge rather than being added.
* **Planar arrays only in the examples** (a task requirement). The converter and
  the features are fully 3-D; `tests/test_converter.py` checks elevation on a
  tetrahedral array.

## Findings

Measured on the synthetic scenes (all directions are known exactly, so every
error is attributable to the array or the algorithm):

* **Converter accuracy.** Inside the array's band the X/Y azimuth is within 1° of
  the truth and Z is zero to 1e-6 for planar arrays; W gain is above 0.9 down to
  the regularisation limit; the output is sample-aligned.
* **SALSA accuracy.** For a clean plane wave the median EIV azimuth error is
  about 0.01°; on the noisy scenes below it is set by the noise, not the method.
* **Open rings reverse the direction above `kr = 2.4`.** The average over a
  plain ring crosses zero there (2.2 kHz for r = 6 cm, 4.4 kHz for r = 3 cm),
  W changes sign and every EIV between that frequency and `fmax` points
  backwards. A centre microphone keeps W at exactly 1 and also lowers the
  mid-band noise gain. This was found while reviewing the drone figures and
  changed the recommended array.
* **The paper's noise floor absorbs a hovering drone in about 3 s.** On an 8 s
  hover the paper's tracker keeps 27 % of the band bins (10 658 cells) and the
  preset 86 % (33 747 cells); both sets peak sharply at the true azimuth, so the
  discarded bins were correct, only declared "background".
* **For a distant drone in wind the coherence test does almost all the work.**
  Drone at 150 m with a wall reflection, wind noise and diffuse noise: median
  azimuth error 16° over all bins, 14° after the magnitude test alone, 8.0° after
  the coherence test alone, 7.8° after both. Wind noise is different at every
  microphone, so it never looks like a single source. Raising the DRR threshold
  from 1 to 10 cuts the median error from 14° to 6°; at the paper's 5, half of
  the bins are kept at 7.7°; at 40 only 5 % remain.
* **Two drones separate by harmonic, not by averaging.** With different motor
  speeds their combs interleave and each bin points at the louder drone; the
  coherence test removes the bins where the combs collide. Two drones at the
  same speed would be much harder.
* **Wrong directions above the aliasing limit are consistent over time**, so
  without `fmax` they would look like confident cues to a network. Wind noise
  lives below 300 Hz, hence `fmin = 150 Hz`.
* **Planar arrays cannot see elevation.** A drone at 0° and at 30° elevation
  gives identical features.
* **Fly-past at 30 m/s (15 m to the side).** Doppler shifts the harmonics by
  about ±9 %. Far from the array the 75th-percentile azimuth error is 20°–40°
  with `T_r = 1`, about 7° with the paper's `T_r = 3` and about 4° with
  `T_r = 12`; at the closest point all three meet at about 10°, where motion
  inside the window dominates. Sound travel time delays the apparent direction
  by up to 0.26 s at 90 m.

## Conclusions and limitations

* Recommended configuration for quadcopter propeller sound on a planar array:
  8-microphone ring of 6 cm radius plus a centre microphone,
  `FoaSalsa.for_sustained_sources(fmax=3000)` (band 150 Hz–3 kHz), and a
  covariance window longer than the paper's 7 frames when the drone is
  distant. Alternatively disable the magnitude test and rely on the coherence
  test alone.
* A planar array yields azimuth only. Elevation needs at least one microphone out
  of the plane; the converter already supports it.
* All results are from synthetic free-field scenes with omnidirectional
  microphones and no scattering body. Real recordings will add microphone
  mismatch, a mount, ground reflections and non-stationary wind, and should be
  used to confirm the thresholds before training a detector.
* The noise-floor tracker remains a heuristic; the preset makes it slow rather
  than principled. Drones with identical motor speeds, or harmonics closer than
  the 47 Hz STFT bin spacing, are not separated per bin.

