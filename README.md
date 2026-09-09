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
