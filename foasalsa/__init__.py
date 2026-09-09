"""foasalsa: microphone-array to FOA conversion and SALSA features for SELD.

Public API
----------
FoaConverter  - converts an A-format (raw microphone array) batch into first-order
                ambisonics (B-format, SN3D normalisation, channel order W, X, Y, Z).
FoaSalsa      - computes SALSA features (log-linear spectrograms stacked with the
                eigenvector-based intensity vector) from an FOA batch, following
                Nguyen et al., "SALSA: Spatial Cue-Augmented Log-Spectrogram Features
                for Polyphonic Sound Event Localization and Detection", 2022
                (https://arxiv.org/abs/2110.00275).
"""

from .converter import FoaConverter
from .salsa import FoaSalsa, SalsaResult

__all__ = ["FoaConverter", "FoaSalsa", "SalsaResult", "__version__"]
__version__ = "0.1.0"
