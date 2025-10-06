from typing import Dict, Tuple

import numpy as np
from monai.transforms import Compose, NormalizeIntensity, ScaleIntensityRange


def normalize_mri_to_unit_range(volume: np.ndarray) -> np.ndarray:
    # Z-score then scale to [0, 1] for stability
    vol = volume.astype(np.float32, copy=False)
    mean = float(np.mean(vol))
    std = float(np.std(vol))
    if std > 0:
        vol = (vol - mean) / std
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    return vol


def normalize_ct_unit_range(volume_hu_scaled: np.ndarray) -> np.ndarray:
    # Assume input already scaled to [0,1]; clamp to be safe
    vol = np.clip(volume_hu_scaled, 0.0, 1.0)
    return vol.astype(np.float32, copy=False)


