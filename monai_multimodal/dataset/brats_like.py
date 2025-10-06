from typing import List, Optional, Sequence

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class NiftiVolumeDataset(Dataset):
    """
    Simple dataset reading preconverted NIfTI volumes.
    Each item returns a numpy array volume (z, y, x) and optional label if provided.
    """

    def __init__(self, image_paths: Sequence[str], label_paths: Optional[Sequence[str]] = None):
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths) if label_paths is not None else None
        if self.label_paths is not None:
            assert len(self.image_paths) == len(self.label_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = nib.load(self.image_paths[idx])
        vol = img.get_fdata(dtype=np.float32)
        if self.label_paths is None:
            return vol
        lab = nib.load(self.label_paths[idx]).get_fdata(dtype=np.float32)
        return vol, lab


