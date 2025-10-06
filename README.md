## MONAI Multi-Modal Model (DICOM → NIfTI)

This package provides:
- DICOM→NIfTI conversion for MRI and CT using PyDicom and NiBabel
- MRI resampling to 1mm isotropic voxels
- CT intensity normalization (HU conversion, clipping, scaling)
- A MONAI 3D U-Net model suitable for brain tumor tasks, configurable for multi-modal inputs
- CLI for conversion: `mm-convert`

### Quickstart

1) Create and activate a virtual environment (recommended):
```bash
python3 -m venv .venv && source .venv/bin/activate
```

2) Install requirements:
```bash
pip install -r requirements.txt
```

3) Install the package (editable for development):
```bash
pip install -e .
```

4) Convert a DICOM series directory to NIfTI:
```bash
mm-convert --input /path/to/dicom_series_dir \
           --output /path/to/output_dir \
           --modality MRI   # or CT
```

This writes a single `.nii.gz` with spacing normalization for MRI, and HU normalization for CT.

### Build an EGG

```bash
python setup.py bdist_egg
```
The egg will be placed under `dist/`.

### Notes
- MRI volumes are resampled to 1mm isotropic voxels.
- CT volumes are converted to Hounsfield Units, clipped to [-1024, 3071], and scaled to [0, 1].
- The included MONAI model (`monai_multimodal.models.brain_tumor_unet`) is a 3D U-Net commonly used in brain tumor segmentation (BraTS-style) and can be used as a base for training.


