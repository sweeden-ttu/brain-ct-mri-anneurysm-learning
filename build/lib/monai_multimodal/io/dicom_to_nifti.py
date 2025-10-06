import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import pydicom
from scipy.ndimage import zoom


def _load_dicom_slices(series_dir: str) -> List[pydicom.dataset.FileDataset]:
    dicom_files: List[str] = []
    for root, _, files in os.walk(series_dir):
        for f in files:
            path = os.path.join(root, f)
            # Quick filter, but still rely on pydicom to verify
            if f.startswith("."):
                continue
            dicom_files.append(path)

    slices: List[pydicom.dataset.FileDataset] = []
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(fp, force=True, stop_before_pixels=False)
            # Ensure PixelData exists; some DICOMs may be metadata-only
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            # Skip files that aren't valid DICOM
            continue

    if not slices:
        raise RuntimeError(f"No DICOM slices with pixel data found under: {series_dir}")

    return slices


def _sort_slices(slices: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
    def slice_sort_key(ds: pydicom.dataset.FileDataset) -> Tuple:
        # Prefer ImagePositionPatient along slice direction; fallback to InstanceNumber
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            # Z is a common axis for ordering, but orientation may vary; using full tuple is fine for monotonicity
            return (float(ipp[2]),)
        inst = getattr(ds, "InstanceNumber", None)
        return (int(inst) if inst is not None else 0,)

    return sorted(slices, key=slice_sort_key)


def _get_spacing_and_orientation(
    sorted_slices: List[pydicom.dataset.FileDataset],
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = sorted_slices[0]

    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    # Compute slice spacing from positions if available; fallback to SliceThickness
    slice_positions: List[np.ndarray] = []
    for ds in sorted_slices:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            slice_positions.append(np.array(ipp, dtype=np.float64))
    if len(slice_positions) >= 2:
        # Use mean distance between consecutive positions
        diffs = [np.linalg.norm(slice_positions[i + 1] - slice_positions[i]) for i in range(len(slice_positions) - 1)]
        slice_spacing = float(np.mean(diffs)) if diffs else float(getattr(first, "SliceThickness", 1.0))
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))

    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        # Default orientation: identity axes
        row_cosines = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        col_cosines = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        row_cosines = np.array(iop[0:3], dtype=np.float64)
        col_cosines = np.array(iop[3:6], dtype=np.float64)

    slice_normal = np.cross(row_cosines, col_cosines)

    origin = np.array(getattr(first, "ImagePositionPatient", [0.0, 0.0, 0.0]), dtype=np.float64)

    return (row_spacing, col_spacing, slice_spacing), row_cosines, col_cosines, slice_normal, origin


def _compute_affine(
    row_spacing: float,
    col_spacing: float,
    slice_spacing: float,
    row_cosines: np.ndarray,
    col_cosines: np.ndarray,
    slice_normal: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    # Construct affine mapping voxel indices (k,j,i) -> patient space (x,y,z) in mm
    # Voxel order in our volume will be (z, y, x) = (slice, row, col) = (k, j, i)
    # Columns correspond to basis vectors scaled by spacing
    R = np.zeros((4, 4), dtype=np.float64)
    R[0:3, 0] = col_cosines * col_spacing  # x-axis (i)
    R[0:3, 1] = row_cosines * row_spacing  # y-axis (j)
    R[0:3, 2] = slice_normal * slice_spacing  # z-axis (k)
    R[0:3, 3] = origin
    R[3, 3] = 1.0
    return R


def _to_hounsfield_units(pixel_array: np.ndarray, ds: pydicom.dataset.FileDataset) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixel_array.astype(np.float32) * slope + intercept
    return hu


def _resample_isotropic_1mm(
    volume_zyx: np.ndarray,
    spacing_zyx_mm: Tuple[float, float, float],
    order: int = 1,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    # spacing_zyx corresponds to (z, y, x)
    z_mm, y_mm, x_mm = spacing_zyx_mm
    zoom_factors = (z_mm / 1.0, y_mm / 1.0, x_mm / 1.0)
    resampled = zoom(volume_zyx, zoom=zoom_factors, order=order)
    return resampled.astype(np.float32, copy=False), (1.0, 1.0, 1.0)


def load_series_as_volume(series_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a DICOM series directory into a 3D volume (z, y, x), along with an affine and minimal metadata.
    Returns (volume, affine, meta)
    """
    slices = _load_dicom_slices(series_dir)
    slices = _sort_slices(slices)

    (row_spacing, col_spacing, slice_spacing), row_cosines, col_cosines, slice_normal, origin = _get_spacing_and_orientation(slices)

    # Stack pixel arrays in (z, y, x) order
    plane_arrays: List[np.ndarray] = []
    for ds in slices:
        arr = ds.pixel_array
        plane_arrays.append(arr)

    # Result shape: (num_slices, rows, cols) -> (z, y, x)
    vol = np.stack(plane_arrays, axis=0).astype(np.float32)

    # Affine: align axes with voxel order (x=i, y=j, z=k)
    affine = _compute_affine(
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        slice_spacing=slice_spacing,
        row_cosines=row_cosines,
        col_cosines=col_cosines,
        slice_normal=slice_normal,
        origin=origin,
    )

    modality = getattr(slices[0], "Modality", "")
    meta = {
        "spacing_zyx": (slice_spacing, row_spacing, col_spacing),
        "modality": str(modality),
    }
    return vol, affine, meta


def convert_series_to_nifti(
    series_dir: str,
    output_path: str,
    modality_hint: Optional[str] = None,
    resample_mri_to_1mm: bool = True,
    normalize_ct: bool = True,
) -> Tuple[str, Dict]:
    """
    Convert a DICOM series to a NIfTI file, applying:
    - MRI: resample to 1mm isotropic voxels
    - CT: HU conversion + clipping + scaling to [0,1]

    Returns (nifti_path, metadata)
    """
    vol, affine, meta = load_series_as_volume(series_dir)

    # Decide modality behavior
    modality_from_dcm = meta.get("modality", "").upper()
    modality = (modality_hint or modality_from_dcm or "").upper()

    # Use first slice for CT HU conversion params
    slices = _load_dicom_slices(series_dir)
    slices = _sort_slices(slices)
    first_slice = slices[0]

    spacing_zyx = meta["spacing_zyx"]

    if modality == "MR" or modality == "MRI":
        if resample_mri_to_1mm:
            vol, spacing_zyx = _resample_isotropic_1mm(vol, spacing_zyx, order=1)
            # Update affine spacing terms to 1mm while preserving orientation and origin
            # Extract direction cosines from affine columns and normalize
            dir_x = affine[0:3, 0]
            dir_y = affine[0:3, 1]
            dir_z = affine[0:3, 2]
            origin = affine[0:3, 3]
            # Normalize direction vectors and scale by 1mm
            def unit(v: np.ndarray) -> np.ndarray:
                n = np.linalg.norm(v)
                return v / n if n > 0 else v

            new_affine = np.eye(4, dtype=np.float64)
            new_affine[0:3, 0] = unit(dir_x) * 1.0
            new_affine[0:3, 1] = unit(dir_y) * 1.0
            new_affine[0:3, 2] = unit(dir_z) * 1.0
            new_affine[0:3, 3] = origin
            affine = new_affine
        # Optional MRI intensity normalization could be added here if desired

    elif modality == "CT":
        if normalize_ct:
            # Convert to HU slice-wise; pixel representation may vary but numpy handles it
            vol_hu = np.empty_like(vol, dtype=np.float32)
            for k, ds in enumerate(slices):
                hu = _to_hounsfield_units(vol[k], ds)
                vol_hu[k] = hu
            # Clip and scale
            vol_hu = np.clip(vol_hu, -1024.0, 3071.0)
            vol = (vol_hu + 1024.0) / (3071.0 + 1024.0)
            vol = vol.astype(np.float32, copy=False)

    # Save NIfTI
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, output_path)

    meta_out = {
        **meta,
        "final_spacing_zyx": spacing_zyx,
        "nifti_path": output_path,
        "modality_effective": modality,
        "shape_zyx": tuple(int(x) for x in vol.shape),
    }
    return output_path, meta_out


def convert_directory(
    input_dir: str,
    output_dir: str,
    modality_hint: Optional[str] = None,
) -> List[Tuple[str, Dict]]:
    """
    Recursively search for DICOM series under `input_dir` and convert each to NIfTI under `output_dir`.
    Each leaf directory with DICOM slices will be treated as a series.
    Returns list of (nifti_path, meta).
    """
    results: List[Tuple[str, Dict]] = []
    for root, dirs, files in os.walk(input_dir):
        # Heuristic: a directory with files containing pixel data is a series
        has_dicom = False
        for f in files:
            if f.startswith("."):
                continue
            fp = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(fp, force=True, stop_before_pixels=False)
                if hasattr(ds, "PixelData"):
                    has_dicom = True
                    break
            except Exception:
                continue

        if has_dicom:
            rel = os.path.relpath(root, input_dir)
            series_name = os.path.basename(root)
            out_path = os.path.join(output_dir, rel + ".nii.gz")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            nifti_path, meta = convert_series_to_nifti(
                series_dir=root,
                output_path=out_path,
                modality_hint=modality_hint,
            )
            results.append((nifti_path, meta))

    if not results:
        raise RuntimeError("No DICOM series with pixel data found to convert.")

    return results


