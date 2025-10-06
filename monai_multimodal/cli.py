import argparse
import os
from typing import Optional

from .io.dicom_to_nifti import convert_series_to_nifti, convert_directory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to NIfTI with modality-specific normalization"
    )
    parser.add_argument("--input", required=True, help="Path to DICOM series directory or a root folder")
    parser.add_argument("--output", required=True, help="Output NIfTI file path or directory if input is a root folder")
    parser.add_argument(
        "--modality",
        choices=["MRI", "CT"],
        help="Optional modality hint in case DICOM header is missing/incorrect",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, will scan input directory recursively and convert all series to output dir",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.recursive:
        os.makedirs(args.output, exist_ok=True)
        convert_directory(args.input, args.output, modality_hint=args.modality)
    else:
        out = args.output
        if os.path.isdir(out):
            # If output is a directory, write file named after leaf input directory
            series_name = os.path.basename(os.path.normpath(args.input))
            out = os.path.join(out, series_name + ".nii.gz")
        convert_series_to_nifti(args.input, out, modality_hint=args.modality)


if __name__ == "__main__":
    main()


