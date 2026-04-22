#!/usr/bin/env python3

from pathlib import Path
from typing import Optional
import traceback

import numpy as np
import pydicom
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/1.C.2")
OUTPUT_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/jpg_checked")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def find_associated_001_file(patient_id: str) -> Optional[Path]:
    """
    For a file named XXX_1x1.png, the associated xray is expected at:
        BASE_PATH / XXX / 001
    """
    candidate = BASE_PATH / patient_id / "001"
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def dicom_to_uint8_array(ds) -> Optional[np.ndarray]:
    if not hasattr(ds, "pixel_array"):
        return None

    arr = ds.pixel_array.astype(np.float32)

    # If multi-frame, keep the first frame
    if arr.ndim > 2 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

    # Handle MONOCHROME1 inversion
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return None

    arr = (arr - arr_min) / (arr_max - arr_min)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def main():
    total_seen = 0
    total_saved = 0
    total_missing = 0

    for png_path in BASE_PATH.rglob("*_1x1.png"):
        if not png_path.is_file():
            continue

        total_seen += 1

        try:
            stem = png_path.stem  # e.g. XXX_1x1
            if not stem.endswith("_1x1"):
                continue

            patient_id = stem[:-4]  # remove "_1x1"

            dcm_path = find_associated_001_file(patient_id)
            if dcm_path is None:
                total_missing += 1
                print(f"Missing 001 file for: {png_path}")
                continue

            out_file = OUTPUT_PATH / f"{patient_id}_1x1.jpg"
            if out_file.exists():
                print(f"Skipped (exists): {out_file}")
                continue

            ds = pydicom.dcmread(dcm_path, force=True)
            arr = dicom_to_uint8_array(ds)
            if arr is None:
                print(f"Skipped (no usable pixel data): {dcm_path}")
                continue

            Image.fromarray(arr).save(out_file, quality=95)
            total_saved += 1
            print(f"Saved: {out_file} from {dcm_path}")

        except Exception:
            print(f"ERROR processing: {png_path}")
            traceback.print_exc()

    print("\nDone.")
    print(f"Seen PNGs: {total_seen}")
    print(f"Saved JPGs: {total_saved}")
    print(f"Missing 001 files: {total_missing}")


if __name__ == "__main__":
    main()