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
    candidate = BASE_PATH / patient_id / "001"
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def dicom_to_uint8_array(ds):
    if not hasattr(ds, "pixel_array"):
        return None

    arr = ds.pixel_array.astype(np.float32)

    # Handle multi-frame
    if arr.ndim > 2 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

    # MONOCHROME1 fix
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    if arr.max() == arr.min():
        return None

    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (arr * 255).astype("uint8")

    return arr


def main():
    total_seen = 0
    total_saved = 0

    # ✅ Correct pattern
    for jpg_path in BASE_PATH.rglob("*_1x1.jpg"):
        if not jpg_path.is_file():
            continue

        total_seen += 1

        try:
            stem = jpg_path.stem  # XXX_1x1
            if not stem.endswith("_1x1"):
                continue

            patient_id = stem[:-4]  # remove "_1x1"

            dcm_path = find_associated_001_file(patient_id)
            if dcm_path is None:
                print(f"Missing DICOM for: {jpg_path}")
                continue

            out_file = OUTPUT_PATH / f"{patient_id}_from_dcm.jpg"

            if out_file.exists():
                print(f"Skipped (exists): {out_file}")
                continue

            ds = pydicom.dcmread(dcm_path, force=True)
            arr = dicom_to_uint8_array(ds)

            if arr is None:
                print(f"Bad DICOM: {dcm_path}")
                continue

            Image.fromarray(arr).save(out_file)
            print(f"Saved: {out_file}")

            total_saved += 1

        except Exception:
            print(f"ERROR: {jpg_path}")
            traceback.print_exc()

    print("\nDone.")
    print(f"Seen: {total_seen}")
    print(f"Saved: {total_saved}")


if __name__ == "__main__":
    main()