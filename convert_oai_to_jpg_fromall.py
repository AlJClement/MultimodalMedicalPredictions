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


def find_associated_001_file(jpg_path: Path) -> Optional[Path]:
    """
    Example:
      /.../9034677/20060118/01341204_1x1.jpg
    maps to:
      /.../9034677/20060118/01341204/001
    """
    parent_dir = jpg_path.parent  # keeps patient/date path intact
    jpg_base = jpg_path.stem.replace("_1x1", "")  # 01341204

    candidate = parent_dir / jpg_base / "001"
    if candidate.exists() and candidate.is_file():
        return candidate

    return None


def dicom_to_uint8_array(ds):
    if not hasattr(ds, "pixel_array"):
        return None

    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim > 2 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

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

    for jpg_path in BASE_PATH.rglob("*_1x1.jpg"):
        if not jpg_path.is_file():
            continue

        total_seen += 1

        try:
            dcm_path = find_associated_001_file(jpg_path)
            if dcm_path is None:
                total_missing += 1
                print(f"Missing DICOM for: {jpg_path}")
                print(f"  Expected: {jpg_path.parent / jpg_path.stem.replace('_1x1', '') / '001'}")
                continue

            rel = jpg_path.relative_to(BASE_PATH)
            patient_id = rel.parts[0]
            study_date = rel.parts[1]
            jpg_base = jpg_path.stem.replace("_1x1", "")

            out_file = OUTPUT_PATH / f"{patient_id}-{study_date}-{jpg_base}.jpg"

            if out_file.exists():
                print(f"Skipped (exists): {out_file}")
                continue

            ds = pydicom.dcmread(dcm_path, force=True)
            arr = dicom_to_uint8_array(ds)

            if arr is None:
                print(f"Skipped (no usable pixel data): {dcm_path}")
                continue

            Image.fromarray(arr).save(out_file, quality=95)
            print(f"Saved: {out_file} from {dcm_path}")
            total_saved += 1

        except Exception:
            print(f"ERROR processing: {jpg_path}")
            traceback.print_exc()

    print("\nDone.")
    print(f"Seen: {total_seen}")
    print(f"Saved: {total_saved}")
    print(f"Missing DICOMs: {total_missing}")


if __name__ == "__main__":
    main()