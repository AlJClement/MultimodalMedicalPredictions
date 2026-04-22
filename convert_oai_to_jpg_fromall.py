#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pydicom
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/1.C.2")
OUTPUT_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/jpg_checked")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TARGET_RATIO = 3.0   # width / height ~= 3 for 1:3
RATIO_TOL = 0.4


def is_about_ratio(width: int, height: int) -> bool:
    if height == 0:
        return False
    return abs((width / height) - TARGET_RATIO) <= RATIO_TOL


def find_associated_001_file(jpg_path: Path) -> Path | None:
    """
    Try to find the associated DICOM file ending in '001' for a given JPG.
    Search order:
      1) same stem + common DICOM extensions
      2) any file in the same patient/date folder tree whose name contains '001'
         and shares the jpg stem prefix
    """
    parent = jpg_path.parent
    stem = jpg_path.stem

    # Common exact-name possibilities
    for ext in [".dcm", ".DCM", ".dicom", ".DICOM", ""]:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

    # Look for files with 001 in the same directory tree
    search_root = jpg_path.parents[1] if len(jpg_path.parents) >= 2 else parent

    # Try to find something like XX_001, XX-001, XX001, etc.
    for cand in search_root.rglob("*"):
        if not cand.is_file():
            continue
        name = cand.name.lower()
        if "001" not in name:
            continue
        if cand.suffix.lower() in [".dcm", ".dicom", ""] or True:
            # prefer files that share the same base prefix as the jpg
            if stem.split(".")[0].lower() in name or stem.lower() in name:
                return cand

    return None


def dicom_to_array(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)

    # Handle multi-frame by taking first frame
    if arr.ndim > 2 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

    # Handle MONOCHROME1 inversion
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    # Normalize to 0-255
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

    for jpg_path in BASE_PATH.rglob("*.jpg"):
        if not jpg_path.is_file():
            continue

        total_seen += 1

        try:
            rel = jpg_path.relative_to(BASE_PATH)
            parts = rel.parts
            if len(parts) < 3:
                continue

            patient_id = parts[0]
            study_date = parts[1]
            base_name = jpg_path.stem

            # Optional: keep only the 1:3-ish images
            # If you do NOT want this filter, delete the next 6 lines.
            with Image.open(jpg_path) as img:
                w, h = img.size
            if not is_about_ratio(w, h):
                continue

            dcm_path = find_associated_001_file(jpg_path)
            if dcm_path is None:
                total_missing += 1
                print(f"Missing 001 file for: {jpg_path}")
                continue

            out_name = f"{patient_id}-{study_date}-{base_name}.jpg"
            out_file = OUTPUT_PATH / out_name

            if out_file.exists():
                print(f"Skipped (exists): {out_file}")
                continue

            ds = pydicom.dcmread(dcm_path, force=True)

            if not hasattr(ds, "pixel_array"):
                print(f"Skipped (no pixel data): {dcm_path}")
                continue

            arr = dicom_to_array(ds)
            if arr is None:
                print(f"Skipped (flat image): {dcm_path}")
                continue

            Image.fromarray(arr).save(out_file, quality=95)
            total_saved += 1
            print(f"Saved: {out_file} from {dcm_path}")

        except Exception as e:
            print(f"ERROR: {jpg_path} -> {e}")

    print("\nDone.")
    print(f"Seen: {total_seen}")
    print(f"Saved: {total_saved}")
    print(f"Missing 001 files: {total_missing}")


if __name__ == "__main__":
    main()