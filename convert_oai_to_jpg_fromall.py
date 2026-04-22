#!/usr/bin/env python3

from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/1.C.2")
OUTPUT_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/jpg_checked")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 3000
TARGET_HEIGHT = 1000
WIDTH_TOL = 300
HEIGHT_TOL = 150


def is_about_size(width, height):
    return (
        abs(width - TARGET_WIDTH) <= WIDTH_TOL
        and abs(height - TARGET_HEIGHT) <= HEIGHT_TOL
    )


def main():
    total_saved = 0

    for img_path in BASE_PATH.rglob("*_1x1.jpg"):
        if not img_path.is_file():
            continue

        try:
            # Expect structure: BASE / patient / date / file
            rel = img_path.relative_to(BASE_PATH)
            parts = rel.parts

            if len(parts) < 3:
                continue  # skip unexpected structure

            patient_id = parts[0]
            study_date = parts[1]
            filename = img_path.stem

            with Image.open(img_path) as img:
                img.load()
                width, height = img.size

            if not is_about_size(width, height):
                continue

            # ✅ Build flattened filename
            out_name = f"{patient_id}-{study_date}-{filename}.jpg"
            out_file = OUTPUT_PATH / out_name

            if out_file.exists():
                print(f"Skipped (exists): {out_file}")
                continue

            with Image.open(img_path) as img:
                img.save(out_file)

            print(f"Saved: {out_file} ({width}x{height})")
            total_saved += 1

        except Exception as e:
            print(f"ERROR: {img_path} -> {e}")

    print("\nDone.")
    print(f"Total saved: {total_saved}")


if __name__ == "__main__":
    main()