#!/usr/bin/env python3

from pathlib import Path
from PIL import Image

BASE_PATH = Path("/data/coml-oxmedis/datasets-in-use/oai/1.C.2")
OUTPUT_PATH = Path("/data/coml-oxmedis/datasets-in-use/oai/jpg_checked")

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

    for patient_dir in BASE_PATH.iterdir():
        if not patient_dir.is_dir():
            continue

        # 🔍 Find images in this patient folder
        images = list(patient_dir.rglob("*_1x1.jpg"))
        count = len(images)

        print(f"\nPatient: {patient_dir.name} | count={count}")

        # ✅ Only proceed if at least 1 (or change to >1 if you want)
        if count == 0:
            continue

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                # Apply size filter
                if not is_about_size(width, height):
                    continue

                # Unique output name per patient
                out_name = f"{patient_dir.name}_{img_path.stem}.jpg"
                out_file = OUTPUT_PATH / out_name

                # Skip if already saved
                if out_file.exists():
                    print(f"  Skipped (exists): {out_file}")
                    continue

                # Save image
                with Image.open(img_path) as img:
                    img.save(out_file)

                print(f"  Saved: {out_file} ({width}x{height})")
                total_saved += 1

            except Exception as e:
                print(f"  ERROR: {img_path} -> {e}")

    print("\nDone.")
    print(f"Total saved: {total_saved}")


if __name__ == "__main__":
    main()