#!/usr/bin/env python3

from pathlib import Path
from PIL import Image, ImageFile
import shutil
import traceback

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/1.C.2")
OUTPUT_PATH = Path("/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/jpg_checked")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 3000
TARGET_HEIGHT = 1000
WIDTH_TOL = 300
HEIGHT_TOL = 150


def is_about_size(width: int, height: int) -> bool:
    return (
        abs(width - TARGET_WIDTH) <= WIDTH_TOL
        and abs(height - TARGET_HEIGHT) <= HEIGHT_TOL
    )


def main():
    total_seen = 0
    total_saved = 0

    for img_path in BASE_PATH.rglob("*.jpg"):
        if not img_path.is_file():
            continue

        total_seen += 1
        print(f"Checking: {img_path}")

        try:
            rel = img_path.relative_to(BASE_PATH)
            parts = rel.parts

            # Expect: patient/date/file.jpg
            if len(parts) < 3:
                print(f"  Skip: unexpected path structure -> {rel}")
                continue

            patient_id = parts[0]
            study_date = parts[1]
            filename = img_path.stem

            with Image.open(img_path) as img:
                img.load()
                width, height = img.size

            print(f"  Size: {width}x{height}")

            if not is_about_size(width, height):
                print("  Skip: size not in range")
                continue

            out_name = f"{patient_id}-{study_date}-{filename}.jpg"
            out_file = OUTPUT_PATH / out_name

            if out_file.exists():
                print(f"  Skipped (exists): {out_file}")
                continue

            shutil.copy2(img_path, out_file)
            print(f"  Saved: {out_file}")
            total_saved += 1

        except Exception:
            print(f"  ERROR: {img_path}")
            traceback.print_exc()

    print("\nDone.")
    print(f"Files seen: {total_seen}")
    print(f"Files saved: {total_saved}")


if __name__ == "__main__":
    main()