#!/usr/bin/env python3

from pathlib import Path
from PIL import Image

BASE_PATH = Path("/data/coml-oxmedis/datasets-in-use/oai/1.C.2")
OUTPUT_PATH = Path("/data/coml-oxmedis/datasets-in-use/oai/jpg_checked")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Size filter (set to None to disable)
TARGET_WIDTH = 3000
TARGET_HEIGHT = 1000
WIDTH_TOL = 300
HEIGHT_TOL = 150


def is_about_size(width: int, height: int) -> bool:
    return (
        abs(width - TARGET_WIDTH) <= WIDTH_TOL
        and abs(height - TARGET_HEIGHT) <= HEIGHT_TOL
    )


def main() -> None:
    for folder in BASE_PATH.iterdir():
        if not folder.is_dir():
            continue

        images = list(folder.rglob("*_1x1.jpg"))
        if len(images) <= 1:
            continue

        print(f"\nFolder: {folder} | count={len(images)}")

        for img_path in images:
            try:
                out_file = OUTPUT_PATH / img_path.name

                # ✅ Skip if output already exists
                if out_file.exists():
                    print(f"  Skipped (exists): {out_file}")
                    continue

                with Image.open(img_path) as img:
                    width, height = img.size

                # Optional size filter
                if not is_about_size(width, height):
                    continue

                # Save (or copy) the image
                with Image.open(img_path) as img:
                    img.save(out_file)

                print(f"  Saved: {out_file} ({width}x{height})")

            except Exception as e:
                print(f"  ERROR: {img_path} -> {e}")


if __name__ == "__main__":
    main()