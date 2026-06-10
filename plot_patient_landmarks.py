#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_TXT_DIR = Path("/home/allent/Desktop/data/ddh/ddh_June5th_finalreselection/txt_orig")
DEFAULT_IMG_DIR = Path("/home/allent/Desktop/data/ddh/ddh_June5th_finalreselection/imgs")
DEFAULT_OUT_DIR = Path("/home/allent/Desktop/repos/MultimodalMedicalPredictions/patient_landmark_plots")
ANNOTATION_COLORS = [
    "#ff3b30",
    "#007aff",
    "#34c759",
    "#ff9500",
    "#af52de",
    "#00c7be",
    "#ffcc00",
    "#8e8e93",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find matching patient annotation files and overlay landmarks onto the corresponding images."
    )
    parser.add_argument(
        "--patient",
        default="RBS19732174",
        help="Patient ID or prefix to search for.",
    )
    parser.add_argument(
        "--txt_dir",
        type=Path,
        default=DEFAULT_TXT_DIR,
        help="Directory containing .txt landmark files.",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=DEFAULT_IMG_DIR,
        help="Directory containing image files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where annotated images will be written.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=7,
        help="Point radius for each landmark overlay.",
    )
    return parser.parse_args()


def read_landmarks(txt_path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        x_str, y_str = line.split(",")
        points.append((float(y_str), float(x_str)))
    return points


def draw_landmarks(
    img_path: Path,
    annotations: list[tuple[str, list[tuple[float, float]], str]],
    out_path: Path,
    radius: int,
) -> None:
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for source_name, landmarks, color in annotations:
        for idx, (x, y) in enumerate(landmarks, start=1):
            left = x - radius
            top = y - radius
            right = x + radius
            bottom = y + radius
            draw.ellipse((left, top, right, bottom), fill=color, outline="white", width=2)
            draw.text((x + radius + 3, y - radius - 3), str(idx), fill=color, font=font)

    legend_padding = 8
    legend_line_height = 16
    legend_width = 220
    legend_height = legend_padding * 2 + legend_line_height * len(annotations)
    legend_box = (10, 10, 10 + legend_width, 10 + legend_height)
    draw.rectangle(legend_box, fill=(255, 255, 255), outline=(0, 0, 0))

    for row_idx, (source_name, _landmarks, color) in enumerate(annotations):
        y = 10 + legend_padding + row_idx * legend_line_height
        draw.rectangle((18, y + 2, 30, y + 14), fill=color, outline=(0, 0, 0))
        draw.text((38, y), source_name, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main():
    args = parse_args()

    txt_dir = args.txt_dir.expanduser().resolve()
    img_dir = args.img_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() / args.patient

    if not txt_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {txt_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    txt_paths = sorted(txt_dir.glob(f"{args.patient}*.txt"))
    if not txt_paths:
        raise FileNotFoundError(f"No annotation files found for patient prefix: {args.patient}")

    annotation_sets = []
    for idx, txt_path in enumerate(txt_paths):
        annotation_sets.append((txt_path.stem, read_landmarks(txt_path), ANNOTATION_COLORS[idx % len(ANNOTATION_COLORS)]))

    written = []
    for txt_path in txt_paths:
        stem = txt_path.stem
        img_candidates = [img_dir / f"{stem}.png", img_dir / f"{stem}.jpg", img_dir / f"{stem}.jpeg"]
        img_path = next((candidate for candidate in img_candidates if candidate.is_file()), None)
        if img_path is None:
            print(f"Skipping {stem}: no matching image found.")
            continue

        out_path = out_dir / f"{stem}_all_annotations.png"
        draw_landmarks(img_path, annotation_sets, out_path, args.radius)
        written.append(out_path)
        print(f"Wrote {out_path}")

    if not written:
        raise FileNotFoundError("No annotated images were written. Matching annotation files were found, but no images matched.")

    print(f"Done. Wrote {len(written)} annotated image(s) to {out_dir}")


if __name__ == "__main__":
    main()
