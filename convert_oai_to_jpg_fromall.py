from pathlib import Path
import matplotlib.pyplot as plt
import pydicom
import numpy as np

input_path = Path('/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/longlegxray_1.C.2')
output_path = Path('/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/jpg')

output_path.mkdir(parents=True, exist_ok=True)

for f in input_path.rglob('*'):
    if not f.is_file():
        continue

    try:
        out_file = output_path / (f.stem + '.jpg')

        # ✅ Skip if already exists
        if out_file.exists():
            print(f"Skipped (exists): {out_file}")
            continue

        ds = pydicom.dcmread(f, force=True)

        if not hasattr(ds, "pixel_array"):
            continue

        img = ds.pixel_array.astype(float)

        # avoid division by zero
        if img.max() == img.min():
            continue

        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Processed: {f}")

    except Exception:
        continue

print("Done.")