from pathlib import Path
import matplotlib.pyplot as plt
import pydicom
import numpy as np

input_path = Path('/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/longlegxray_1.E.1')
output_path = Path('/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/jpg')

output_path.mkdir(parents=True, exist_ok=True)

for f in input_path.rglob('*'):
    if not f.is_file():
        continue

    try:
        # force=True lets pydicom try even without .dcm extension
        ds = pydicom.dcmread(f, force=True)

        # skip non-image DICOMs
        if not hasattr(ds, "pixel_array"):
            continue

        img = ds.pixel_array.astype(float)

        # normalize
        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img, cmap='gray')
        plt.axis('off')

        out_file = output_path / (f.stem + '.jpg')
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Processed: {f}")

    except Exception:
        # silently skip non-DICOM or corrupted files
        continue

print("Done.")