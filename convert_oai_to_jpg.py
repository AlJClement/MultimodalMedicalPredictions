from pathlib import Path
import matplotlib.pyplot as plt
import pydicom

input_path = Path('/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/1.E.longlegxray_1.C.2')
output_path = Path('/data/coml-oxmedis/datasets-in-use/xray-longlegs-land/jpgs')

# create output directory
output_path.mkdir(parents=True, exist_ok=True)

for file in input_path.glob('*'):
    try:
        ds = pydicom.dcmread(file)
        img = ds.pixel_array

        plt.imshow(img, cmap='gray')
        plt.axis('off')

        out_file = output_path / (file.stem + '.jpg')
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Failed on {file.name}: {e}")

print("Done.")