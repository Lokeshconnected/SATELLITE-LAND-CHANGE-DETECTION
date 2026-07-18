import rasterio
import numpy as np
import os
from PIL import Image

data_path = "sentinel_data/"
output_path = "dataset/"

os.makedirs(output_path + "change", exist_ok=True)
os.makedirs(output_path + "no_change", exist_ok=True)

PATCH_SIZE = 64
THRESHOLD = 0.2

# -----------------------------
# LOAD BANDS
# -----------------------------

with rasterio.open(data_path + "B04_2015.tiff") as red15:
    red15 = red15.read(1).astype(float)

with rasterio.open(data_path + "B08_2015.tiff") as nir15:
    nir15 = nir15.read(1).astype(float)

with rasterio.open(data_path + "B04_2026.tiff") as red26:
    red26 = red26.read(1).astype(float)

with rasterio.open(data_path + "B08_2026.tiff") as nir26:
    nir26 = nir26.read(1).astype(float)

# -----------------------------
# NDVI
# -----------------------------

ndvi15 = (nir15 - red15) / (nir15 + red15 + 1e-10)
ndvi26 = (nir26 - red26) / (nir26 + red26 + 1e-10)

ndvi_change = ndvi26 - ndvi15

height, width = ndvi15.shape

change_count = 0
nochange_count = 0

# -----------------------------
# CREATE PATCHES
# -----------------------------

for y in range(0, height - PATCH_SIZE, PATCH_SIZE):

    for x in range(0, width - PATCH_SIZE, PATCH_SIZE):

        p1 = ndvi15[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        p2 = ndvi26[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        p3 = ndvi_change[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        change_value = np.mean(np.abs(p3))

        combined = np.stack([p1, p2, p3], axis=2)

        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-10)
        combined = (combined * 255).astype(np.uint8)

        img = Image.fromarray(combined)

        if change_value > THRESHOLD:

            img.save(output_path + "change/" + str(change_count) + ".png")
            change_count += 1

        else:

            img.save(output_path + "no_change/" + str(nochange_count) + ".png")
            nochange_count += 1

print("Dataset created")
print("Change patches:", change_count)
print("No change patches:", nochange_count)