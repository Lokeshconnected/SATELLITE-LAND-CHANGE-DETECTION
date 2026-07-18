import rasterio
import numpy as np
import matplotlib.pyplot as plt

data_path = "sentinel_data/"

# -----------------------------
# LOAD SENTINEL BANDS
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
# NDVI CALCULATION
# -----------------------------

ndvi15 = (nir15 - red15) / (nir15 + red15 + 1e-10)
ndvi26 = (nir26 - red26) / (nir26 + red26 + 1e-10)

ndvi15 = np.clip(ndvi15, -1, 1)
ndvi26 = np.clip(ndvi26, -1, 1)

print("NDVI 2015 min:", np.min(ndvi15))
print("NDVI 2015 max:", np.max(ndvi15))

print("NDVI 2026 min:", np.min(ndvi26))
print("NDVI 2026 max:", np.max(ndvi26))

# -----------------------------
# NDVI CHANGE
# -----------------------------

ndvi_change = ndvi26 - ndvi15

# -----------------------------
# VEGETATION ANALYSIS
# -----------------------------

threshold = 0.2

gain = np.sum(ndvi_change > threshold)
loss = np.sum(ndvi_change < -threshold)
no_change = np.sum((ndvi_change >= -threshold) & (ndvi_change <= threshold))

total = gain + loss + no_change

print("\nVEGETATION CHANGE ANALYSIS")
print("Vegetation Gain:", (gain / total) * 100)
print("Vegetation Loss:", (loss / total) * 100)
print("No Major Change:", (no_change / total) * 100)

# -----------------------------
# HOTSPOT DETECTION
# -----------------------------

valid_mask = (ndvi15 > -1) & (ndvi26 > -1)

valid_change = ndvi_change.copy()
valid_change[~valid_mask] = np.nan

flat = valid_change.flatten()
indices = np.argsort(flat)

top_n = 5
loss_indices = indices[:top_n]

rows, cols = np.unravel_index(loss_indices, ndvi_change.shape)

print("\nStrongest vegetation loss hotspots:")

for r, c in zip(rows, cols):
    print("Pixel location:", r, c, " NDVI change:", ndvi_change[r, c])

# -----------------------------
# LAND COVER CLASSIFICATION
# -----------------------------

def classify_landcover(ndvi):

    water = ndvi < 0
    barren = (ndvi >= 0) & (ndvi < 0.2)
    vegetation = ndvi >= 0.2

    landcover = np.zeros((*ndvi.shape, 3), dtype=np.uint8)

    # blue → water
    landcover[water] = [0, 0, 255]

    # brown → barren
    landcover[barren] = [165, 42, 42]

    # green → vegetation
    landcover[vegetation] = [0, 255, 0]

    return landcover


landcover15 = classify_landcover(ndvi15)
landcover26 = classify_landcover(ndvi26)

# -----------------------------
# VISUALIZATION
# -----------------------------

plt.figure(figsize=(20,5))

# NDVI 2015
plt.subplot(1,5,1)
plt.imshow(ndvi15, cmap="RdYlGn")
plt.title("NDVI 2015")
plt.colorbar()

# NDVI 2026
plt.subplot(1,5,2)
plt.imshow(ndvi26, cmap="RdYlGn")
plt.title("NDVI 2026")
plt.colorbar()

# NDVI CHANGE
plt.subplot(1,5,3)
plt.imshow(ndvi_change, cmap="RdYlGn")
plt.title("NDVI Change (Red = Loss, Green = Gain)")
plt.colorbar()

# mark hotspot pixels
for r, c in zip(rows, cols):
    plt.scatter(c, r, color="blue", s=25)

# LAND COVER 2015
plt.subplot(1,5,4)
plt.imshow(landcover15)
plt.title("Land Cover 2015")

# LAND COVER 2026
plt.subplot(1,5,5)
plt.imshow(landcover26)
plt.title("Land Cover 2026")

plt.tight_layout()

plt.savefig("outputs/landcover_change_analysis.png", dpi=300)

plt.show()

print("\nSaved visualization to outputs/landcover_change_analysis.png")