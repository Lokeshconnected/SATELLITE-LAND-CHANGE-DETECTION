import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Folder containing the Sentinel images
data_path = "sentinel_data/"

# -----------------------------
# STEP 1: Load Sentinel Bands
# -----------------------------

with rasterio.open(data_path + "B04_2015.tiff") as red15:
    red15_band = red15.read(1).astype(float)

with rasterio.open(data_path + "B08_2015.tiff") as nir15:
    nir15_band = nir15.read(1).astype(float)

with rasterio.open(data_path + "B04_2026.tiff") as red26:
    red26_band = red26.read(1).astype(float)

with rasterio.open(data_path + "B08_2026.tiff") as nir26:
    nir26_band = nir26.read(1).astype(float)

# -----------------------------
# STEP 2: Calculate NDVI
# -----------------------------

ndvi2015 = (nir15_band - red15_band) / (nir15_band + red15_band + 1e-10)
ndvi2026 = (nir26_band - red26_band) / (nir26_band + red26_band + 1e-10)

# keep NDVI within valid range
ndvi2015 = np.clip(ndvi2015, -1, 1)
ndvi2026 = np.clip(ndvi2026, -1, 1)

# -----------------------------
# STEP 3: NDVI statistics
# -----------------------------

print("NDVI 2015 min:", np.min(ndvi2015))
print("NDVI 2015 max:", np.max(ndvi2015))

print("NDVI 2026 min:", np.min(ndvi2026))
print("NDVI 2026 max:", np.max(ndvi2026))

# -----------------------------
# STEP 4: NDVI change
# -----------------------------

ndvi_change = ndvi2026 - ndvi2015

# -----------------------------
# STEP 5: Vegetation analysis
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
# STEP 6: Create Change Map
# -----------------------------

change_map = np.zeros_like(ndvi_change)

change_map[ndvi_change > threshold] = 1
change_map[ndvi_change < -threshold] = -1

# -----------------------------
# STEP 7: Visualization
# -----------------------------

plt.figure(figsize=(15,5))

# NDVI 2015
plt.subplot(1,3,1)
plt.imshow(ndvi2015, cmap="RdYlGn")
plt.title("NDVI 2015")
plt.colorbar()

# NDVI 2026
plt.subplot(1,3,2)
plt.imshow(ndvi2026, cmap="RdYlGn")
plt.title("NDVI 2026")
plt.colorbar()

# NDVI Change
plt.subplot(1,3,3)
plt.imshow(change_map, cmap="RdYlGn")
plt.title("Vegetation Change (Red = Loss, Green = Gain)")
plt.colorbar()

plt.tight_layout()
plt.show()