import rasterio
import numpy as np
import matplotlib.pyplot as plt

# load 2015 image
with rasterio.open("sentinel_data/2015_rgb.tif") as src:
    blue15 = src.read(1).astype(float)
    green15 = src.read(2).astype(float)
    red15 = src.read(3).astype(float)

# load 2026 image
with rasterio.open("sentinel_data/2026_rgb.tif") as src:
    blue26 = src.read(1).astype(float)
    green26 = src.read(2).astype(float)
    red26 = src.read(3).astype(float)

# compute difference
diff = np.abs(red26 - red15) + np.abs(green26 - green15) + np.abs(blue26 - blue15)

# normalize
diff = (diff - diff.min()) / (diff.max() - diff.min())

# threshold (detect strong change)
change_mask = diff > 0.25

# create red overlay
overlay = np.zeros((diff.shape[0], diff.shape[1], 4))

overlay[:,:,0] = 1  # red
overlay[:,:,3] = change_mask * 0.6  # transparency

plt.imsave("sentinel_data/change_overlay.png", overlay)

print("Change overlay generated")