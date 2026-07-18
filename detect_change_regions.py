import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2

# load images
with rasterio.open("sentinel_data/2015_rgb.tif") as src:
    img15 = src.read().astype(float)

with rasterio.open("sentinel_data/2026_rgb.tif") as src:
    img26 = src.read().astype(float)

# compute change difference
diff = np.abs(img26 - img15).sum(axis=0)

# normalize
diff = (diff - diff.min()) / (diff.max() - diff.min())

# threshold to detect strong change
mask = (diff > 0.25).astype(np.uint8) * 255

# find change regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# create visualization
rgb = np.dstack((img26[2], img26[1], img26[0]))
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(rgb)

for cnt in contours:

    x,y,w,h = cv2.boundingRect(cnt)

    if w*h > 2000:   # ignore small noise

        rect = plt.Rectangle((x,y),w,h,
                             edgecolor='yellow',
                             facecolor='none',
                             linewidth=2)

        ax.add_patch(rect)

plt.title("Detected Land Change Regions")
plt.axis("off")

plt.savefig("sentinel_data/change_regions.png")

print("Change regions image saved")