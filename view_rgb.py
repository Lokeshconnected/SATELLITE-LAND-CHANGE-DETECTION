import rasterio
import matplotlib.pyplot as plt
import numpy as np

with rasterio.open("sentinel_data/2026_rgb.tif") as src:

    blue = src.read(1)
    green = src.read(2)
    red = src.read(3)

rgb = np.dstack((red, green, blue))

rgb = rgb / rgb.max()

plt.imshow(rgb)
plt.title("Satellite Image (True Color)")
plt.axis("off")
plt.show()