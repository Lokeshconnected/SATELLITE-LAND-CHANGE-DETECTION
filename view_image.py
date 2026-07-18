import rasterio
import matplotlib.pyplot as plt

with rasterio.open("sentinel_data/2015.tif") as src:
    img = src.read(1)

plt.imshow(img, cmap="gray")
plt.title("Satellite Image 2015")
plt.colorbar()
plt.show()