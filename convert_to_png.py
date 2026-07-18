import rasterio
import numpy as np
import matplotlib.pyplot as plt

def convert(input_file, output_file):

    with rasterio.open(input_file) as src:
        blue = src.read(1).astype(float)
        green = src.read(2).astype(float)
        red = src.read(3).astype(float)

    rgb = np.dstack((red, green, blue))

    # contrast stretch (important for satellite images)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

    plt.imsave(output_file, rgb)

convert("sentinel_data/2015_rgb.tif", "sentinel_data/2015.png")
convert("sentinel_data/2026_rgb.tif", "sentinel_data/2026.png")

print("Better PNG images created")