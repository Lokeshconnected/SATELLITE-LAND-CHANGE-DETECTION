import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from unet_model import UNet

device = torch.device("cpu")

# load trained model
model = UNet().to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.eval()

# load satellite images
with rasterio.open("sentinel_data/2015_rgb.tif") as src:
    img15 = src.read().astype(float)

with rasterio.open("sentinel_data/2026_rgb.tif") as src:
    img26 = src.read().astype(float)

# normalize
img15 = img15 / img15.max()
img26 = img26 / img26.max()

# combine 6 channels
image = np.concatenate([img15, img26], axis=0)

image = torch.tensor(image).unsqueeze(0).float()

# resize to match model training
image = F.interpolate(image, size=(256,256), mode="bilinear")

# run prediction
with torch.no_grad():
    output = model(image)

mask = torch.sigmoid(output).squeeze().numpy()

# threshold mask
mask = mask > 0.5

# visualize
rgb = np.dstack((img26[2], img26[1], img26[0]))
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.figure(figsize=(8,6))
plt.imshow(rgb)
plt.imshow(mask, cmap="Reds", alpha=0.5)
plt.title("AI Land Change Detection")
plt.axis("off")

plt.savefig("sentinel_data/ai_change_mask.png")

print("AI change mask generated")