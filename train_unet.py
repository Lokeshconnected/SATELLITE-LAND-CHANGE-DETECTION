import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rasterio
from unet_model import UNet
import torch.nn.functional as F

device = torch.device("cpu")

model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

# load images
with rasterio.open("sentinel_data/2015_rgb.tif") as src:
    img15 = src.read().astype(float)

with rasterio.open("sentinel_data/2026_rgb.tif") as src:
    img26 = src.read().astype(float)

# normalize
img15 = img15 / img15.max()
img26 = img26 / img26.max()

# stack input channels (6 channels)
image = np.concatenate([img15, img26], axis=0)

image = torch.tensor(image).unsqueeze(0).float()

# resize to 256x256
image = F.interpolate(image, size=(256,256), mode="bilinear")

# create pseudo-label using difference
diff = np.abs(img26 - img15).sum(axis=0)
diff = (diff - diff.min()) / (diff.max() - diff.min())

mask = (diff > 0.25).astype(np.float32)
mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()

mask = F.interpolate(mask, size=(256,256), mode="nearest")

# training loop
for epoch in range(20):

    optimizer.zero_grad()

    output = model(image)

    loss = loss_fn(output, mask)

    loss.backward()

    optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), "unet_model.pth")

print("Model trained and saved")