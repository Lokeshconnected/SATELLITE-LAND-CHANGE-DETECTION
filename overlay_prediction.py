import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# -----------------------------
# LOAD MODEL
# -----------------------------

class ChangeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)
        x = self.fc(x)

        return x


model = ChangeCNN()
model.load_state_dict(torch.load("change_model.pth", map_location="cpu"))
model.eval()

print("Model loaded")

# -----------------------------
# LOAD SATELLITE DATA
# -----------------------------

data_path = "sentinel_data/"

with rasterio.open(data_path + "B04_2026.tiff") as red:
    red = red.read(1).astype(float)

with rasterio.open(data_path + "B08_2026.tiff") as nir:
    nir = nir.read(1).astype(float)

ndvi = (nir - red) / (nir + red + 1e-10)

# Normalize for display
image = (ndvi + 1) / 2

height, width = image.shape

PATCH = 64

change_map = np.zeros((height,width))

# -----------------------------
# LOAD NDVI FEATURES
# -----------------------------

with rasterio.open(data_path + "B04_2015.tiff") as red15:
    red15 = red15.read(1).astype(float)

with rasterio.open(data_path + "B08_2015.tiff") as nir15:
    nir15 = nir15.read(1).astype(float)

with rasterio.open(data_path + "B04_2026.tiff") as red26:
    red26 = red26.read(1).astype(float)

with rasterio.open(data_path + "B08_2026.tiff") as nir26:
    nir26 = nir26.read(1).astype(float)

ndvi15 = (nir15-red15)/(nir15+red15+1e-10)
ndvi26 = (nir26-red26)/(nir26+red26+1e-10)

ndvi_diff = ndvi26 - ndvi15

# -----------------------------
# PATCH PREDICTION
# -----------------------------

for y in range(0,height-PATCH,PATCH):

    for x in range(0,width-PATCH,PATCH):

        p1 = ndvi15[y:y+PATCH,x:x+PATCH]
        p2 = ndvi26[y:y+PATCH,x:x+PATCH]
        p3 = ndvi_diff[y:y+PATCH,x:x+PATCH]

        patch = np.stack([p1,p2,p3],axis=0)

        tensor = torch.tensor(patch).unsqueeze(0).float()

        output = model(tensor)

        pred = torch.argmax(output).item()

        if pred == 0:

            change_map[y:y+PATCH,x:x+PATCH] = 1


# -----------------------------
# OVERLAY VISUALIZATION
# -----------------------------

plt.figure(figsize=(8,6))

plt.imshow(image,cmap="gray")

plt.imshow(change_map,cmap="Reds",alpha=0.5)

plt.title("AI Detected Land Change Overlay")

plt.colorbar()

os.makedirs("outputs",exist_ok=True)

plt.savefig("outputs/change_overlay.png")

print("Overlay saved to outputs/change_overlay.png")