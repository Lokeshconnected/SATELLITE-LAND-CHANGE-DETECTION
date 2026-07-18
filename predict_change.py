import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------

class ChangeCNN(nn.Module):

    def __init__(self):
        super(ChangeCNN, self).__init__()

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


# -----------------------------
# LOAD MODEL
# -----------------------------

model = ChangeCNN()
model.load_state_dict(torch.load("change_model.pth", map_location="cpu"))
model.eval()

print("Model loaded")

# -----------------------------
# LOAD SATELLITE DATA
# -----------------------------

data_path = "sentinel_data/"

with rasterio.open(data_path + "B04_2015.tiff") as red15:
    red15 = red15.read(1).astype(float)

with rasterio.open(data_path + "B08_2015.tiff") as nir15:
    nir15 = nir15.read(1).astype(float)

with rasterio.open(data_path + "B04_2026.tiff") as red26:
    red26 = red26.read(1).astype(float)

with rasterio.open(data_path + "B08_2026.tiff") as nir26:
    nir26 = nir26.read(1).astype(float)

# -----------------------------
# NDVI
# -----------------------------

ndvi15 = (nir15 - red15) / (nir15 + red15 + 1e-10)
ndvi26 = (nir26 - red26) / (nir26 + red26 + 1e-10)

img15 = (ndvi15 + 1) / 2
img26 = (ndvi26 + 1) / 2

# -----------------------------
# PATCH SCANNING
# -----------------------------

PATCH = 64

height, width = img15.shape

change_map = np.zeros((height,width))

for y in range(0, height-PATCH, PATCH):

    for x in range(0, width-PATCH, PATCH):

        patch15 = img15[y:y+PATCH, x:x+PATCH]
        patch26 = img26[y:y+PATCH, x:x+PATCH]

        combined = np.stack([patch15, patch26, patch26], axis=0)

        tensor = torch.tensor(combined).unsqueeze(0).float()

        output = model(tensor)

        prediction = torch.argmax(output).item()

        if prediction == 0:
            change_map[y:y+PATCH, x:x+PATCH] = 1


# -----------------------------
# VISUALIZATION
# -----------------------------

plt.figure(figsize=(8,6))

plt.imshow(change_map, cmap="Reds")

plt.title("AI Detected Land Change")

plt.colorbar()

plt.show()