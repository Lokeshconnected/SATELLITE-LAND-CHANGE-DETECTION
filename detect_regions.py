import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import cv2


# -----------------------------
# MODEL
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
# LOAD DATA
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

ndvi15 = (nir15-red15)/(nir15+red15+1e-10)
ndvi26 = (nir26-red26)/(nir26+red26+1e-10)

ndvi_diff = ndvi26 - ndvi15

image = (ndvi26 + 1) / 2


height,width = image.shape

PATCH = 64

change_map = np.zeros((height,width))


# -----------------------------
# MODEL PREDICTION
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
# REGION DETECTION
# -----------------------------

change_binary = (change_map*255).astype(np.uint8)

contours,_ = cv2.findContours(change_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


# -----------------------------
# VISUALIZATION
# -----------------------------

fig,ax = plt.subplots(figsize=(8,6))

ax.imshow(image,cmap="gray")

ax.imshow(change_map,cmap="Reds",alpha=0.4)


for cnt in contours:

    x,y,w,h = cv2.boundingRect(cnt)

    if w*h > 500:

        rect = plt.Rectangle((x,y),w,h,edgecolor='yellow',facecolor='none',linewidth=2)

        ax.add_patch(rect)


plt.title("AI Detected Change Regions")

os.makedirs("outputs",exist_ok=True)

plt.savefig("outputs/change_regions.png")

print("Saved: outputs/change_regions.png")