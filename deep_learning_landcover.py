

import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# LOAD BANDS
# -----------------------------

data_path = "sentinel_data/"

with rasterio.open(data_path + "B04_2015.tiff") as red:
    red = red.read(1).astype(float)

with rasterio.open(data_path + "B08_2015.tiff") as nir:
    nir = nir.read(1).astype(float)

# -----------------------------
# NDVI
# -----------------------------

ndvi = (nir - red) / (nir + red + 1e-10)
ndvi = np.clip(ndvi, -1, 1)

# -----------------------------
# LABELS FROM NDVI
# -----------------------------

labels = np.zeros_like(ndvi)

labels[ndvi < 0] = 0
labels[(ndvi >= 0) & (ndvi < 0.2)] = 1
labels[ndvi >= 0.2] = 2

# -----------------------------
# CREATE PATCH DATASET
# -----------------------------

patch_size = 5
half = patch_size // 2

X = []
y = []

for i in range(half, ndvi.shape[0] - half):
    for j in range(half, ndvi.shape[1] - half):

        patch = ndvi[i-half:i+half+1, j-half:j+half+1]

        X.append(patch)
        y.append(labels[i,j])

X = np.array(X)
y = np.array(y)

X = X.reshape(-1,1,patch_size,patch_size)

# convert to tensors
X = torch.tensor(X).float()
y = torch.tensor(y).long()

# -----------------------------
# SIMPLE CNN
# -----------------------------

class PatchCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1,8,3),
            nn.ReLU(),

            nn.Conv2d(8,16,3),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(16,32),
            nn.ReLU(),

            nn.Linear(32,3)
        )

    def forward(self,x):
        return self.net(x)

model = PatchCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# TRAIN
# -----------------------------

epochs = 5

for epoch in range(epochs):

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nCNN training complete.")