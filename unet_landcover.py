import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



data_path = "sentinel_data/"

with rasterio.open(data_path + "B04_2015.tiff") as red:
    red = red.read(1).astype(float)

with rasterio.open(data_path + "B08_2015.tiff") as nir:
    nir = nir.read(1).astype(float)

ndvi = (nir - red) / (nir + red + 1e-10)
ndvi = np.clip(ndvi, -1, 1)


labels = np.zeros_like(ndvi)

labels[ndvi < 0] = 0
labels[(ndvi >= 0) & (ndvi < 0.2)] = 1
labels[ndvi >= 0.2] = 2


X = torch.tensor(ndvi).unsqueeze(0).unsqueeze(0).float()
y = torch.tensor(labels).unsqueeze(0).long()



class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,3,1)
        )

    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


model = UNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



epochs = 10

for epoch in range(epochs):

    outputs = model(X)

    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nU-Net training finished")



with torch.no_grad():

    pred = model(X)
    pred = torch.argmax(pred, dim=1)

pred = pred.squeeze().numpy()

print("Segmentation map generated.")

color_map = np.zeros((*pred.shape, 3), dtype=np.uint8)


color_map[pred == 0] = [0, 0, 255]


color_map[pred == 1] = [165, 42, 42]


color_map[pred == 2] = [0, 255, 0]



import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.imshow(color_map)
plt.title("U-Net Land Cover Segmentation")
plt.axis("off")

plt.savefig("outputs/unet_landcover_map.png", dpi=300)

plt.show()

print("Segmentation map saved to outputs/unet_landcover_map.png")