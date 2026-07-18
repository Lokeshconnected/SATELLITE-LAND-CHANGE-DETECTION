import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os

app = FastAPI()

# -----------------------------
# MODEL
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
# CREATE OUTPUT FOLDER
# -----------------------------

os.makedirs("outputs", exist_ok=True)


# -----------------------------
# PREDICTION
# -----------------------------

def run_prediction():

    data_path = "sentinel_data/"

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

    img15 = (ndvi15+1)/2
    img26 = (ndvi26+1)/2

    PATCH = 64
    height,width = img15.shape

    change_map = np.zeros((height,width))

    for y in range(0,height-PATCH,PATCH):
        for x in range(0,width-PATCH,PATCH):

            p1 = img15[y:y+PATCH,x:x+PATCH]
            p2 = img26[y:y+PATCH,x:x+PATCH]
            p3 = ndvi_diff[y:y+PATCH,x:x+PATCH]

            combined = np.stack([p1, p2, p3], axis=0)

            tensor = torch.tensor(combined).unsqueeze(0).float()

            output = model(tensor)

            pred = torch.argmax(output).item()

            if pred == 0:
                change_map[y:y+PATCH,x:x+PATCH] = 1

    plt.figure(figsize=(8,6))
    plt.imshow(change_map,cmap="Reds")
    plt.title("AI Detected Land Change")
    plt.colorbar()

    output_path = "outputs/change_map.png"

    plt.savefig(output_path)
    plt.close()

    return output_path


# -----------------------------
# API
# -----------------------------

@app.get("/predict")
def predict():
    try:
        image_path = run_prediction()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return FileResponse(image_path)
