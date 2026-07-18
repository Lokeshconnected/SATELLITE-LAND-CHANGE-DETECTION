import ee
import geemap
import os
import torch
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from unet_model import UNet

app = FastAPI()

device = torch.device("cpu")

model = UNet().to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.eval()


def initialize_earth_engine():
    try:
        ee.Initialize(project="satellite-change-ai")
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine is not available. Check your network connection and Google Earth Engine authentication."
        ) from exc


def download_images(lat, lon):
    initialize_earth_engine()

    point = ee.Geometry.Point(lon, lat)

    collection2015 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate("2015-01-01", "2015-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    collection2026 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate("2024-01-01", "2024-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    img2015 = collection2015.first().select(["B2","B3","B4"])
    img2026 = collection2026.first().select(["B2","B3","B4"])

    os.makedirs("sentinel_data", exist_ok=True)

    geemap.ee_export_image(
        img2015,
        filename="sentinel_data/2015_rgb.tif",
        scale=10,
        region=point.buffer(5000).bounds()
    )

    geemap.ee_export_image(
        img2026,
        filename="sentinel_data/2026_rgb.tif",
        scale=10,
        region=point.buffer(5000).bounds()
    )


def run_prediction():

    with rasterio.open("sentinel_data/2015_rgb.tif") as src:
        img15 = src.read().astype(float)

    with rasterio.open("sentinel_data/2026_rgb.tif") as src:
        img26 = src.read().astype(float)

    img15 = img15 / img15.max()
    img26 = img26 / img26.max()

    image = np.concatenate([img15, img26], axis=0)

    image = torch.tensor(image).unsqueeze(0).float()

    image = F.interpolate(image, size=(256,256), mode="bilinear")

    with torch.no_grad():
        output = model(image)

    mask = torch.sigmoid(output).squeeze().numpy()

    mask = mask > 0.5

    rgb = np.dstack((img26[2], img26[1], img26[0]))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.figure(figsize=(8,6))
    plt.imshow(rgb)
    plt.imshow(mask, cmap="Reds", alpha=0.5)
    plt.axis("off")

    os.makedirs("outputs", exist_ok=True)

    plt.savefig("outputs/result.png")

    return "outputs/result.png"


@app.get("/analyze")

def analyze(lat: float, lon: float):
    try:
        download_images(lat, lon)
        img = run_prediction()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return FileResponse(img)
