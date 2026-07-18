import ee
import geemap
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()


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

    collection1 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate("2015-01-01", "2015-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image2015 = collection1.first().select(["B4","B8"])

    collection2 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate("2026-01-01", "2026-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image2026 = collection2.first().select(["B4","B8"])


    os.makedirs("sentinel_data",exist_ok=True)

    geemap.ee_export_image(
        image2015,
        filename="sentinel_data/2015.tif",
        scale=10,
        region=point.buffer(5000).bounds()
    )

    geemap.ee_export_image(
        image2026,
        filename="sentinel_data/2026.tif",
        scale=10,
        region=point.buffer(5000).bounds()
    )


@app.get("/analyze")

def analyze(lat:float, lon:float):
    try:
        download_images(lat, lon)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image download failed: {exc}") from exc

    return {"status":"images downloaded"}
