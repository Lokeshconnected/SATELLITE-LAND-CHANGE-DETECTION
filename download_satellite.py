import ee
import geemap
import os

def main():
    ee.Initialize(project="satellite-change-ai")

    # example location
    lat = 12.97
    lon = 77.59

    point = ee.Geometry.Point(lon, lat)

    collection2015 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate("2015-01-01", "2015-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image2015 = collection2015.first().select(["B2","B3","B4"])

    collection2026 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate("2024-01-01", "2024-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image2026 = collection2026.first().select(["B2","B3","B4"])

    os.makedirs("sentinel_data", exist_ok=True)

    geemap.ee_export_image(
        image2015,
        filename="sentinel_data/2015_rgb.tif",
        scale=10,
        region=point.buffer(5000).bounds()
    )

    geemap.ee_export_image(
        image2026,
        filename="sentinel_data/2026_rgb.tif",
        scale=10,
        region=point.buffer(5000).bounds()
    )

    print("RGB satellite images downloaded")


if __name__ == "__main__":
    main()
