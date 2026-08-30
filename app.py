import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import cv2
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from unet_model import UNet


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "outputs" / "web_runs"
MODEL_PATH = BASE_DIR / "unet_model.pth"
INDEX_PATH = BASE_DIR / "index.html"
MODEL_INPUT_SIZE = (256, 256)
MODEL_THRESHOLD = 0.5
FINAL_THRESHOLD_FLOOR = 0.12
MAX_UPLOAD_SIZE_MB = 20
EARTH_ENGINE_PROJECT = os.getenv("EARTH_ENGINE_PROJECT", "satellite-change-ai-507112")
EARTH_ENGINE_SERVICE_ACCOUNT_JSON = os.getenv("EARTH_ENGINE_SERVICE_ACCOUNT_JSON", "").strip()
SEARCH_USER_AGENT = "satellite-change-detection-ai/1.0"
MAX_MAP_RESULTS = 5
DEFAULT_FETCH_DIMENSION = 768
DEFAULT_BUFFER_METERS = 2500
SENTINEL_START_DATE = "2015-06-27"


app = FastAPI(title="Satellite Change Detection AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")


def load_model() -> UNet:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}")

    model = UNet()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = load_model()
EE_INITIALIZED = False


def ensure_earth_engine_initialized() -> None:
    global EE_INITIALIZED
    if EE_INITIALIZED:
        return

    try:
        if EARTH_ENGINE_SERVICE_ACCOUNT_JSON:
            service_account_info = json.loads(EARTH_ENGINE_SERVICE_ACCOUNT_JSON)
            service_account_email = service_account_info.get("client_email")
            if not service_account_email:
                raise ValueError("The Earth Engine service-account JSON has no client_email.")
            credentials = ee.ServiceAccountCredentials(
                email=service_account_email,
                key_data=EARTH_ENGINE_SERVICE_ACCOUNT_JSON,
            )
            ee.Initialize(credentials=credentials, project=EARTH_ENGINE_PROJECT)
        else:
            ee.Initialize(project=EARTH_ENGINE_PROJECT)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Earth Engine is not available. Make sure your authenticated account has access to "
                f"the Google Cloud project '{EARTH_ENGINE_PROJECT}' and that hosted deployments have "
                "EARTH_ENGINE_SERVICE_ACCOUNT_JSON configured."
            ),
        ) from exc

    EE_INITIALIZED = True


def slugify_year(value: str, fallback: str) -> str:
    cleaned = "".join(ch for ch in value.strip() if ch.isdigit())
    return cleaned[:4] if cleaned else fallback


def validate_year_order(year_before: str, year_after: str) -> None:
    try:
        before_int = int(year_before)
        after_int = int(year_after)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Years must be valid 4-digit numbers.") from exc

    if before_int >= after_int:
        raise HTTPException(
            status_code=400,
            detail=f"The first year must be earlier than the second year. Received {before_int} and {after_int}.",
        )


def read_upload_image(upload: UploadFile) -> Image.Image:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Both image files are required.")

    raw_bytes = upload.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename} is larger than {MAX_UPLOAD_SIZE_MB} MB. Use smaller images.",
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is not a readable image.") from exc

    if min(image.size) < 64:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is too small. Use images at least 64x64.")

    return image


def image_from_bytes(raw_bytes: bytes, file_name: str = "downloaded-image") -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{file_name} is not a readable image.") from exc

    if min(image.size) < 64:
        raise HTTPException(status_code=400, detail=f"{file_name} is too small. Use images at least 64x64.")

    return image


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_heatmap(path: Path, image: np.ndarray, cmap: str = "magma") -> None:
    plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap=cmap, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return image / 255.0


def percentile_stretch(image: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    p_low, p_high = np.percentile(image, (low, high))
    return np.clip((image - p_low) / (p_high - p_low + 1e-10), 0, 1)


def resize_image(image: np.ndarray, target_shape: Tuple[int, int], interpolation: int) -> np.ndarray:
    target_height, target_width = target_shape
    if image.shape[:2] == target_shape:
        return image
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def prepare_images(before_image: Image.Image, after_image: Image.Image) -> Dict[str, np.ndarray]:
    before_np = np.array(before_image)
    after_np = np.array(after_image)

    common_height = min(before_np.shape[0], after_np.shape[0])
    common_width = min(before_np.shape[1], after_np.shape[1])
    target_shape = (common_height, common_width)

    before_resized = resize_image(before_np, target_shape, cv2.INTER_AREA)
    after_resized = resize_image(after_np, target_shape, cv2.INTER_AREA)

    before_norm = normalize_rgb(before_resized)
    after_norm = normalize_rgb(after_resized)

    display_before = percentile_stretch(before_norm)
    display_after = percentile_stretch(after_norm)

    return {
        "before_norm": before_norm,
        "after_norm": after_norm,
        "display_before": display_before,
        "display_after": display_after,
        "target_shape": np.array(target_shape, dtype=np.int32),
    }


def fetch_place_suggestions(query: str) -> List[Dict[str, Any]]:
    query_value = query.strip()
    if len(query_value) < 3:
        raise HTTPException(status_code=400, detail="Enter at least 3 characters to search for a place.")

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query_value,
                "format": "jsonv2",
                "limit": MAX_MAP_RESULTS,
                "addressdetails": 1,
            },
            headers={"User-Agent": SEARCH_USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="Place search is temporarily unavailable.") from exc

    results = []
    for item in payload:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        bounding_box = item.get("boundingbox", [])
        bounds = None
        if len(bounding_box) == 4:
            try:
                south, north, west, east = map(float, bounding_box)
                bounds = {"south": south, "north": north, "west": west, "east": east}
            except ValueError:
                bounds = None

        results.append(
            {
                "name": item.get("display_name", query_value),
                "lat": lat,
                "lon": lon,
                "bounds": bounds,
            }
        )

    if not results:
        raise HTTPException(status_code=404, detail=f"No place results found for '{query_value}'.")

    return results


def validate_search_year(value: str, fallback: str) -> int:
    year_text = slugify_year(value, fallback)
    if len(year_text) != 4:
        raise HTTPException(status_code=400, detail="Years must be 4-digit values.")

    year_value = int(year_text)
    current_year = datetime.now().year
    if year_value < 2015:
        raise HTTPException(
            status_code=400,
            detail=(
                "This version fetches Sentinel-2 imagery, which starts on June 27, 2015. "
                "Use 2015 or later."
            ),
        )
    if year_value > current_year:
        raise HTTPException(status_code=400, detail=f"Year {year_value} is in the future.")
    return year_value


def validate_buffer_km(buffer_km: float) -> float:
    if buffer_km < 0.5 or buffer_km > 10:
        raise HTTPException(status_code=400, detail="Area radius must be between 0.5 km and 10 km.")
    return buffer_km


def get_year_date_range(year: int) -> Tuple[str, str]:
    if year == 2015:
        return SENTINEL_START_DATE, "2015-12-31"
    return f"{year}-01-01", f"{year}-12-31"


def get_region_bounds(lat: float, lon: float, buffer_meters: float) -> Dict[str, float]:
    lat_offset = buffer_meters / 111320.0
    lon_offset = buffer_meters / max(111320.0 * np.cos(np.deg2rad(lat)), 1e-6)
    return {
        "south": lat - lat_offset,
        "north": lat + lat_offset,
        "west": lon - lon_offset,
        "east": lon + lon_offset,
    }


def fetch_image_bytes(url: str, label: str) -> bytes:
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Failed to download the {label} satellite image.") from exc
    return response.content


def pick_sentinel_image(lat: float, lon: float, year: int, buffer_meters: float) -> Dict[str, Any]:
    ensure_earth_engine_initialized()

    start_date, end_date = get_year_date_range(year)
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_meters).bounds()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    try:
        count = int(collection.size().getInfo())
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Failed to query satellite imagery for the selected place.") from exc

    if count == 0:
        fallback_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        try:
            fallback_count = int(fallback_collection.size().getInfo())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Failed to query fallback satellite imagery.") from exc
        if fallback_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No Sentinel-2 image was found for {year} at the selected location.",
            )
        collection = fallback_collection

    image = ee.Image(collection.first())
    image_info = image.toDictionary(["PRODUCT_ID", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]).getInfo()

    rgb_visual = image.select(["B4", "B3", "B2"]).visualize(min=0, max=3000, gamma=1.15)
    thumbnail_url = rgb_visual.getThumbURL(
        {
            "region": region,
            "dimensions": DEFAULT_FETCH_DIMENSION,
            "format": "png",
        }
    )

    acquired_at = datetime.utcfromtimestamp(image_info["system:time_start"] / 1000.0)
    bounds = get_region_bounds(lat, lon, buffer_meters)

    return {
        "thumbnail_url": thumbnail_url,
        "product_id": image_info.get("PRODUCT_ID", "Unknown product"),
        "acquired_at": acquired_at.strftime("%Y-%m-%d"),
        "cloud_cover": float(image_info.get("CLOUDY_PIXEL_PERCENTAGE", 0.0)),
        "bounds": bounds,
        "search_window": {"start": start_date, "end": end_date},
    }


def fetch_satellite_pair(place_name: str, lat: float, lon: float, before_year: int, after_year: int, buffer_km: float) -> Dict[str, Any]:
    buffer_meters = validate_buffer_km(buffer_km) * 1000.0
    before_meta = pick_sentinel_image(lat, lon, before_year, buffer_meters)
    after_meta = pick_sentinel_image(lat, lon, after_year, buffer_meters)

    before_image = image_from_bytes(fetch_image_bytes(before_meta["thumbnail_url"], "before"), "before-satellite-image")
    after_image = image_from_bytes(fetch_image_bytes(after_meta["thumbnail_url"], "after"), "after-satellite-image")

    return {
        "place_name": place_name,
        "lat": lat,
        "lon": lon,
        "buffer_km": buffer_km,
        "before_image": before_image,
        "after_image": after_image,
        "before_meta": before_meta,
        "after_meta": after_meta,
        "map_bounds": before_meta["bounds"],
    }


def run_unet_probability(before_norm: np.ndarray, after_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(
        [before_norm.transpose(2, 0, 1), after_norm.transpose(2, 0, 1)],
        axis=0,
    )
    input_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    resized = F.interpolate(input_tensor, size=MODEL_INPUT_SIZE, mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = MODEL(resized)
        probability_256 = torch.sigmoid(logits).squeeze().cpu().numpy()

    return probability_256, resized.squeeze(0).numpy()


def compute_index_maps(before_norm: np.ndarray, after_norm: np.ndarray) -> Dict[str, np.ndarray]:
    before_r, before_g, before_b = before_norm[..., 0], before_norm[..., 1], before_norm[..., 2]
    after_r, after_g, after_b = after_norm[..., 0], after_norm[..., 1], after_norm[..., 2]

    rgb_diff = np.mean(np.abs(after_norm - before_norm), axis=2)
    gray_before = cv2.cvtColor((before_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_after = cv2.cvtColor((after_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    brightness_delta = gray_after - gray_before

    exg_before = 2 * before_g - before_r - before_b
    exg_after = 2 * after_g - after_r - after_b
    vegetation_delta = exg_after - exg_before

    blue_ratio_before = before_b - (before_r + before_g) / 2
    blue_ratio_after = after_b - (after_r + after_g) / 2
    water_delta = blue_ratio_after - blue_ratio_before

    return {
        "rgb_diff": rgb_diff,
        "brightness_delta": brightness_delta,
        "vegetation_delta": vegetation_delta,
        "water_delta": water_delta,
    }


def compute_change_products(before_norm: np.ndarray, after_norm: np.ndarray) -> Dict[str, Any]:
    probability_256, model_tensor = run_unet_probability(before_norm, after_norm)
    probability_full = resize_image(probability_256, before_norm.shape[:2], cv2.INTER_LINEAR)

    maps = compute_index_maps(before_norm, after_norm)
    rgb_diff_norm = percentile_stretch(maps["rgb_diff"])
    texture_diff = np.abs(cv2.Laplacian(cv2.cvtColor((after_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F))
    texture_diff -= texture_diff.min()
    if texture_diff.max() > 0:
        texture_diff /= texture_diff.max()

    hybrid_score = 0.7 * probability_full + 0.2 * rgb_diff_norm + 0.1 * texture_diff
    adaptive_threshold = max(FINAL_THRESHOLD_FLOOR, float(np.percentile(hybrid_score, 88)))

    def extract_regions(score_threshold: float, min_area: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        model_mask = probability_full >= MODEL_THRESHOLD
        hybrid_mask = hybrid_score >= score_threshold
        combined_mask = np.logical_or(model_mask, hybrid_mask).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask_local = np.zeros_like(combined_mask)
        regions_local: List[Dict[str, Any]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            single_region_mask = np.zeros_like(combined_mask)
            cv2.drawContours(single_region_mask, [contour], -1, 1, thickness=-1)
            cv2.drawContours(cleaned_mask_local, [contour], -1, 1, thickness=-1)
            region_mask = single_region_mask[y:y + h, x:x + w] == 1

            mean_prob = float(np.mean(probability_full[y:y + h, x:x + w][region_mask]))
            mean_rgb = float(np.mean(maps["rgb_diff"][y:y + h, x:x + w][region_mask]))
            vegetation_delta = float(np.mean(maps["vegetation_delta"][y:y + h, x:x + w][region_mask]))
            brightness_delta = float(np.mean(maps["brightness_delta"][y:y + h, x:x + w][region_mask]))
            water_delta = float(np.mean(maps["water_delta"][y:y + h, x:x + w][region_mask]))

            regions_local.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "mean_probability": mean_prob,
                    "mean_rgb_diff": mean_rgb,
                    "vegetation_delta": vegetation_delta,
                    "brightness_delta": brightness_delta,
                    "water_delta": water_delta,
                }
            )

        regions_local.sort(key=lambda item: item["area"], reverse=True)
        return cleaned_mask_local, regions_local

    pixel_count = int(probability_full.size)
    default_min_area = max(120, int(pixel_count * 0.00015))
    cleaned_mask, regions = extract_regions(adaptive_threshold, default_min_area)

    if not regions:
        fallback_threshold = max(0.10, float(np.percentile(hybrid_score, 80)))
        fallback_min_area = max(80, int(pixel_count * 0.00008))
        cleaned_mask, regions = extract_regions(fallback_threshold, fallback_min_area)
        adaptive_threshold = fallback_threshold

    return {
        "probability_256": probability_256,
        "probability_full": probability_full,
        "hybrid_score": hybrid_score,
        "final_mask": cleaned_mask,
        "regions": regions,
        "adaptive_threshold": adaptive_threshold,
        "maps": maps,
        "model_input_shape": tuple(model_tensor.shape),
    }


def classify_region(region: Dict[str, Any]) -> str:
    vegetation_delta = region["vegetation_delta"]
    brightness_delta = region["brightness_delta"]
    water_delta = region["water_delta"]

    if water_delta > 0.08:
        return "Possible water expansion or wetter surface"
    if water_delta < -0.08:
        return "Possible water reduction or drier surface"
    if vegetation_delta > 0.10:
        return "Vegetation increase or greener cover"
    if vegetation_delta < -0.10 and brightness_delta > 0.03:
        return "Vegetation loss with exposed soil or built surface"
    if brightness_delta > 0.08:
        return "Bright surface increase, possibly construction"
    if brightness_delta < -0.08:
        return "Darkening change, possibly shadow, water, or denser cover"
    return "General land-surface change"


def make_overlay(after_display: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = after_display.copy()
    red = np.array([1.0, 0.08, 0.05], dtype=np.float32)
    overlay[mask == 1] = 0.45 * overlay[mask == 1] + 0.55 * red
    return overlay


def annotate_regions(image: np.ndarray, regions: List[Dict[str, Any]], limit: int = 5) -> np.ndarray:
    canvas = (np.clip(image, 0, 1) * 255).astype(np.uint8).copy()

    for index, region in enumerate(regions[:limit], start=1):
        start = (region["x"], region["y"])
        end = (region["x"] + region["w"], region["y"] + region["h"])
        cv2.rectangle(canvas, start, end, (255, 212, 59), 2)
        cv2.putText(
            canvas,
            f"R{index}",
            (region["x"] + 6, region["y"] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return canvas.astype(np.float32) / 255.0


def summarize_changes(mask: np.ndarray, regions: List[Dict[str, Any]], maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
    changed_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    changed_percent = 100.0 * changed_pixels / max(total_pixels, 1)

    vegetation_gain = float(np.logical_and(mask == 1, maps["vegetation_delta"] > 0.08).sum() * 100.0 / max(total_pixels, 1))
    vegetation_loss = float(np.logical_and(mask == 1, maps["vegetation_delta"] < -0.08).sum() * 100.0 / max(total_pixels, 1))
    water_related = float(np.logical_and(mask == 1, np.abs(maps["water_delta"]) > 0.08).sum() * 100.0 / max(total_pixels, 1))
    mean_rgb_diff = float(np.mean(maps["rgb_diff"][mask == 1])) if changed_pixels else 0.0

    summary = {
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_percent": changed_percent,
        "stable_percent": max(0.0, 100.0 - changed_percent),
        "vegetation_gain_percent": vegetation_gain,
        "vegetation_loss_percent": vegetation_loss,
        "water_related_percent": water_related,
        "mean_rgb_diff": mean_rgb_diff,
        "detected_regions": len(regions),
    }

    region_items = []
    for index, region in enumerate(regions[:5], start=1):
        region_items.append(
            {
                "id": f"R{index}",
                "label": classify_region(region),
                "area_percent": 100.0 * region["area"] / max(total_pixels, 1),
                "mean_probability": region["mean_probability"],
                "brightness_delta": region["brightness_delta"],
                "vegetation_delta": region["vegetation_delta"],
            }
        )

    return {"summary": summary, "top_regions": region_items}


def create_explanation_figure(
    run_dir: Path,
    year_before: str,
    year_after: str,
    display_before: np.ndarray,
    display_after: np.ndarray,
    overlay: np.ndarray,
    probability_full: np.ndarray,
    hybrid_score: np.ndarray,
    hybrid_threshold: float,
    mask: np.ndarray,
    regions: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    figure_path = run_dir / "detailed_explanation.png"
    boxed_overlay = annotate_regions(overlay, regions)

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], height_ratios=[1, 1], wspace=0.08, hspace=0.16)

    ax_before = fig.add_subplot(grid[0, 0])
    ax_after = fig.add_subplot(grid[0, 1])
    ax_prob = fig.add_subplot(grid[1, 0])
    ax_overlay = fig.add_subplot(grid[1, 1])
    ax_summary = fig.add_subplot(grid[:, 2])

    fig.suptitle(f"Deep Learning Change Analysis: {year_before} to {year_after}", fontsize=22, weight="bold", y=0.98)

    ax_before.imshow(display_before)
    ax_before.set_title(f"Before Image ({year_before})", fontsize=13, weight="bold")

    ax_after.imshow(display_after)
    ax_after.set_title(f"After Image ({year_after})", fontsize=13, weight="bold")

    prob_plot = ax_prob.imshow(probability_full, cmap="magma", vmin=0, vmax=1)
    ax_prob.contour(mask, levels=[0.5], colors="white", linewidths=0.8)
    ax_prob.set_title("U-Net Change Probability", fontsize=13, weight="bold")
    cbar = fig.colorbar(prob_plot, ax=ax_prob, fraction=0.046, pad=0.02)
    cbar.set_label("Change probability", fontsize=9)

    ax_overlay.imshow(boxed_overlay)
    ax_overlay.set_title("Final Change Overlay and Regions", fontsize=13, weight="bold")

    for axis in [ax_before, ax_after, ax_prob, ax_overlay]:
        axis.set_xticks([])
        axis.set_yticks([])

    ax_summary.axis("off")
    ax_summary.set_title("Explanation Summary", loc="left", fontsize=16, weight="bold", pad=12)

    rows = [
        ("Deep model", "6-channel U-Net on before/after RGB"),
        ("Model resize", "Inputs resized to 256 x 256 for inference"),
        ("Model threshold", f"Probability >= {MODEL_THRESHOLD:.2f}"),
        ("Hybrid threshold", f"Adaptive score >= {hybrid_threshold:.2f}"),
        ("Changed area", f"{summary['changed_percent']:.2f}% of pixels"),
        ("Stable area", f"{summary['stable_percent']:.2f}% of pixels"),
        ("Vegetation gain", f"{summary['vegetation_gain_percent']:.2f}%"),
        ("Vegetation loss", f"{summary['vegetation_loss_percent']:.2f}%"),
        ("Water-related change", f"{summary['water_related_percent']:.2f}%"),
        ("Detected regions", str(summary["detected_regions"])),
    ]

    y = 0.90
    for label, value in rows:
        ax_summary.text(0.02, y, label, transform=ax_summary.transAxes, fontsize=10, color="#3a3a3a")
        ax_summary.text(0.54, y, value, transform=ax_summary.transAxes, fontsize=10.5, weight="bold", color="#111111")
        y -= 0.062

    ax_summary.text(0.02, 0.26, "Top Regions", transform=ax_summary.transAxes, fontsize=12, weight="bold", color="#111111")
    y = 0.21
    if not regions:
        ax_summary.text(0.02, y, "No large regions passed the final threshold.", transform=ax_summary.transAxes, fontsize=10)
    else:
        for index, region in enumerate(regions[:4], start=1):
            area_percent = 100.0 * region["area"] / max(summary["total_pixels"], 1)
            text = (
                f"R{index}: {classify_region(region)} | "
                f"area {area_percent:.2f}% | mean prob {region['mean_probability']:.3f}"
            )
            ax_summary.text(0.02, y, text, transform=ax_summary.transAxes, fontsize=9.1, color="#222222", wrap=True)
            y -= 0.05

    ax_summary.text(
        0.02,
        0.03,
        "Method: the saved U-Net predicts change probability from two RGB images. A hybrid score then combines "
        "the deep-learning output with image-difference cues to improve robustness on uploaded screenshots or map exports.",
        transform=ax_summary.transAxes,
        fontsize=8.8,
        color="#555555",
        wrap=True,
    )

    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.035, right=0.985)
    fig.savefig(figure_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return f"/outputs/web_runs/{run_dir.name}/detailed_explanation.png"


def build_response_payload(
    run_id: str,
    year_before: str,
    year_after: str,
    prepared: Dict[str, np.ndarray],
    products: Dict[str, Any],
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    display_before = prepared["display_before"]
    display_after = prepared["display_after"]
    final_mask = products["final_mask"]
    probability_full = products["probability_full"]
    overlay = make_overlay(display_after, final_mask)
    boxed_overlay = annotate_regions(overlay, products["regions"])

    save_rgb_image(run_dir / "before.png", display_before)
    save_rgb_image(run_dir / "after.png", display_after)
    save_rgb_image(run_dir / "overlay.png", overlay)
    save_rgb_image(run_dir / "overlay_boxed.png", boxed_overlay)
    save_heatmap(run_dir / "probability.png", probability_full)
    save_heatmap(run_dir / "hybrid_score.png", products["hybrid_score"])
    save_mask(run_dir / "mask.png", final_mask)

    summary_bundle = summarize_changes(final_mask, products["regions"], products["maps"])
    summary = summary_bundle["summary"]
    top_regions = summary_bundle["top_regions"]

    detailed_figure_url = create_explanation_figure(
        run_dir=run_dir,
        year_before=year_before,
        year_after=year_after,
        display_before=display_before,
        display_after=display_after,
        overlay=overlay,
        probability_full=probability_full,
        hybrid_score=products["hybrid_score"],
        hybrid_threshold=products["adaptive_threshold"],
        mask=final_mask,
        regions=products["regions"],
        summary=summary,
    )

    explanation_lines = [
        f"The deep-learning model compared the {year_before} and {year_after} images as a 6-channel U-Net input.",
        f"It marked about {summary['changed_percent']:.2f}% of the scene as changed after combining U-Net output with visual difference cues.",
    ]

    if top_regions:
        explanation_lines.append(f"The largest region is {top_regions[0]['label'].lower()}.")
    else:
        explanation_lines.append("No large change region crossed the final confidence threshold.")

    explanation_lines.append(
        "You can use the heatmap to explain model confidence and the overlay to explain where the final detected changes lie."
    )

    report = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "years": {"before": year_before, "after": year_after},
        "model": {
            "name": "U-Net change detector",
            "input_shape": list(products["model_input_shape"]),
            "probability_threshold": MODEL_THRESHOLD,
            "hybrid_threshold": products["adaptive_threshold"],
        },
        "summary": summary,
        "top_regions": top_regions,
        "explanation": explanation_lines,
    }
    if context:
        report["context"] = context
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_lines = ["region,label,area_percent,mean_probability,brightness_delta,vegetation_delta"]
    for region in top_regions:
        csv_lines.append(
            ",".join([
                region["id"],
                json.dumps(region["label"]),
                f"{region['area_percent']:.6f}",
                f"{region['mean_probability']:.6f}",
                f"{region['brightness_delta']:.6f}",
                f"{region['vegetation_delta']:.6f}",
            ])
        )
    (run_dir / "regions.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    base_url = f"/outputs/web_runs/{run_id}"
    response_payload = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "years": {"before": year_before, "after": year_after},
        "model": {
            "name": "U-Net change detector",
            "input_shape": list(products["model_input_shape"]),
            "probability_threshold": MODEL_THRESHOLD,
            "hybrid_threshold": products["adaptive_threshold"],
        },
        "summary": summary,
        "top_regions": top_regions,
        "explanation": explanation_lines,
        "assets": {
            "before": f"{base_url}/before.png",
            "after": f"{base_url}/after.png",
            "overlay": f"{base_url}/overlay.png",
            "overlay_boxed": f"{base_url}/overlay_boxed.png",
            "probability": f"{base_url}/probability.png",
            "hybrid_score": f"{base_url}/hybrid_score.png",
            "mask": f"{base_url}/mask.png",
            "detailed_figure": detailed_figure_url,
            "report_json": f"{base_url}/report.json",
            "regions_csv": f"{base_url}/regions.csv",
        },
    }
    if context:
        response_payload["context"] = context
    return response_payload


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="index.html is missing.")
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": "loaded"}


@app.get("/api/search-place")
def search_place(q: str) -> Dict[str, Any]:
    return {"results": fetch_place_suggestions(q)}


@app.post("/api/analyze")
def analyze(
    before_year: str = Form(...),
    after_year: str = Form(...),
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
) -> Dict[str, Any]:
    before_year_value = slugify_year(before_year, "2015")
    after_year_value = slugify_year(after_year, "2026")
    validate_year_order(before_year_value, after_year_value)

    before_pil = read_upload_image(before_image)
    after_pil = read_upload_image(after_image)

    prepared = prepare_images(before_pil, after_pil)
    products = compute_change_products(prepared["before_norm"], prepared["after_norm"])

    run_id = uuid.uuid4().hex[:12]
    return build_response_payload(run_id, before_year_value, after_year_value, prepared, products)


@app.post("/api/analyze-place")
def analyze_place(
    place_name: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    before_year: str = Form(...),
    after_year: str = Form(...),
    buffer_km: float = Form(2.5),
) -> Dict[str, Any]:
    before_year_value = validate_search_year(before_year, "2016")
    after_year_value = validate_search_year(after_year, "2024")
    validate_year_order(str(before_year_value), str(after_year_value))

    fetched = fetch_satellite_pair(place_name.strip(), lat, lon, before_year_value, after_year_value, buffer_km)
    prepared = prepare_images(fetched["before_image"], fetched["after_image"])
    products = compute_change_products(prepared["before_norm"], prepared["after_norm"])

    run_id = uuid.uuid4().hex[:12]
    context = {
        "place_name": fetched["place_name"],
        "coordinates": {"lat": fetched["lat"], "lon": fetched["lon"]},
        "buffer_km": fetched["buffer_km"],
        "map_bounds": fetched["map_bounds"],
        "imagery": {
            "before": fetched["before_meta"],
            "after": fetched["after_meta"],
        },
    }
    return build_response_payload(run_id, str(before_year_value), str(after_year_value), prepared, products, context=context)
