import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import rasterio


DATA_DIR = "sentinel_data"
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "change_explanation_figure.png")


def read_rgb(path):
    """Read Sentinel RGB TIFFs stored as blue, green, red and return display RGB."""
    with rasterio.open(path) as src:
        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)

    rgb = np.dstack((red, green, blue))
    return percentile_stretch(rgb)


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def percentile_stretch(image, low=2, high=98):
    """Normalize image for clear plotting while avoiding outlier washout."""
    image = image.astype(np.float32)
    p_low, p_high = np.percentile(image, (low, high))
    return np.clip((image - p_low) / (p_high - p_low + 1e-10), 0, 1)


def normalize01(image):
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-10)


def compute_ndvi(red, nir):
    ndvi = (nir - red) / (nir + red + 1e-10)
    return np.clip(ndvi, -1, 1)


def resize_to_match(image, target_shape):
    """Resize arrays if exported rasters have slightly different dimensions."""
    target_height, target_width = target_shape
    if image.shape[:2] == target_shape:
        return image

    interpolation = cv2.INTER_LINEAR if image.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def build_change_products(rgb15, rgb26, ndvi15, ndvi26):
    ndvi_diff = ndvi26 - ndvi15
    vegetation_loss = np.clip(-ndvi_diff, 0, None)
    vegetation_gain = np.clip(ndvi_diff, 0, None)
    rgb_diff = np.mean(np.abs(rgb26 - rgb15), axis=2)

    change_score = 0.65 * normalize01(np.abs(ndvi_diff)) + 0.35 * normalize01(rgb_diff)
    threshold = max(0.22, np.percentile(change_score, 88))

    change_mask = (change_score >= threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(change_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(250, int(change_mask.size * 0.001))

    clean_mask = np.zeros_like(change_mask)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(clean_mask, [contour], -1, 1, thickness=-1)

        region_slice = (slice(y, y + h), slice(x, x + w))
        region_ndvi_delta = float(np.mean(ndvi_diff[region_slice][clean_mask[region_slice] == 1]))
        region_score = float(np.mean(change_score[region_slice][clean_mask[region_slice] == 1]))

        regions.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": int(area),
                "ndvi_delta": region_ndvi_delta,
                "score": region_score,
            }
        )

    regions = sorted(regions, key=lambda item: item["area"], reverse=True)

    return {
        "ndvi_diff": ndvi_diff,
        "vegetation_loss": vegetation_loss,
        "vegetation_gain": vegetation_gain,
        "rgb_diff": rgb_diff,
        "change_score": change_score,
        "change_mask": clean_mask,
        "regions": regions,
        "threshold": threshold,
    }


def make_overlay(rgb, mask):
    overlay = rgb.copy()
    red = np.array([1.0, 0.08, 0.05])
    overlay[mask == 1] = 0.45 * overlay[mask == 1] + 0.55 * red
    return overlay


def add_region_boxes(ax, regions, limit=5):
    for index, region in enumerate(regions[:limit], start=1):
        rect = Rectangle(
            (region["x"], region["y"]),
            region["w"],
            region["h"],
            fill=False,
            edgecolor="#ffd43b",
            linewidth=2.2,
        )
        ax.add_patch(rect)
        ax.text(
            region["x"] + 4,
            region["y"] + 16,
            f"R{index}",
            color="black",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffd43b", edgecolor="none"),
        )


def classify_main_change(ndvi_delta):
    if ndvi_delta <= -0.08:
        return "vegetation decrease / exposed surface"
    if ndvi_delta >= 0.08:
        return "vegetation increase"
    return "surface or brightness change"


def draw_summary_panel(ax, products):
    mask = products["change_mask"]
    ndvi_diff = products["ndvi_diff"]
    regions = products["regions"]

    changed_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    changed_percent = 100 * changed_pixels / max(total_pixels, 1)

    loss_percent = 100 * np.logical_and(mask == 1, ndvi_diff < -0.05).sum() / max(total_pixels, 1)
    gain_percent = 100 * np.logical_and(mask == 1, ndvi_diff > 0.05).sum() / max(total_pixels, 1)
    stable_percent = max(0, 100 - changed_percent)
    mean_delta = float(np.mean(ndvi_diff[mask == 1])) if changed_pixels else 0.0

    ax.axis("off")
    ax.set_title("What Changed?", loc="left", fontsize=16, weight="bold", pad=12)

    summary = [
        ("Changed area", f"{changed_percent:.1f}% of image"),
        ("Mostly stable", f"{stable_percent:.1f}% of image"),
        ("NDVI loss", f"{loss_percent:.1f}% vegetation decrease"),
        ("NDVI gain", f"{gain_percent:.1f}% vegetation increase"),
        ("Mean NDVI delta", f"{mean_delta:+.3f} inside changed pixels"),
        ("Detected regions", f"{len(regions)} significant region(s)"),
    ]

    y = 0.89
    for label, value in summary:
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=10, color="#3b3b3b")
        ax.text(0.54, y, value, transform=ax.transAxes, fontsize=10.5, weight="bold", color="#111111")
        y -= 0.062

    ax.text(
        0.02,
        0.46,
        "Legend",
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        color="#111111",
    )
    legend_items = [
        ("Red overlay", "Strong land-cover change"),
        ("Yellow boxes", "Largest detected regions"),
        ("Red NDVI", "Vegetation decreased"),
        ("Green NDVI", "Vegetation increased"),
    ]

    y = 0.405
    colors = ["#f03e3e", "#ffd43b", "#c92a2a", "#2f9e44"]
    for color, (label, value) in zip(colors, legend_items):
        ax.add_patch(Rectangle((0.02, y - 0.012), 0.035, 0.025, transform=ax.transAxes, color=color))
        ax.text(0.075, y, label, transform=ax.transAxes, fontsize=9.5, weight="bold", va="center")
        ax.text(0.37, y, value, transform=ax.transAxes, fontsize=9.5, va="center", color="#333333")
        y -= 0.048

    ax.text(
        0.02,
        0.20,
        "Largest Region Notes",
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        color="#111111",
    )

    y = 0.15
    if not regions:
        ax.text(0.02, y, "No large change regions passed the threshold.", transform=ax.transAxes, fontsize=10)
        return

    for index, region in enumerate(regions[:3], start=1):
        area_percent = 100 * region["area"] / max(total_pixels, 1)
        meaning = classify_main_change(region["ndvi_delta"])
        text = f"R{index}: {area_percent:.2f}% area, NDVI {region['ndvi_delta']:+.3f}; {meaning}"
        ax.text(0.02, y, text, transform=ax.transAxes, fontsize=8.9, color="#222222", wrap=True)
        y -= 0.04

    ax.text(
        0.02,
        0.02,
        "Method: RGB difference and NDVI difference were combined, thresholded, cleaned, and grouped into regions.",
        transform=ax.transAxes,
        fontsize=8.7,
        color="#555555",
        wrap=True,
    )


def create_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    red15 = read_band(os.path.join(DATA_DIR, "B04_2015.tiff"))
    nir15 = read_band(os.path.join(DATA_DIR, "B08_2015.tiff"))
    red26 = read_band(os.path.join(DATA_DIR, "B04_2026.tiff"))
    nir26 = read_band(os.path.join(DATA_DIR, "B08_2026.tiff"))

    target_shape = red15.shape

    rgb15 = resize_to_match(read_rgb(os.path.join(DATA_DIR, "2015_rgb.tif")), target_shape)
    rgb26 = resize_to_match(read_rgb(os.path.join(DATA_DIR, "2026_rgb.tif")), target_shape)

    red26 = resize_to_match(red26, red15.shape)
    nir26 = resize_to_match(nir26, nir15.shape)

    ndvi15 = compute_ndvi(red15, nir15)
    ndvi26 = compute_ndvi(red26, nir26)

    products = build_change_products(rgb15, rgb26, ndvi15, ndvi26)
    overlay = make_overlay(rgb26, products["change_mask"])

    fig = plt.figure(figsize=(17, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], height_ratios=[1, 1], wspace=0.08, hspace=0.18)

    ax_before = fig.add_subplot(grid[0, 0])
    ax_after = fig.add_subplot(grid[0, 1])
    ax_overlay = fig.add_subplot(grid[1, 0])
    ax_ndvi = fig.add_subplot(grid[1, 1])
    ax_summary = fig.add_subplot(grid[:, 2])

    fig.suptitle("Satellite Land Change Explanation: 2015 to 2026", fontsize=22, weight="bold", y=0.985)

    ax_before.imshow(rgb15)
    ax_before.set_title("2015 True Color", fontsize=13, weight="bold")

    ax_after.imshow(rgb26)
    ax_after.set_title("2026 True Color", fontsize=13, weight="bold")

    ax_overlay.imshow(overlay)
    add_region_boxes(ax_overlay, products["regions"])
    ax_overlay.set_title("Change Overlay on 2026 Image", fontsize=13, weight="bold")

    ndvi_plot = ax_ndvi.imshow(products["ndvi_diff"], cmap="RdYlGn", vmin=-0.45, vmax=0.45)
    ax_ndvi.contour(products["change_mask"], levels=[0.5], colors="#111111", linewidths=0.7)
    ax_ndvi.set_title("NDVI Difference: 2026 - 2015", fontsize=13, weight="bold")

    for ax in [ax_before, ax_after, ax_overlay, ax_ndvi]:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(ndvi_plot, ax=ax_ndvi, fraction=0.046, pad=0.02)
    cbar.set_label("NDVI change", fontsize=9)

    draw_summary_panel(ax_summary, products)

    fig.subplots_adjust(top=0.92, bottom=0.035, left=0.035, right=0.985)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = create_figure()
    print(f"Saved explained change figure to {output_path}")
