import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from unet_model import UNet


DATA_DIR = "sentinel_data"
OUTPUT_DIR = "outputs"
MODEL_PATH = "unet_model.pth"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "actual_deep_learning_unet_figure.png")
THRESHOLD = 0.5


def read_raw_rgb_bands(path):
    """Return channels exactly as the U-Net training code used them."""
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
    return image / (image.max() + 1e-10)


def to_display_rgb(raw_image):
    """Convert stored blue, green, red channels into red, green, blue display order."""
    rgb = np.dstack((raw_image[2], raw_image[1], raw_image[0]))
    p_low, p_high = np.percentile(rgb, (2, 98))
    return np.clip((rgb - p_low) / (p_high - p_low + 1e-10), 0, 1)


def resize_image(image, target_shape, interpolation=cv2.INTER_LINEAR):
    target_height, target_width = target_shape
    if image.shape[:2] == target_shape:
        return image
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def run_unet_inference():
    img15 = read_raw_rgb_bands(os.path.join(DATA_DIR, "2015_rgb.tif"))
    img26 = read_raw_rgb_bands(os.path.join(DATA_DIR, "2026_rgb.tif"))

    model_input = np.concatenate([img15, img26], axis=0)
    input_tensor = torch.tensor(model_input).unsqueeze(0).float()
    input_tensor = F.interpolate(input_tensor, size=(256, 256), mode="bilinear", align_corners=False)

    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probability_256 = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask_256 = probability_256 >= THRESHOLD

    display15 = to_display_rgb(img15)
    display26 = to_display_rgb(img26)

    probability_full = resize_image(probability_256, display26.shape[:2], interpolation=cv2.INTER_LINEAR)
    mask_full = resize_image(mask_256.astype(np.uint8), display26.shape[:2], interpolation=cv2.INTER_NEAREST).astype(bool)

    return {
        "display15": display15,
        "display26": display26,
        "probability_256": probability_256,
        "probability_full": probability_full,
        "mask_256": mask_256,
        "mask_full": mask_full,
        "input_shape": tuple(model_input.shape),
        "model_shape": tuple(input_tensor.shape),
    }


def make_overlay(rgb, mask):
    overlay = rgb.copy()
    red = np.array([1.0, 0.04, 0.02])
    overlay[mask] = 0.42 * overlay[mask] + 0.58 * red
    return overlay


def draw_summary(ax, result):
    probability = result["probability_256"]
    mask = result["mask_256"]

    changed_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    changed_percent = 100 * changed_pixels / max(total_pixels, 1)

    prob_min = float(probability.min())
    prob_mean = float(probability.mean())
    prob_max = float(probability.max())
    prob_p95 = float(np.percentile(probability, 95))

    ax.axis("off")
    ax.set_title("What The Deep Model Actually Did", loc="left", fontsize=15, weight="bold")

    rows = [
        ("Model", "U-Net segmentation network"),
        ("Input to model", f"{result['input_shape'][0]} channels: 2015 RGB + 2026 RGB"),
        ("Model resize", "input resized to 256 x 256"),
        ("Output", "one change probability per pixel"),
        ("Decision rule", f"change if probability >= {THRESHOLD:.2f}"),
        ("Predicted change", f"{changed_percent:.2f}% of pixels"),
        ("Changed pixels", f"{changed_pixels} of {total_pixels}"),
        ("Probability min", f"{prob_min:.4f}"),
        ("Probability mean", f"{prob_mean:.4f}"),
        ("Probability 95th pct", f"{prob_p95:.4f}"),
        ("Probability max", f"{prob_max:.4f}"),
    ]

    y = 0.91
    for label, value in rows:
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=9.8, color="#333333")
        ax.text(0.48, y, value, transform=ax.transAxes, fontsize=10.2, weight="bold", color="#111111")
        y -= 0.065

    if changed_pixels == 0:
        conclusion = (
            "Conclusion shown by the model: with the saved weights and a 0.50 threshold, "
            "the U-Net did not mark any pixel as changed. The heatmap still shows the raw "
            "confidence values, so you can see how close the model came to predicting change."
        )
    else:
        conclusion = (
            "Conclusion shown by the model: red overlay pixels are the areas where the U-Net "
            "probability crossed the 0.50 change threshold."
        )

    ax.text(
        0.02,
        0.05,
        conclusion,
        transform=ax.transAxes,
        fontsize=9.5,
        color="#333333",
        wrap=True,
    )


def create_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = run_unet_inference()

    overlay = make_overlay(result["display26"], result["mask_full"])

    fig = plt.figure(figsize=(17, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], height_ratios=[1, 1], wspace=0.09, hspace=0.2)

    ax_2015 = fig.add_subplot(grid[0, 0])
    ax_2026 = fig.add_subplot(grid[0, 1])
    ax_summary = fig.add_subplot(grid[:, 2])
    ax_prob = fig.add_subplot(grid[1, 0])
    ax_overlay = fig.add_subplot(grid[1, 1])

    fig.suptitle("Actual Deep Learning Output: U-Net Change Detection", fontsize=22, weight="bold", y=0.985)

    ax_2015.imshow(result["display15"])
    ax_2015.set_title("Input Part 1: 2015 RGB", fontsize=13, weight="bold")

    ax_2026.imshow(result["display26"])
    ax_2026.set_title("Input Part 2: 2026 RGB", fontsize=13, weight="bold")

    prob_plot = ax_prob.imshow(result["probability_256"], cmap="magma", vmin=0, vmax=1)
    ax_prob.set_title("Raw U-Net Probability Map", fontsize=13, weight="bold")
    cbar = fig.colorbar(prob_plot, ax=ax_prob, fraction=0.046, pad=0.02)
    cbar.set_label("Predicted probability of change", fontsize=9)

    ax_overlay.imshow(overlay)
    ax_overlay.set_title(f"Final AI Mask Overlay: threshold {THRESHOLD:.2f}", fontsize=13, weight="bold")

    if result["mask_full"].sum() == 0:
        ax_overlay.text(
            0.5,
            0.5,
            "No pixels crossed the 0.50 threshold",
            transform=ax_overlay.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color="white",
            bbox=dict(facecolor="black", alpha=0.68, edgecolor="none", boxstyle="round,pad=0.35"),
        )

    for ax in [ax_2015, ax_2026, ax_prob, ax_overlay]:
        ax.set_xticks([])
        ax.set_yticks([])

    draw_summary(ax_summary, result)

    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.035, right=0.985)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return OUTPUT_PATH


if __name__ == "__main__":
    output = create_figure()
    print(f"Saved actual deep learning figure to {output}")
