import rasterio
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# LOAD RESNET MODEL
# -----------------------------

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# remove classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# FUNCTION: TIFF → IMAGE
# -----------------------------

def load_satellite_image(path):

    with rasterio.open(path) as src:
        band = src.read(1).astype(float)

    # normalize
    band = (band - np.min(band)) / (np.max(band) - np.min(band))

    band = (band * 255).astype(np.uint8)

    img = Image.fromarray(band)

    return img.convert("RGB")


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_features(img):

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    return features.flatten().numpy()


# -----------------------------
# LOAD SATELLITE DATA
# -----------------------------

img1 = load_satellite_image("sentinel_data/B04_2015.tiff")
img2 = load_satellite_image("sentinel_data/B04_2026.tiff")

# -----------------------------
# EXTRACT FEATURES
# -----------------------------

f1 = extract_features(img1)
f2 = extract_features(img2)

# -----------------------------
# FEATURE DIFFERENCE
# -----------------------------

difference = np.linalg.norm(f1 - f2)

print("\nDeep Learning Change Detection")
print("Feature Difference Score:", difference)

if difference > 5:
    print("Significant land change detected")
else:
    print("Minor land change detected")