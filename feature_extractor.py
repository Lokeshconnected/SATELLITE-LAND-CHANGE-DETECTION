import os
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


ImageInput = Union[str, Image.Image]


class FeatureExtractor:
    

    def __init__(self, image_size: Tuple[int, int] = (224, 224), threshold: float = 0.35):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.threshold = threshold

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        self.feature_backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.feature_backbone.eval()
        self.feature_backbone.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_image(self, image_input: ImageInput) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        return Image.open(image_input).convert("RGB")

    def _prepare_tensor(self, image_input: ImageInput) -> torch.Tensor:
        image = self._load_image(image_input)
        return self.transform(image).unsqueeze(0).to(self.device)

    def _extract_feature_map(self, image_input: ImageInput) -> torch.Tensor:
        image_tensor = self._prepare_tensor(image_input)
        with torch.no_grad():
            feature_map = self.feature_backbone(image_tensor)
        return feature_map

    def extract_features(self, image_input: ImageInput) -> torch.Tensor:
        

        feature_map = self._extract_feature_map(image_input)
        pooled = self.pool(feature_map)
        return pooled.flatten(start_dim=1)

    def similarity_score(self, image_a: ImageInput, image_b: ImageInput) -> float:
        

        features_a = self.extract_features(image_a)
        features_b = self.extract_features(image_b)
        score = F.cosine_similarity(features_a, features_b).item()
        return float(score)

    def generate_change_heatmap(self, image_a: ImageInput, image_b: ImageInput) -> np.ndarray:
       

        fmap_a = self._extract_feature_map(image_a)
        fmap_b = self._extract_feature_map(image_b)

        diff_map = torch.mean(torch.abs(fmap_a - fmap_b), dim=1, keepdim=True)
        diff_map = F.interpolate(
            diff_map,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        heatmap = diff_map.squeeze().detach().cpu().numpy()
        heatmap = heatmap - heatmap.min()
        max_value = heatmap.max()
        if max_value > 0:
            heatmap = heatmap / max_value
        return heatmap.astype(np.float32)

    def generate_change_mask(self, image_a: ImageInput, image_b: ImageInput) -> np.ndarray:
        

        heatmap = self.generate_change_heatmap(image_a, image_b)
        return (heatmap >= self.threshold).astype(np.uint8)

    def change_percentage(self, image_a: ImageInput, image_b: ImageInput) -> float:
        

        mask = self.generate_change_mask(image_a, image_b)
        return float(mask.mean() * 100.0)

    def analyze_pair(self, image_a: ImageInput, image_b: ImageInput) -> Dict[str, np.ndarray]:
        

        before_image = np.array(self._load_image(image_a).resize(self.image_size))
        after_image = np.array(self._load_image(image_b).resize(self.image_size))

        heatmap = self.generate_change_heatmap(image_a, image_b)
        mask = (heatmap >= self.threshold).astype(np.uint8)
        similarity = self.similarity_score(image_a, image_b)
        change_pct = float(mask.mean() * 100.0)

        overlay = after_image.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + (heatmap * 255).astype(np.uint8), 0, 255)
        overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.45 * heatmap), 0, 255).astype(np.uint8)
        overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - 0.65 * heatmap), 0, 255).astype(np.uint8)

        mask_rgb = np.zeros_like(after_image)
        mask_rgb[..., 0] = mask * 255

        comparison_panel = np.concatenate(
            [before_image, after_image, overlay, mask_rgb],
            axis=1,
        )

        return {
            "before_image": before_image,
            "after_image": after_image,
            "heatmap": heatmap,
            "change_mask": mask,
            "overlay": overlay.astype(np.uint8),
            "comparison_panel": comparison_panel.astype(np.uint8),
            "similarity_score": np.array([similarity], dtype=np.float32),
            "change_percentage": np.array([change_pct], dtype=np.float32),
        }

    def save_analysis_outputs(
        self,
        image_a: ImageInput,
        image_b: ImageInput,
        output_dir: str = "outputs",
        prefix: str = "deep_change",
    ) -> Dict[str, Union[str, float]]:
        """
        Save structured visual outputs to disk for demos and review.
        """

        os.makedirs(output_dir, exist_ok=True)
        result = self.analyze_pair(image_a, image_b)

        heatmap_path = os.path.join(output_dir, f"{prefix}_heatmap.png")
        mask_path = os.path.join(output_dir, f"{prefix}_mask.png")
        overlay_path = os.path.join(output_dir, f"{prefix}_overlay.png")
        panel_path = os.path.join(output_dir, f"{prefix}_panel.png")

        Image.fromarray((result["heatmap"] * 255).astype(np.uint8)).save(heatmap_path)
        Image.fromarray((result["change_mask"] * 255).astype(np.uint8)).save(mask_path)
        Image.fromarray(result["overlay"]).save(overlay_path)
        Image.fromarray(result["comparison_panel"]).save(panel_path)

        return {
            "heatmap_path": heatmap_path,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "panel_path": panel_path,
            "similarity_score": float(result["similarity_score"][0]),
            "change_percentage": float(result["change_percentage"][0]),
        }


if __name__ == "__main__":
    extractor = FeatureExtractor()
    outputs = extractor.save_analysis_outputs(
        "sentinel_data/2015.png",
        "sentinel_data/2026.png",
    )

    print("Deep feature similarity score:", round(outputs["similarity_score"], 4))
    print("Estimated change percentage:", round(outputs["change_percentage"], 2), "%")
    print("Heatmap:", outputs["heatmap_path"])
    print("Mask:", outputs["mask_path"])
    print("Overlay:", outputs["overlay_path"])
    print("Comparison panel:", outputs["panel_path"])
