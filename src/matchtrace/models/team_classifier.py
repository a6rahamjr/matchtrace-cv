from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


class TinyTeamCNN(nn.Module):
    def __init__(self, channels: int = 24, classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(channels * 2, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Team-classifier checkpoint not found: {path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


class TeamClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        checkpoint = load_checkpoint(checkpoint_path)
        self.class_names = tuple(checkpoint["class_names"])
        self.image_size = int(checkpoint["image_size"])
        self.device = torch.device(device)
        self.model = TinyTeamCNN(
            channels=int(checkpoint["channels"]),
            classes=len(self.class_names),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, bgr_crop: np.ndarray) -> torch.Tensor:
        if bgr_crop.size == 0:
            raise ValueError("Cannot classify an empty player crop.")
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor

    def prepare(self, bgr_crop: np.ndarray) -> torch.Tensor:
        return self._prepare_image(bgr_crop).unsqueeze(0).to(self.device)

    def prepare_batch(self, bgr_crops: list[np.ndarray]) -> torch.Tensor:
        if not bgr_crops:
            raise ValueError("At least one player crop is required.")
        return torch.stack(
            [self._prepare_image(crop) for crop in bgr_crops]
        ).to(self.device)

    @torch.inference_mode()
    def predict(self, bgr_crop: np.ndarray) -> tuple[int, float]:
        return self.predict_batch([bgr_crop])[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        bgr_crops: list[np.ndarray],
    ) -> list[tuple[int, float]]:
        probabilities = torch.softmax(
            self.model(self.prepare_batch(bgr_crops)),
            dim=1,
        )
        confidences, indices = probabilities.max(dim=1)
        return [
            (int(index), float(confidence))
            for index, confidence in zip(
                indices.cpu().tolist(),
                confidences.cpu().tolist(),
                strict=True,
            )
        ]
