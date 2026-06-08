from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from matchtrace.inference.types import BoundingBox, Detection
from matchtrace.utils.config import resolve_path


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for a BGR frame."""


class SyntheticColorDetector:
    def __init__(self, minimum_area: float = 45.0):
        self.minimum_area = minimum_area

    @staticmethod
    def _boxes(mask: np.ndarray, label: str, minimum_area: float) -> list[Detection]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if label == "player":
                box = BoundingBox(x - 2, y - 7, x + width + 2, y + height + 9)
            else:
                box = BoundingBox(x, y, x + width, y + height)
            detections.append(Detection(label, 1.0, box))
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        red = cv2.inRange(frame, (0, 0, 145), (130, 135, 255))
        blue = cv2.inRange(frame, (145, 25, 0), (255, 155, 145))
        ball = cv2.inRange(frame, (0, 165, 165), (90, 255, 255))
        player_mask = cv2.bitwise_or(red, blue)
        kernel = np.ones((3, 3), dtype=np.uint8)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, kernel)

        detections = self._boxes(player_mask, "player", self.minimum_area)
        detections.extend(self._boxes(ball, "ball", 20.0))
        return detections


class UltralyticsDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Detector model not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install the real-video extra to use Ultralytics: "
                "pip install -e .[real-video]"
            ) from exc

        self.model = YOLO(str(path))
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        names: dict[int, str] = result.names
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            source_label = names[class_id].lower()
            if source_label in {"player", "goalkeeper"}:
                label = "player"
            elif source_label == "ball":
                label = "ball"
            else:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf.item()),
                    bbox=BoundingBox(x1, y1, x2, y2),
                )
            )
        return detections


def create_detector(settings: dict[str, Any]) -> Detector:
    config = settings["detector"]
    backend = str(config["backend"]).lower()
    if backend == "synthetic_color":
        return SyntheticColorDetector()
    if backend == "ultralytics":
        model_path = str(config["model_path"]).strip()
        if not model_path:
            raise ValueError(
                "detector.model_path or MATCHTRACE_MODEL_PATH is required "
                "for the Ultralytics backend."
            )
        return UltralyticsDetector(
            resolve_path(settings, model_path),
            confidence=float(config["confidence"]),
        )
    raise ValueError(f"Unsupported detector backend: {backend}")
