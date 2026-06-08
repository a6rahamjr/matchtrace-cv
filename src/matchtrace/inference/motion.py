from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class CameraMotionEstimator:
    def __init__(self, settings: dict[str, Any]):
        config = settings["motion"]
        self.enabled = bool(config["enabled"])
        self.feature_params = {
            "maxCorners": int(config["max_corners"]),
            "qualityLevel": float(config["quality_level"]),
            "minDistance": float(config["min_distance"]),
            "blockSize": 7,
        }
        self.previous_gray: np.ndarray | None = None
        self.previous_points: np.ndarray | None = None
        self.cumulative = np.zeros(2, dtype=np.float64)

    def _features(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        if not self.enabled:
            return 0.0, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            self.previous_points = self._features(gray)
            return 0.0, 0.0

        if self.previous_points is None or len(self.previous_points) < 4:
            self.previous_points = self._features(self.previous_gray)

        movement = np.zeros(2, dtype=np.float64)
        if self.previous_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                self.previous_points,
                None,
            )
            if next_points is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                old = self.previous_points.reshape(-1, 2)[valid]
                new = next_points.reshape(-1, 2)[valid]
                if len(old):
                    movement = np.median(new - old, axis=0)

        self.cumulative += movement
        self.previous_gray = gray
        self.previous_points = self._features(gray)
        return float(self.cumulative[0]), float(self.cumulative[1])

