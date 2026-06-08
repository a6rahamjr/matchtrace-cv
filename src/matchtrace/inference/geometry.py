from __future__ import annotations

from typing import Any


class PitchProjector:
    def __init__(self, frame_size: tuple[int, int], settings: dict[str, Any]):
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive.")
        self.frame_width = width
        self.frame_height = height
        self.pitch_width = float(settings["pitch"]["width_meters"])
        self.pitch_height = float(settings["pitch"]["height_meters"])
        if self.pitch_width <= 0 or self.pitch_height <= 0:
            raise ValueError("Pitch dimensions must be positive.")

    def project(
        self,
        point: tuple[float, float],
        camera_offset: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        x = min(max(point[0] - camera_offset[0], 0.0), self.frame_width)
        y = min(max(point[1] - camera_offset[1], 0.0), self.frame_height)
        return (
            x / self.frame_width * self.pitch_width,
            y / self.frame_height * self.pitch_height,
        )

