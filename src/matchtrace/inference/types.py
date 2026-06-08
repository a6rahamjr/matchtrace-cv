from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)

    def clipped(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            max(0.0, min(self.x1, width - 1)),
            max(0.0, min(self.y1, height - 1)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )

    def as_int(self) -> tuple[int, int, int, int]:
        return tuple(round(value) for value in (self.x1, self.y1, self.x2, self.y2))

    def shifted(self, dx: float, dy: float) -> "BoundingBox":
        return BoundingBox(
            self.x1 + dx,
            self.y1 + dy,
            self.x2 + dx,
            self.y2 + dy,
        )

    def iou(self, other: "BoundingBox") -> float:
        intersection_width = max(
            0.0,
            min(self.x2, other.x2) - max(self.x1, other.x1),
        )
        intersection_height = max(
            0.0,
            min(self.y2, other.y2) - max(self.y1, other.y1),
        )
        intersection = intersection_width * intersection_height
        union = (
            self.width * self.height
            + other.width * other.height
            - intersection
        )
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None


@dataclass(frozen=True)
class PlayerObservation:
    detection: Detection
    team: int
    team_confidence: float
    pitch_position: tuple[float, float]
    speed_mps: float
    distance_m: float
    has_ball: bool
