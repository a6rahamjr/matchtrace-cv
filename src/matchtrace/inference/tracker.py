from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

from matchtrace.inference.types import BoundingBox, Detection


@dataclass
class _Track:
    bbox: BoundingBox
    velocity: tuple[float, float] = (0.0, 0.0)
    missed: int = 0
    hits: int = 1


class MotionAwareTracker:
    def __init__(
        self,
        max_distance: float = 70.0,
        max_age: int = 10,
        distance_weight: float = 0.7,
        iou_weight: float = 0.3,
        velocity_smoothing: float = 0.65,
    ):
        if max_distance <= 0 or max_age < 0:
            raise ValueError("Tracker distance must be positive and age non-negative.")
        if distance_weight < 0 or iou_weight < 0:
            raise ValueError("Tracker association weights cannot be negative.")
        if distance_weight + iou_weight <= 0:
            raise ValueError("At least one tracker association weight must be positive.")
        if not 0 <= velocity_smoothing < 1:
            raise ValueError("Velocity smoothing must be in the range [0, 1).")
        self.max_distance = max_distance
        self.max_age = max_age
        weight_total = distance_weight + iou_weight
        self.distance_weight = distance_weight / weight_total
        self.iou_weight = iou_weight / weight_total
        self.velocity_smoothing = velocity_smoothing
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    @staticmethod
    def _predicted_bbox(track: _Track) -> BoundingBox:
        return track.bbox.shifted(
            track.velocity[0] * track.missed,
            track.velocity[1] * track.missed,
        )

    def update(self, detections: list[Detection]) -> list[Detection]:
        for track in self._tracks.values():
            track.missed += 1

        candidates = []
        for track_id, track in self._tracks.items():
            predicted_bbox = self._predicted_bbox(track)
            track_x, track_y = predicted_bbox.center
            distance_gate = self.max_distance * min(2.0, 1.0 + 0.2 * track.missed)
            for detection_index, detection in enumerate(detections):
                detection_x, detection_y = detection.bbox.center
                distance = hypot(track_x - detection_x, track_y - detection_y)
                if distance <= distance_gate:
                    # Distance handles fast motion; IoU favors spatial continuity.
                    cost = (
                        self.distance_weight * (distance / distance_gate)
                        + self.iou_weight
                        * (1.0 - predicted_bbox.iou(detection.bbox))
                    )
                    candidates.append((cost, track_id, detection_index))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_id, detection_index in sorted(candidates):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            assignments[detection_index] = track_id
            track = self._tracks[track_id]
            old_x, old_y = track.bbox.center
            new_x, new_y = detections[detection_index].bbox.center
            elapsed_frames = max(1, track.missed)
            measured_velocity = (
                (new_x - old_x) / elapsed_frames,
                (new_y - old_y) / elapsed_frames,
            )
            smoothing = self.velocity_smoothing
            track.velocity = (
                smoothing * track.velocity[0]
                + (1.0 - smoothing) * measured_velocity[0],
                smoothing * track.velocity[1]
                + (1.0 - smoothing) * measured_velocity[1],
            )
            track.bbox = detections[detection_index].bbox
            track.missed = 0
            track.hits += 1

        for detection_index, detection in enumerate(detections):
            if detection_index in assignments:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(detection.bbox)
            assignments[detection_index] = track_id

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed > self.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]

        return [
            replace(detection, track_id=assignments[index])
            for index, detection in enumerate(detections)
        ]
