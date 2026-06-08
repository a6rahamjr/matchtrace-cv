from __future__ import annotations

import copy
import unittest

import numpy as np

from matchtrace.inference.analytics import AnalyticsEngine
from matchtrace.inference.geometry import PitchProjector
from matchtrace.inference.pipeline import TrackTeamVotes
from matchtrace.inference.tracker import MotionAwareTracker
from matchtrace.inference.types import BoundingBox, Detection
from matchtrace.utils.config import load_settings


def player_detection(
    center_x: float,
    center_y: float = 20.0,
    track_id: int | None = None,
) -> Detection:
    return Detection(
        label="player",
        confidence=1.0,
        bbox=BoundingBox(
            center_x - 5,
            center_y - 10,
            center_x + 5,
            center_y + 10,
        ),
        track_id=track_id,
    )


class FakeBatchClassifier:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[int, float]]:
        self.batch_sizes.append(len(crops))
        return [(index % 2, 0.9) for index in range(len(crops))]


class RuntimeImprovementTests(unittest.TestCase):
    def test_motion_tracker_recovers_identity_after_occlusion(self) -> None:
        tracker = MotionAwareTracker(
            max_distance=25,
            max_age=3,
            distance_weight=1.0,
            iou_weight=0.0,
            velocity_smoothing=0.0,
        )
        first_id = tracker.update([player_detection(10)])[0].track_id
        second_id = tracker.update([player_detection(30)])[0].track_id
        tracker.update([])
        recovered_id = tracker.update([player_detection(70)])[0].track_id

        self.assertEqual(first_id, second_id)
        self.assertEqual(first_id, recovered_id)

    def test_team_classifier_uses_one_batch_per_frame(self) -> None:
        classifier = FakeBatchClassifier()
        cache = TrackTeamVotes(classifier, maximum_votes=5)
        frame = np.zeros((60, 120, 3), dtype=np.uint8)
        detections = [
            player_detection(20, track_id=1),
            player_detection(55, track_id=2),
            player_detection(90, track_id=3),
        ]

        teams = cache.update(frame, detections)

        self.assertEqual(classifier.batch_sizes, [3])
        self.assertEqual(set(teams), {1, 2, 3})

    def test_analytics_holds_possession_and_rejects_large_jump(self) -> None:
        settings = copy.deepcopy(load_settings())
        settings["analytics"].update(
            {
                "position_smoothing_alpha": 1.0,
                "possession_hold_frames": 2,
                "maximum_jump_multiplier": 1.0,
            }
        )
        settings["pitch"].update(
            {"width_meters": 100.0, "height_meters": 100.0}
        )
        projector = PitchProjector((100, 100), settings)
        analytics = AnalyticsEngine(settings, fps=10.0)
        teams = {1: (0, 1.0)}
        player = player_detection(10, center_y=10, track_id=1)
        ball = Detection(
            "ball",
            1.0,
            BoundingBox(9, 18, 11, 20),
        )

        _, possession = analytics.update(
            [player],
            ball,
            teams,
            projector,
            (0.0, 0.0),
        )
        self.assertEqual(possession, 0)

        moved = player_detection(11, center_y=10, track_id=1)
        _, possession = analytics.update(
            [moved],
            None,
            teams,
            projector,
            (0.0, 0.0),
        )
        self.assertEqual(possession, 0)

        jumped = player_detection(90, center_y=10, track_id=1)
        analytics.update(
            [jumped],
            None,
            teams,
            projector,
            (0.0, 0.0),
        )
        _, possession = analytics.update(
            [jumped],
            None,
            teams,
            projector,
            (0.0, 0.0),
        )

        summary = analytics.summary()
        self.assertIsNone(possession)
        self.assertEqual(summary["quality"]["rejected_movement_updates"], 1)
        self.assertAlmostEqual(summary["distance_by_track_m"]["1"], 1.0)


if __name__ == "__main__":
    unittest.main()
