from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from matchtrace import __version__
from matchtrace.inference.analytics import AnalyticsEngine
from matchtrace.inference.detectors import create_detector
from matchtrace.inference.geometry import PitchProjector
from matchtrace.inference.motion import CameraMotionEstimator
from matchtrace.inference.render import render_frame
from matchtrace.inference.tracker import MotionAwareTracker
from matchtrace.inference.types import Detection
from matchtrace.models.team_classifier import TeamClassifier
from matchtrace.utils.config import configured_path
from matchtrace.utils.fingerprint import file_sha256, settings_sha256
from matchtrace.utils.logging import configure_logging
from matchtrace.utils.video import VideoReader, VideoWriter


@dataclass(frozen=True)
class AnalysisPaths:
    source: Path
    video: Path
    summary: Path

    @classmethod
    def from_settings(
        cls,
        settings: dict[str, Any],
        input_path: str | Path | None,
        output_path: str | Path | None,
        summary_path: str | Path | None,
    ) -> "AnalysisPaths":
        source = (
            Path(input_path)
            if input_path is not None
            else configured_path(settings, "demo_video")
        ).resolve()
        video = (
            Path(output_path)
            if output_path is not None
            else configured_path(settings, "output_video")
        ).resolve()
        summary = (
            Path(summary_path)
            if summary_path is not None
            else configured_path(settings, "run_summary")
        ).resolve()

        if len({source, video, summary}) != 3:
            raise ValueError(
                "Input video, output video, and summary JSON must use "
                "distinct paths."
            )
        return cls(source=source, video=video, summary=summary)


class TrackTeamVotes:
    def __init__(self, classifier: TeamClassifier, maximum_votes: int = 5):
        self.classifier = classifier
        self.maximum_votes = maximum_votes
        self.votes: dict[int, Counter[int]] = defaultdict(Counter)
        self.observations: Counter[int] = Counter()

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> dict[int, tuple[int, float]]:
        height, width = frame.shape[:2]
        track_ids = []
        crops = []
        for detection in detections:
            if detection.track_id is None:
                continue
            if self.observations[detection.track_id] >= self.maximum_votes:
                continue
            box = detection.bbox.clipped(width, height)
            x1, y1, x2, y2 = box.as_int()
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            track_ids.append(detection.track_id)
            crops.append(crop)

        if crops:
            predictions = self.classifier.predict_batch(crops)
            for track_id, (team, confidence) in zip(
                track_ids,
                predictions,
                strict=True,
            ):
                self.votes[track_id][team] += confidence
                self.observations[track_id] += 1

        result = {}
        for track_id, votes in self.votes.items():
            team, winning_score = votes.most_common(1)[0]
            total_score = sum(votes.values())
            result[track_id] = (team, winning_score / total_score)
        return result


class VideoAnalyzer:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.checkpoint = configured_path(settings, "model")
        self.logger = configure_logging(configured_path(settings, "log"))

    def _reset_runtime(self) -> None:
        self.detector = create_detector(self.settings)
        self.team_votes = TrackTeamVotes(TeamClassifier(self.checkpoint))
        tracker = self.settings["tracker"]
        self.tracker = MotionAwareTracker(
            max_distance=float(tracker["max_distance_pixels"]),
            max_age=int(tracker["max_age_frames"]),
            distance_weight=float(tracker["distance_weight"]),
            iou_weight=float(tracker["iou_weight"]),
            velocity_smoothing=float(tracker["velocity_smoothing"]),
        )
        self.motion = CameraMotionEstimator(self.settings)

    def _process_frame(
        self,
        frame: np.ndarray,
        projector: PitchProjector,
        analytics: AnalyticsEngine,
    ) -> np.ndarray:
        detections = self.detector.detect(frame)
        players = [
            detection for detection in detections if detection.label == "player"
        ]
        balls = [
            detection for detection in detections if detection.label == "ball"
        ]

        tracked_players = self.tracker.update(players)
        teams = self.team_votes.update(frame, tracked_players)
        ball = max(balls, key=lambda item: item.confidence) if balls else None
        observations, _ = analytics.update(
            tracked_players,
            ball,
            teams,
            projector,
            self.motion.update(frame),
        )
        return render_frame(
            frame,
            observations,
            ball,
            analytics.summary()["possession"],
            self.settings,
        )

    def _fingerprints(self, paths: AnalysisPaths) -> dict[str, str]:
        fingerprints = {
            "settings_sha256": settings_sha256(self.settings),
            "model_sha256": file_sha256(self.checkpoint),
        }
        if bool(self.settings["audit"]["hash_input"]):
            fingerprints["input_sha256"] = file_sha256(paths.source)
        if bool(self.settings["audit"]["hash_output"]):
            fingerprints["output_sha256"] = file_sha256(paths.video)
        return fingerprints

    def run(self, paths: AnalysisPaths) -> dict[str, Any]:
        self._reset_runtime()
        started = time.perf_counter()
        frames_written = 0

        with VideoReader(paths.source) as reader:
            projector = PitchProjector((reader.width, reader.height), self.settings)
            analytics = AnalyticsEngine(self.settings, reader.fps)
            with VideoWriter(
                paths.video,
                reader.fps,
                (reader.width, reader.height),
                str(self.settings["render"]["codec"]),
            ) as writer:
                for _, frame in reader:
                    writer.write(self._process_frame(frame, projector, analytics))
                    frames_written += 1

            video_metadata = {
                "expected_frames": reader.frame_count,
                "fps": reader.fps,
                "resolution": [reader.width, reader.height],
            }

        elapsed = time.perf_counter() - started
        summary = {
            "project": self.settings["project"]["name"],
            "version": __version__,
            "detector_backend": self.settings["detector"]["backend"],
            "input": str(paths.source),
            "output": str(paths.video),
            "frames": frames_written,
            **video_metadata,
            "elapsed_seconds": elapsed,
            "throughput_fps": frames_written / elapsed if elapsed else 0.0,
            "fingerprints": self._fingerprints(paths),
            **analytics.summary(),
        }
        paths.summary.parent.mkdir(parents=True, exist_ok=True)
        paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.logger.info(
            "Analyzed %s frames at %.2f frames/second",
            frames_written,
            summary["throughput_fps"],
        )
        return summary


def analyze_video(
    settings: dict[str, Any],
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = AnalysisPaths.from_settings(
        settings,
        input_path,
        output_path,
        summary_path,
    )
    return VideoAnalyzer(settings).run(paths)
