from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any

from matchtrace.inference.geometry import PitchProjector
from matchtrace.inference.types import Detection, PlayerObservation


@dataclass
class _MovementState:
    smoothed_position: tuple[float, float] | None = None
    distance: float = 0.0
    speeds: deque[float] = field(default_factory=deque)
    rejected_updates: int = 0


class AnalyticsEngine:
    def __init__(self, settings: dict[str, Any], fps: float):
        if fps <= 0:
            raise ValueError("Frames per second must be positive.")
        config = settings["analytics"]
        self.fps = fps
        self.possession_distance = float(config["possession_distance_meters"])
        self.speed_window = int(config["speed_window_frames"])
        self.maximum_speed = float(config["maximum_speed_mps"])
        self.position_smoothing = float(config["position_smoothing_alpha"])
        self.maximum_jump_multiplier = float(config["maximum_jump_multiplier"])
        self.possession_hold_frames = int(config["possession_hold_frames"])
        self.possession_switch_margin = float(
            config["possession_switch_margin_meters"]
        )
        if not 0 < self.position_smoothing <= 1:
            raise ValueError("Position smoothing alpha must be in the range (0, 1].")
        if self.maximum_jump_multiplier < 1:
            raise ValueError("Maximum jump multiplier must be at least 1.")
        if self.possession_hold_frames < 0:
            raise ValueError("Possession hold frames cannot be negative.")
        self.movement: dict[int, _MovementState] = {}
        self.possession_frames: Counter[int] = Counter()
        self.current_possession_track: int | None = None
        self.current_possession_team: int | None = None
        self.possession_missing_frames = 0
        self.possession_switches = 0

    def _resolve_possession(
        self,
        positions: dict[int, tuple[float, float]],
        ball_position: tuple[float, float] | None,
        teams: dict[int, tuple[int, float]],
    ) -> tuple[int | None, int | None]:
        candidate_track: int | None = None
        candidate_team: int | None = None
        distances: dict[int, float] = {}
        if ball_position is not None and positions:
            distances = {
                track_id: hypot(
                    position[0] - ball_position[0],
                    position[1] - ball_position[1],
                )
                for track_id, position in positions.items()
            }
            nearest_track = min(distances, key=distances.get)
            if (
                distances[nearest_track] <= self.possession_distance
                and nearest_track in teams
            ):
                candidate_track = nearest_track
                candidate_team = teams[nearest_track][0]

        current_track = self.current_possession_track
        if (
            candidate_track is not None
            and current_track is not None
            and candidate_track != current_track
            and current_track in distances
            and current_track in teams
            and distances[current_track] <= self.possession_distance
            and distances[candidate_track] + self.possession_switch_margin
            >= distances[current_track]
        ):
            # Keep the current owner unless the new candidate is clearly closer.
            candidate_track = current_track
            candidate_team = teams[current_track][0]

        if candidate_track is not None and candidate_team is not None:
            if (
                self.current_possession_track is not None
                and candidate_track != self.current_possession_track
            ):
                self.possession_switches += 1
            self.current_possession_track = candidate_track
            self.current_possession_team = candidate_team
            self.possession_missing_frames = 0
        else:
            self.possession_missing_frames += 1
            if self.possession_missing_frames > self.possession_hold_frames:
                self.current_possession_track = None
                self.current_possession_team = None

        return self.current_possession_track, self.current_possession_team

    def update(
        self,
        players: list[Detection],
        ball: Detection | None,
        teams: dict[int, tuple[int, float]],
        projector: PitchProjector,
        camera_offset: tuple[float, float],
    ) -> tuple[list[PlayerObservation], int | None]:
        positions: dict[int, tuple[float, float]] = {}
        for player in players:
            if player.track_id is None:
                continue
            positions[player.track_id] = projector.project(
                player.bbox.foot,
                camera_offset,
            )

        ball_position = (
            projector.project(ball.bbox.center, camera_offset)
            if ball is not None
            else None
        )
        possession_track, possession_team = self._resolve_possession(
            positions,
            ball_position,
            teams,
        )

        self.possession_frames[
            possession_team if possession_team is not None else -1
        ] += 1

        observations = []
        for player in players:
            track_id = player.track_id
            if track_id is None or track_id not in teams:
                continue
            position = positions[track_id]
            state = self.movement.setdefault(track_id, _MovementState())
            speed = 0.0
            if state.smoothed_position is not None:
                smoothed_position = (
                    self.position_smoothing * position[0]
                    + (1.0 - self.position_smoothing)
                    * state.smoothed_position[0],
                    self.position_smoothing * position[1]
                    + (1.0 - self.position_smoothing)
                    * state.smoothed_position[1],
                )
                step = hypot(
                    smoothed_position[0] - state.smoothed_position[0],
                    smoothed_position[1] - state.smoothed_position[1],
                )
                maximum_step = (
                    self.maximum_speed
                    / self.fps
                    * self.maximum_jump_multiplier
                )
                if step <= maximum_step:
                    state.distance += step
                    speed = min(step * self.fps, self.maximum_speed)
                else:
                    state.rejected_updates += 1
                state.smoothed_position = smoothed_position
            else:
                state.smoothed_position = position
            state.speeds.append(speed)
            while len(state.speeds) > self.speed_window:
                state.speeds.popleft()
            smoothed_speed = sum(state.speeds) / len(state.speeds)
            team, confidence = teams[track_id]
            observations.append(
                PlayerObservation(
                    detection=player,
                    team=team,
                    team_confidence=confidence,
                    pitch_position=position,
                    speed_mps=smoothed_speed,
                    distance_m=state.distance,
                    has_ball=track_id == possession_track,
                )
            )
        return observations, possession_team

    def summary(self) -> dict[str, Any]:
        total = sum(self.possession_frames.values())
        possession = {
            "team_0": self.possession_frames[0] / total if total else 0.0,
            "team_1": self.possession_frames[1] / total if total else 0.0,
            "unknown": self.possession_frames[-1] / total if total else 0.0,
        }
        distances = {
            str(track_id): state.distance
            for track_id, state in sorted(self.movement.items())
        }
        rejected_updates = sum(
            state.rejected_updates for state in self.movement.values()
        )
        return {
            "possession": possession,
            "distance_by_track_m": distances,
            "quality": {
                "possession_switches": self.possession_switches,
                "rejected_movement_updates": rejected_updates,
            },
        }
