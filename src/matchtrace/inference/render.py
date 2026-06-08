from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from matchtrace.inference.types import Detection, PlayerObservation


TEAM_COLORS = ((40, 40, 235), (235, 105, 40))


def _draw_tag(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )
    x = max(0, min(origin[0], image.shape[1] - text_width - 8))
    y = max(text_height + 5, min(origin[1], image.shape[0] - baseline - 3))
    cv2.rectangle(
        image,
        (x, y - text_height - 5),
        (x + text_width + 7, y + baseline + 2),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 2),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_frame(
    frame: np.ndarray,
    players: list[PlayerObservation],
    ball: Detection | None,
    possession: dict[str, float],
    settings: dict[str, Any],
) -> np.ndarray:
    output = frame.copy()
    render_config = settings["render"]
    for player in players:
        x1, y1, x2, y2 = player.detection.bbox.as_int()
        color = TEAM_COLORS[player.team]
        thickness = 4 if player.has_ball else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        title = f"T{player.team + 1}"
        if render_config["show_track_id"]:
            title += f"  #{player.detection.track_id}"
        _draw_tag(output, title, (x1, y1 - 4), color)

        metrics = []
        if render_config["show_speed"]:
            metrics.append(f"{player.speed_mps:.1f}m/s")
        if render_config["show_distance"]:
            metrics.append(f"{player.distance_m:.1f}m")
        if metrics:
            _draw_tag(output, "  ".join(metrics), (x1, y2 + 17), color)
        if player.has_ball:
            cv2.circle(output, ((x1 + x2) // 2, y1 - 12), 6, (0, 255, 255), -1)

    if ball is not None:
        center = tuple(round(value) for value in ball.bbox.center)
        cv2.circle(output, center, 7, (0, 230, 255), 2)

    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (300, 72), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0, output)
    cv2.putText(
        output,
        f"Possession  T1 {possession['team_0'] * 100:5.1f}%",
        (24, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEAM_COLORS[0],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"            T2 {possession['team_1'] * 100:5.1f}%",
        (24, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEAM_COLORS[1],
        2,
        cv2.LINE_AA,
    )
    return output
