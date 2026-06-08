from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from matchtrace.utils.config import configured_path


TEAM_RGB = np.asarray(((220, 45, 45), (45, 90, 220)), dtype=np.int16)
TEAM_BGR = ((45, 45, 220), (220, 90, 45))
FIELD_BGR = (52, 132, 61)
BALL_BGR = (0, 220, 255)


def _player_crop(
    rng: np.random.Generator,
    team: int,
    image_size: int,
) -> np.ndarray:
    image = np.full((image_size, image_size, 3), (45, 120, 55), dtype=np.int16)
    image += rng.normal(0, 8, image.shape).astype(np.int16)

    center_x = image_size // 2 + int(rng.integers(-3, 4))
    top = int(rng.integers(5, 9))
    half_width = int(rng.integers(6, 9))
    bottom = min(image_size - 5, top + int(rng.integers(15, 20)))

    color = TEAM_RGB[team] + rng.integers(-28, 29, size=3)
    color = np.clip(color, 0, 255)
    image[top:bottom, center_x - half_width : center_x + half_width] = color

    if rng.random() < 0.65:
        stripe_x = center_x + int(rng.integers(-half_width + 2, half_width - 1))
        image[top:bottom, stripe_x : stripe_x + 2] = np.clip(color + 45, 0, 255)

    skin = np.asarray((215, 165, 125), dtype=np.int16)
    head_radius = max(2, image_size // 10)
    cv2.circle(image, (center_x, top - head_radius), head_radius, skin.tolist(), -1)

    brightness = float(rng.uniform(0.72, 1.28))
    image = np.clip(image * brightness, 0, 255)
    image += rng.normal(0, 5, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _make_split(
    rng: np.random.Generator,
    count: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(count, dtype=np.int64) % 2
    rng.shuffle(labels)
    images = np.stack(
        [_player_crop(rng, int(label), image_size) for label in labels]
    )
    return images, labels


def generate_classification_dataset(
    output_path: str | Path,
    *,
    seed: int,
    image_size: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    train_images, train_labels = _make_split(rng, train_samples, image_size)
    val_images, val_labels = _make_split(rng, validation_samples, image_size)
    test_images, test_labels = _make_split(rng, test_samples, image_size)

    np.savez_compressed(
        output,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=val_images,
        validation_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
        class_names=np.asarray(("red", "blue")),
        seed=np.asarray(seed),
    )
    return output


def _bounce(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    phase = (value - lower) % (2 * span)
    return lower + (phase if phase <= span else 2 * span - phase)


def generate_demo_video(
    video_path: str | Path,
    truth_path: str | Path,
    *,
    seed: int,
    width: int,
    height: int,
    frames: int,
    fps: float,
    players_per_team: int,
) -> tuple[Path, Path]:
    if width < 320 or height < 180 or frames < 2 or fps <= 0:
        raise ValueError("Synthetic video dimensions, frames, and FPS are invalid.")

    output = Path(video_path)
    truth_output = Path(truth_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    truth_output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create synthetic video: {output}")

    rng = np.random.default_rng(seed)
    players: list[dict[str, Any]] = []
    margin_x, margin_y = 45, 45
    for team in (0, 1):
        for index in range(players_per_team):
            players.append(
                {
                    "id": team * players_per_team + index + 1,
                    "team": team,
                    "x0": float(rng.uniform(margin_x, width - margin_x)),
                    "y0": float(rng.uniform(margin_y, height - margin_y)),
                    "vx": float(rng.uniform(0.65, 1.65) * (-1 if team else 1)),
                    "vy": float(rng.uniform(-0.65, 0.65)),
                }
            )

    ground_truth: dict[str, Any] = {
        "seed": seed,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": [],
    }
    try:
        for frame_index in range(frames):
            frame = np.full((height, width, 3), FIELD_BGR, dtype=np.uint8)
            cv2.rectangle(frame, (18, 18), (width - 18, height - 18), (225, 225, 225), 2)
            cv2.line(
                frame,
                (width // 2, 18),
                (width // 2, height - 18),
                (225, 225, 225),
                2,
            )
            cv2.circle(
                frame,
                (width // 2, height // 2),
                min(width, height) // 7,
                (225, 225, 225),
                2,
            )

            frame_players = []
            player_centers: dict[int, tuple[int, int]] = {}
            for player in players:
                x = int(
                    _bounce(
                        player["x0"] + player["vx"] * frame_index,
                        margin_x,
                        width - margin_x,
                    )
                )
                y = int(
                    _bounce(
                        player["y0"] + player["vy"] * frame_index,
                        margin_y,
                        height - margin_y,
                    )
                )
                player_centers[player["id"]] = (x, y)
                team = int(player["team"])
                bbox = [x - 8, y - 14, x + 8, y + 14]
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    TEAM_BGR[team],
                    -1,
                )
                cv2.circle(frame, (x, y - 19), 5, (125, 170, 220), -1)
                cv2.line(frame, (x - 4, y + 14), (x - 6, y + 22), (30, 30, 30), 2)
                cv2.line(frame, (x + 4, y + 14), (x + 6, y + 22), (30, 30, 30), 2)
                frame_players.append(
                    {"id": player["id"], "team": team, "bbox": bbox}
                )

            possession_segment = frame_index // max(1, frames // 6)
            owner_team = possession_segment % 2
            owner_index = (possession_segment // 2) % players_per_team
            owner_slot = owner_team * players_per_team + owner_index
            owner = players[owner_slot]
            owner_x, owner_y = player_centers[owner["id"]]
            ball_center = (
                int(owner_x + 14 + 4 * np.sin(frame_index / 3)),
                int(owner_y + 10 + 3 * np.cos(frame_index / 4)),
            )
            cv2.circle(frame, ball_center, 5, BALL_BGR, -1)
            writer.write(frame)
            ground_truth["frames"].append(
                {
                    "index": frame_index,
                    "players": frame_players,
                    "ball": {"center": list(ball_center), "owner_id": owner["id"]},
                }
            )
    finally:
        writer.release()

    truth_output.write_text(
        json.dumps(ground_truth, indent=2),
        encoding="utf-8",
    )
    return output, truth_output


def generate_all(settings: dict[str, Any]) -> dict[str, str]:
    synthetic = settings["synthetic"]
    seed = int(settings["project"]["seed"])
    dataset_path = generate_classification_dataset(
        configured_path(settings, "dataset"),
        seed=seed,
        image_size=int(synthetic["image_size"]),
        train_samples=int(synthetic["train_samples"]),
        validation_samples=int(synthetic["validation_samples"]),
        test_samples=int(synthetic["test_samples"]),
    )
    video_path, truth_path = generate_demo_video(
        configured_path(settings, "demo_video"),
        configured_path(settings, "demo_truth"),
        seed=seed,
        width=int(synthetic["video_width"]),
        height=int(synthetic["video_height"]),
        frames=int(synthetic["video_frames"]),
        fps=float(synthetic["video_fps"]),
        players_per_team=int(synthetic["players_per_team"]),
    )
    return {
        "dataset": str(dataset_path),
        "demo_video": str(video_path),
        "ground_truth": str(truth_path),
    }
