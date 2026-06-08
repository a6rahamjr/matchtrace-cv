from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {self.path}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            self.close()
            raise ValueError(f"Video metadata is invalid: {self.path}")

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        while True:
            success, frame = self._capture.read()
            if not success:
                break
            yield index, frame
            index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        codec: str = "mp4v",
    ):
        if len(codec) != 4:
            raise ValueError("Video codec must contain exactly four characters.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise ValueError(f"Could not create output video: {self.path}")
        self.size = size

    def write(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (width, height) != self.size:
            raise ValueError(
                f"Frame size {(width, height)} does not match writer {self.size}."
            )
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

