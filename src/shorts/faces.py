"""OpenCV YuNet face detection on frames downscaled to 960 px wide.

MediaPipe's short-range model misses the small faces in a podcast wide shot; YuNet doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .assets import yunet_path

ANALYSIS_W = 960

try:  # silence "Targets are not supported by the new graph engine" spam
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def analysis_size(src_w: int, src_h: int) -> tuple[int, int]:
    h = int(round(ANALYSIS_W * src_h / src_w / 2)) * 2
    return ANALYSIS_W, h


@dataclass
class Face:
    x: float  # analysis-frame pixels
    y: float
    w: float
    h: float
    score: float
    mouth: tuple[float, float, float, float]  # right corner x,y, left corner x,y

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class FaceDetector:
    def __init__(self, width: int, height: int, score: float = 0.6):
        self.det = cv2.FaceDetectorYN.create(str(yunet_path()), "", (width, height), score, 0.3, 5000)
        self.min_size = max(10, width // 80)

    def __call__(self, frame_bgr: np.ndarray) -> list[Face]:
        _, faces = self.det.detect(frame_bgr)
        out: list[Face] = []
        if faces is None:
            return out
        for f in faces:
            if f[2] < self.min_size:
                continue
            out.append(Face(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14]),
                            (float(f[10]), float(f[11]), float(f[12]), float(f[13]))))
        return out


def mouth_box(face: Face, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = min(face.mouth[0], face.mouth[2]) - 0.1 * face.w
    x2 = max(face.mouth[0], face.mouth[2]) + 0.1 * face.w
    my = (face.mouth[1] + face.mouth[3]) / 2
    y1, y2 = my - 0.12 * face.h, my + 0.2 * face.h
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(x2) + 1), min(h, int(y2) + 1)
    return xi1, yi1, xi2, yi2


def mouth_motion(face: Face, gray: np.ndarray, prev_gray: np.ndarray | None) -> float:
    """Mean abs difference in the mouth region vs the previous frame (fallback speaker cue only)."""
    if prev_gray is None:
        return 0.0
    x1, y1, x2, y2 = mouth_box(face, gray.shape[1], gray.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = gray[y1:y2, x1:x2].astype(np.int16)
    b = prev_gray[y1:y2, x1:x2].astype(np.int16)
    return float(np.abs(a - b).mean())
