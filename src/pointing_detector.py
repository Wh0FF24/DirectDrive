"""Pointing detection: MediaPipe pose + arm extension analysis.

Wraps MediaPipePoseExtractor and keypoint_utils to detect when a person
is pointing and extract the shoulder/wrist pixel coordinates for stereo
triangulation.
"""

from dataclasses import dataclass
import numpy as np

from mediapipe_extractor import MediaPipePoseExtractor
from keypoint_utils import (
    detect_pointing_arm,
    compute_arm_extension_ratio,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_INDEX, RIGHT_INDEX,
)


@dataclass
class PointingResult:
    side: str                   # "left" or "right"
    shoulder_px: np.ndarray     # (2,) pixel coords [x, y]
    wrist_px: np.ndarray        # (2,) pixel coords [x, y]
    index_px: np.ndarray        # (2,) index fingertip pixel coords
    confidence: float           # arm extension ratio [0, 1]
    landmarks: np.ndarray       # full (33, 4) for debugging
    landmarks_px: np.ndarray    # full (33, 2) pixel coords


class PointingDetector:
    """Detects pointing gestures from a single camera frame."""

    def __init__(self, extension_threshold: float = 0.55,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.extractor = MediaPipePoseExtractor(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.threshold = extension_threshold

    def detect(self, frame: np.ndarray) -> PointingResult | None:
        """Run pose detection and check for pointing gesture.

        Args:
            frame: BGR image from camera.

        Returns:
            PointingResult with shoulder/wrist pixel coords, or None.
        """
        result = self.extractor.extract(frame)
        if result is None:
            return None

        landmarks = result["landmarks"]       # (33, 4)
        landmarks_px = result["landmarks_px"] # (33, 2)

        # Detect which arm (if any) is extended enough
        side = detect_pointing_arm(landmarks, extension_threshold=self.threshold)
        if side is None:
            return None

        # Get pixel coordinates for the detected pointing arm
        if side == "right":
            shoulder_px = landmarks_px[RIGHT_SHOULDER]
            wrist_px = landmarks_px[RIGHT_WRIST]
            index_px = landmarks_px[RIGHT_INDEX]
        else:
            shoulder_px = landmarks_px[LEFT_SHOULDER]
            wrist_px = landmarks_px[LEFT_WRIST]
            index_px = landmarks_px[LEFT_INDEX]

        confidence = compute_arm_extension_ratio(landmarks, side)

        return PointingResult(
            side=side,
            shoulder_px=shoulder_px.copy(),
            wrist_px=wrist_px.copy(),
            index_px=index_px.copy(),
            confidence=confidence,
            landmarks=landmarks,
            landmarks_px=landmarks_px,
        )

    def close(self):
        self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
