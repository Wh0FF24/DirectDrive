"""MediaPipe-based pose extraction with structured keypoint output.

Copied from spatial-intent/src/pose/mediapipe_extractor.py with imports
adjusted for CyberTruck flat module layout.
"""

import numpy as np
import cv2
import mediapipe as mp


# MediaPipe landmark indices for commonly used body keypoints
LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# Upper body subset relevant for pointing/gesture analysis
UPPER_BODY_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


class MediaPipePoseExtractor:
    """Extracts human pose keypoints from RGB frames using MediaPipe Pose."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract(self, frame_bgr: np.ndarray) -> dict | None:
        """Extract pose keypoints from a BGR frame.

        Returns:
            Dictionary with 'landmarks' (33,4), 'landmarks_px' (33,2), 'image_size'
            or None if no pose detected.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks is None:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )

        landmarks_px = np.zeros((33, 2), dtype=np.float32)
        landmarks_px[:, 0] = landmarks[:, 0] * w
        landmarks_px[:, 1] = landmarks[:, 1] * h

        return {
            "landmarks": landmarks,
            "landmarks_px": landmarks_px,
            "image_size": (h, w),
        }

    def draw_landmarks(self, frame_bgr: np.ndarray, draw_connections: bool = True) -> np.ndarray:
        """Draw pose landmarks on the frame. Returns annotated copy."""
        annotated = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            connections = self.mp_pose.POSE_CONNECTIONS if draw_connections else None
            self.mp_draw.draw_landmarks(annotated, results.pose_landmarks, connections)

        return annotated

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
