"""Ground target estimation using stereo triangulation.

Triangulates shoulder and wrist from both camera views to get 3D positions,
computes the pointing ray, and intersects it with the ground plane.
"""

from dataclasses import dataclass
import numpy as np

from stereo_vision import StereoVision
from pointing_detector import PointingResult
from geometry import normalize_vector, ray_plane_intersection


@dataclass
class GroundTarget:
    point_3d: np.ndarray    # (3,) XYZ in camera frame (centimeters)
    distance: float         # straight-line ground distance from camera (cm)
    heading_angle: float    # degrees: positive = target is to the right


class GroundTargetEstimator:
    """Estimates where a pointed-at ground location is in 3D."""

    def __init__(self, stereo: StereoVision,
                 ground_y: float = 0.0,
                 smooth_alpha: float = 0.5,
                 stable_frames_needed: int = 5):
        """
        Args:
            stereo: Initialized StereoVision instance.
            ground_y: Ground plane Y coordinate in camera frame (cm).
                      Positive Y is downward in OpenCV convention.
            smooth_alpha: EMA smoothing factor for target position.
            stable_frames_needed: Consecutive detections needed before "stable".
        """
        self.stereo = stereo
        # Ground plane: normal pointing up (negative Y in OpenCV convention)
        # The ground is at some Y value below the camera
        self.ground_normal = np.array([0.0, -1.0, 0.0])
        self.ground_point = np.array([0.0, ground_y, 0.0])

        self.smooth_alpha = smooth_alpha
        self.stable_frames_needed = stable_frames_needed

        self._smooth_target = None
        self._stable_count = 0

    def estimate(self, left_frame: np.ndarray, right_frame: np.ndarray,
                 pointing_left: PointingResult,
                 pointing_right: PointingResult) -> GroundTarget | None:
        """Estimate 3D ground target from stereo pointing detections.

        Args:
            left_frame: Left camera BGR frame (for potential future use).
            right_frame: Right camera BGR frame.
            pointing_left: PointingResult from left camera.
            pointing_right: PointingResult from right camera.

        Returns:
            GroundTarget or None if estimation fails.
        """
        # Triangulate shoulder and wrist using pixel coords from both views
        pts_left = np.array([
            pointing_left.shoulder_px,
            pointing_left.wrist_px,
        ])
        pts_right = np.array([
            pointing_right.shoulder_px,
            pointing_right.wrist_px,
        ])

        try:
            pts_3d = self.stereo.triangulate(pts_left, pts_right)  # (2, 3)
        except Exception:
            self._stable_count = 0
            return None

        shoulder_3d = pts_3d[0]
        wrist_3d = pts_3d[1]

        # Sanity check: points should be in front of camera (positive Z)
        if shoulder_3d[2] < 10 or wrist_3d[2] < 10:
            self._stable_count = 0
            return None

        # Pointing direction: shoulder -> wrist, extended to ground
        direction = normalize_vector(wrist_3d - shoulder_3d)

        # The direction should point somewhat downward (positive Y in OpenCV)
        # to intersect the ground. If pointing upward, no ground hit.
        if direction[1] < 0.01:  # Not pointing down enough
            self._stable_count = 0
            return None

        # Ray-plane intersection
        hit = ray_plane_intersection(
            shoulder_3d, direction,
            self.ground_point, self.ground_normal,
        )
        if hit is None:
            self._stable_count = 0
            return None

        # Sanity: hit should be in front of camera and within reasonable range
        if hit[2] < 0 or np.linalg.norm(hit) > 2000:  # > 20 meters
            self._stable_count = 0
            return None

        # EMA smoothing
        if self._smooth_target is None:
            self._smooth_target = hit.copy()
        else:
            self._smooth_target = (self.smooth_alpha * hit +
                                   (1 - self.smooth_alpha) * self._smooth_target)

        # Ground distance (X-Z plane, ignoring Y)
        distance = np.sqrt(self._smooth_target[0]**2 + self._smooth_target[2]**2)

        # Heading angle: positive = target is to the right
        heading = np.degrees(np.arctan2(self._smooth_target[0],
                                         self._smooth_target[2]))

        # Stability tracking
        self._stable_count += 1

        return GroundTarget(
            point_3d=self._smooth_target.copy(),
            distance=float(distance),
            heading_angle=float(heading),
        )

    @property
    def is_stable(self) -> bool:
        """True if we've had enough consecutive detections."""
        return self._stable_count >= self.stable_frames_needed

    def reset(self):
        """Reset smoothing and stability state."""
        self._smooth_target = None
        self._stable_count = 0
