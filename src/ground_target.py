"""Ground target estimation using stereo triangulation + 3D ray-plane intersection.

Triangulates shoulder and wrist in 3D from both stereo views,
forms a pointing ray (shoulder -> wrist), and intersects it with
the ground plane to find where the person is pointing.
"""

from dataclasses import dataclass
import numpy as np
import cv2

from stereo_vision import StereoVision
from pointing_detector import PointingResult
from geometry import ray_plane_intersection, normalize_vector, project_point_to_image


@dataclass
class GroundTarget:
    point_3d: np.ndarray    # (3,) XYZ intersection on ground plane (units match T)
    distance: float         # straight-line ground distance from camera
    heading_angle: float    # degrees: positive = target is to the right
    ground_px: tuple        # (x, y) pixel in left image for crosshair
    shoulder_3d: np.ndarray | None = None  # for debug visualization
    wrist_3d: np.ndarray | None = None     # for debug visualization


class GroundTargetEstimator:
    """Estimates where a pointed-at ground location is using stereo triangulation."""

    def __init__(self, stereo: StereoVision,
                 ground_y: float = 170.5,
                 smooth_alpha: float = 0.3,
                 max_jump: float = 1000.0,
                 stable_frames_needed: int = 5,
                 stable_heading_tol: float = 15.0,
                 stable_distance_tol: float = 300.0):
        self.stereo = stereo
        self.ground_y = ground_y          # mm below camera to ground plane
        self.smooth_alpha = smooth_alpha
        self.max_jump = max_jump          # outlier rejection threshold
        self.stable_frames_needed = stable_frames_needed
        self.stable_heading_tol = stable_heading_tol
        self.stable_distance_tol = stable_distance_tol

        self._smooth_target = None
        self._prev_distance = None
        self._prev_heading = None
        self._stable_count = 0

    def estimate(self, left_frame: np.ndarray, right_frame: np.ndarray,
                 pointing_left: PointingResult,
                 pointing_right: PointingResult) -> GroundTarget | None:
        h, w = left_frame.shape[:2]

        # --- Triangulate shoulder and wrist from both views ---
        pts_left = np.array([pointing_left.shoulder_px,
                             pointing_left.wrist_px], dtype=np.float64)
        pts_right = np.array([pointing_right.shoulder_px,
                              pointing_right.wrist_px], dtype=np.float64)

        pts_3d = self.stereo.triangulate(pts_left, pts_right)
        shoulder_3d = pts_3d[0]  # (3,)
        wrist_3d = pts_3d[1]    # (3,)

        # Sanity: both points should be in front of camera (Z > 0)
        if shoulder_3d[2] <= 0 or wrist_3d[2] <= 0:
            self._stable_count = 0
            return None

        # Sanity: reject if points are unreasonably far (> 20m)
        if np.linalg.norm(shoulder_3d) > 20000 or np.linalg.norm(wrist_3d) > 20000:
            self._stable_count = 0
            return None

        # --- Compute pointing ray: shoulder -> wrist, extended to ground ---
        ray_origin = shoulder_3d
        ray_direction = wrist_3d - shoulder_3d

        if np.linalg.norm(ray_direction) < 1e-6:
            self._stable_count = 0
            return None

        # Ground plane: Y = +ground_y (OpenCV convention: Y points down)
        # The ground is ground_y units below camera origin
        plane_point = np.array([0.0, self.ground_y, 0.0])
        plane_normal = np.array([0.0, 1.0, 0.0])  # pointing down (same as Y axis)

        hit = ray_plane_intersection(ray_origin, ray_direction,
                                     plane_point, plane_normal)
        if hit is None:
            # Ray doesn't intersect ground (pointing up or parallel)
            self._stable_count = 0
            return None

        # Sanity: intersection should be in front of camera and reasonable distance
        if hit[2] <= 0 or np.linalg.norm(hit) > 50000:
            self._stable_count = 0
            return None

        # --- Outlier rejection: reject if target jumps too far from previous ---
        if self._smooth_target is not None:
            jump = np.linalg.norm(hit - self._smooth_target)
            if jump > self.max_jump:
                # Keep previous estimate, don't update stable count
                return self._make_target(self._smooth_target, shoulder_3d, wrist_3d, w, h)

        # --- EMA smoothing on ground intersection point ---
        if self._smooth_target is None:
            self._smooth_target = hit.copy()
        else:
            self._smooth_target = (self.smooth_alpha * hit +
                                   (1 - self.smooth_alpha) * self._smooth_target)

        distance = float(np.sqrt(self._smooth_target[0]**2 + self._smooth_target[2]**2))
        heading = float(np.degrees(np.arctan2(self._smooth_target[0],
                                               self._smooth_target[2])))

        # --- Stability tracking ---
        if self._prev_distance is not None and self._prev_heading is not None:
            d_delta = abs(distance - self._prev_distance)
            h_delta = abs(heading - self._prev_heading)
            if d_delta < self.stable_distance_tol and h_delta < self.stable_heading_tol:
                self._stable_count += 1
            else:
                self._stable_count = 1  # reset but count this frame
        else:
            self._stable_count = 1

        self._prev_distance = distance
        self._prev_heading = heading

        return self._make_target(self._smooth_target, shoulder_3d, wrist_3d, w, h)

    def _make_target(self, point_3d: np.ndarray,
                     shoulder_3d: np.ndarray, wrist_3d: np.ndarray,
                     w: int, h: int) -> GroundTarget:
        """Build GroundTarget from smoothed 3D point."""
        distance = float(np.sqrt(point_3d[0]**2 + point_3d[2]**2))
        heading = float(np.degrees(np.arctan2(point_3d[0], point_3d[2])))

        # Project ground hit back to left image for crosshair
        ground_px = self._project_to_left(point_3d, w, h)

        return GroundTarget(
            point_3d=point_3d.copy(),
            distance=distance,
            heading_angle=heading,
            ground_px=ground_px,
            shoulder_3d=shoulder_3d,
            wrist_3d=wrist_3d,
        )

    def _project_to_left(self, point_3d: np.ndarray, w: int, h: int) -> tuple:
        """Reproject a 3D point to left camera pixel coordinates."""
        pt_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        px_proj = self.stereo.P1_tri @ pt_h
        if px_proj[2] > 0:
            proj_x = int(np.clip(px_proj[0] / px_proj[2], 0, w - 1))
            proj_y = int(np.clip(px_proj[1] / px_proj[2], 0, h - 1))
            return (proj_x, proj_y)
        return (w // 2, h // 2)

    @property
    def is_stable(self) -> bool:
        return self._stable_count >= self.stable_frames_needed

    def reset(self):
        self._smooth_target = None
        self._prev_distance = None
        self._prev_heading = None
        self._stable_count = 0
