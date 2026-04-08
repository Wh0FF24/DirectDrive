"""Ground target estimation using ankle depth + 2D pointing ray.

Uses stereo depth at the ankle (reliable) to establish the person's distance,
then the 2D wrist position in the left camera to determine heading.
The target is projected onto the ground at the person's depth.
"""

from dataclasses import dataclass
import numpy as np
import cv2

from stereo_vision import StereoVision
from pointing_detector import PointingResult

LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class GroundTarget:
    point_3d: np.ndarray    # (3,) XYZ in camera frame (mm)
    distance: float         # straight-line ground distance from camera (mm)
    heading_angle: float    # degrees: positive = target is to the right
    ground_px: tuple        # (x, y) pixel in left image for crosshair


class GroundTargetEstimator:
    """Estimates where a pointed-at ground location is in 3D."""

    def __init__(self, stereo: StereoVision,
                 ground_y: float = 170.5,
                 smooth_alpha: float = 0.5,
                 stable_frames_needed: int = 5):
        self.stereo = stereo
        self.ground_y = ground_y
        self.smooth_alpha = smooth_alpha
        self.stable_frames_needed = stable_frames_needed

        self._smooth_target = None
        self._stable_count = 0

        self.stereo_matcher = cv2.StereoBM.create(numDisparities=128, blockSize=21)

    def _get_depth_at_pixel(self, disparity: np.ndarray, x: int, y: int,
                            patch: int = 25) -> float:
        h, w = disparity.shape
        y0, y1 = max(0, y - patch), min(h, y + patch)
        x0, x1 = max(0, x - patch), min(w, x + patch)
        region = disparity[y0:y1, x0:x1]
        valid = region[region > 1.0]
        if len(valid) < 10:
            return -1.0
        return self.stereo.get_disparity_depth(float(np.median(valid)))

    def estimate(self, left_frame: np.ndarray, right_frame: np.ndarray,
                 pointing_left: PointingResult,
                 pointing_right: PointingResult) -> GroundTarget | None:
        h, w = left_frame.shape[:2]
        landmarks_px = pointing_left.landmarks_px

        # --- Get reliable depth from hip (always visible, same depth as feet) ---
        hip_l = landmarks_px[LEFT_HIP]
        hip_r = landmarks_px[RIGHT_HIP]
        hip_px = (hip_l + hip_r) / 2.0  # midpoint between hips

        if not (0 <= hip_px[0] < w and 0 <= hip_px[1] < h):
            self._stable_count = 0
            return None

        # Rectify the hip pixel coordinate to match the disparity map
        hip_pt = np.array([[hip_px]], dtype=np.float64)
        hip_rect = cv2.undistortPoints(hip_pt,
                                        self.stereo.K_L, self.stereo.dist_L,
                                        R=self.stereo.R1, P=self.stereo.P1)
        ax = int(np.clip(hip_rect[0, 0, 0], 0, w - 1))
        ay = int(np.clip(hip_rect[0, 0, 1], 0, h - 1))

        left_rect, right_rect = self.stereo.rectify(left_frame, right_frame)
        gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0

        depth_mm = self._get_depth_at_pixel(disparity, ax, ay)
        if depth_mm <= 0 or depth_mm > 20000:
            self._stable_count = 0
            return None

        # --- Use wrist pixel to determine where on the ground ---
        wrist_pt = np.array([[pointing_left.wrist_px]], dtype=np.float64)
        wrist_rect = cv2.undistortPoints(wrist_pt,
                                          self.stereo.K_L, self.stereo.dist_L,
                                          R=self.stereo.R1, P=self.stereo.P1)
        wx = float(wrist_rect[0, 0, 0])
        wy = float(wrist_rect[0, 0, 1])
        fx = self.stereo.P1[0, 0]
        fy = self.stereo.P1[1, 1]
        cx = self.stereo.P1[0, 2]
        cy = self.stereo.P1[1, 2]

        # 3D point: wrist at ankle depth, projected to ground
        X = float((wx - cx) * depth_mm / fx)
        Y = self.ground_y
        Z = depth_mm

        hit = np.array([X, Y, Z])

        if np.linalg.norm(hit) > 20000:
            self._stable_count = 0
            return None

        # --- Project hit back to left image for crosshair ---
        # Use P1_tri (original camera matrix) to project back
        pt_h = np.array([hit[0], hit[1], hit[2], 1.0])
        px_proj = self.stereo.P1_tri @ pt_h
        if px_proj[2] > 0:
            proj_x = int(px_proj[0] / px_proj[2])
            proj_y = int(px_proj[1] / px_proj[2])
            ground_px = (np.clip(proj_x, 0, w-1), np.clip(proj_y, 0, h-1))
        else:
            ground_px = (int(wx), int(wy))

        # EMA smoothing
        if self._smooth_target is None:
            self._smooth_target = hit.copy()
        else:
            self._smooth_target = (self.smooth_alpha * hit +
                                   (1 - self.smooth_alpha) * self._smooth_target)

        distance = np.sqrt(self._smooth_target[0]**2 + self._smooth_target[2]**2)
        heading = np.degrees(np.arctan2(self._smooth_target[0],
                                         self._smooth_target[2]))

        self._stable_count += 1

        return GroundTarget(
            point_3d=self._smooth_target.copy(),
            distance=float(distance),
            heading_angle=float(heading),
            ground_px=ground_px,
        )

    @property
    def is_stable(self) -> bool:
        return self._stable_count >= self.stable_frames_needed

    def reset(self):
        self._smooth_target = None
        self._stable_count = 0
