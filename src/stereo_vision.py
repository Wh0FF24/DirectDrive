"""Stereo vision: calibration loading, rectification, and triangulation.

Loads stereo calibration from .npz (K_L, K_R, dist_L, dist_R, R, T),
computes rectification maps, and provides rectify() + triangulate() methods.
"""

import numpy as np
import cv2


class StereoVision:
    def __init__(self, calib_path: str, image_size: tuple[int, int] = (1280, 720),
                 calib_size: tuple[int, int] | None = None):
        """Load stereo calibration and compute rectification maps.

        Args:
            calib_path: Path to .npz file with K_L, K_R, dist_L, dist_R, R, T.
            image_size: (width, height) of the frames we'll process.
            calib_size: (width, height) the calibration was done at. If different
                        from image_size, intrinsics are scaled automatically.
        """
        data = np.load(calib_path)
        self.K_L = data["K_L"].astype(np.float64)
        self.K_R = data["K_R"].astype(np.float64)
        self.dist_L = data["dist_L"].astype(np.float64)
        self.dist_R = data["dist_R"].astype(np.float64)
        self.R = data["R"].astype(np.float64)
        self.T = data["T"].astype(np.float64)

        self.image_size = image_size  # (w, h)

        # Scale intrinsics if calibration was at a different resolution
        if calib_size is not None and calib_size != image_size:
            sx = image_size[0] / calib_size[0]
            sy = image_size[1] / calib_size[1]
            self.K_L[0, :] *= sx  # fx, cx
            self.K_L[1, :] *= sy  # fy, cy
            self.K_R[0, :] *= sx
            self.K_R[1, :] *= sy
            print(f"[StereoVision] Scaled intrinsics from {calib_size} -> {image_size} "
                  f"(sx={sx:.2f}, sy={sy:.2f})")

        # Compute rectification transforms
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = \
            cv2.stereoRectify(
                self.K_L, self.dist_L,
                self.K_R, self.dist_R,
                (image_size[0], image_size[1]),
                self.R, self.T,
                alpha=0,  # 0 = crop to valid pixels
            )

        # Precompute undistort+rectify maps for fast remap
        self.map_L1, self.map_L2 = cv2.initUndistortRectifyMap(
            self.K_L, self.dist_L, self.R1, self.P1,
            (image_size[0], image_size[1]), cv2.CV_32FC1
        )
        self.map_R1, self.map_R2 = cv2.initUndistortRectifyMap(
            self.K_R, self.dist_R, self.R2, self.P2,
            (image_size[0], image_size[1]), cv2.CV_32FC1
        )

        # Projection matrices for triangulation (using original K, not rectified P)
        # T is negated to match OpenCV triangulatePoints convention
        self.P1_tri = self.K_L @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2_tri = self.K_R @ np.hstack([self.R, self.T])

        # Baseline in the same units as T (centimeters from stereo_calib)
        self.baseline = np.linalg.norm(self.T)
        print(f"[StereoVision] Loaded calibration from {calib_path}")
        print(f"  Baseline: {self.baseline:.2f} cm")
        print(f"  Image size: {image_size}")

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rectify a stereo pair using precomputed maps."""
        left_rect = cv2.remap(left, self.map_L1, self.map_L2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map_R1, self.map_R2, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def triangulate(self, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
        """Triangulate 2D point correspondences -> 3D world points.

        Args:
            pts_left: (N, 2) pixel coordinates in left image (raw/unrectified).
            pts_right: (N, 2) pixel coordinates in right image (raw/unrectified).

        Returns:
            (N, 3) 3D points in left camera frame (units match T).
        """
        pts_left = pts_left.astype(np.float64).reshape(-1, 1, 2)
        pts_right = pts_right.astype(np.float64).reshape(-1, 1, 2)

        # Undistort pixel coordinates (no rectification — use original K)
        pts_left_u = cv2.undistortPoints(pts_left, self.K_L, self.dist_L, P=self.K_L)
        pts_right_u = cv2.undistortPoints(pts_right, self.K_R, self.dist_R, P=self.K_R)

        # Triangulate using original-K projection matrices
        pts4d = cv2.triangulatePoints(self.P1_tri, self.P2_tri,
                                       pts_left_u.reshape(-1, 2).T,
                                       pts_right_u.reshape(-1, 2).T)

        # Convert from homogeneous (4, N) -> (N, 3)
        pts3d = (pts4d[:3] / pts4d[3:]).T
        return pts3d

    def get_disparity_depth(self, disparity_px: float) -> float:
        """Convert pixel disparity to depth using baseline and focal length.

        Returns depth in same units as baseline (centimeters).
        """
        f = self.P1[0, 0]  # focal length in pixels after rectification
        if abs(disparity_px) < 1e-6:
            return float('inf')
        return float(f * self.baseline / disparity_px)
