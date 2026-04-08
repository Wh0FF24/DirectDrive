"""3D geometry helpers for spatial intent computation.

Copied from spatial-intent/src/utils/geometry.py with imports
adjusted for CyberTruck flat module layout.
"""

import numpy as np


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length. Returns zero vector if input is near-zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def angular_error_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angular error between two direction vectors in degrees."""
    v1_n = normalize_vector(v1)
    v2_n = normalize_vector(v2)

    if np.linalg.norm(v1_n) < 1e-8 or np.linalg.norm(v2_n) < 1e-8:
        return 180.0

    cos_angle = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def ray_plane_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Find the intersection point of a ray with a plane.

    Returns 3D intersection point, or None if ray is parallel or behind origin.
    """
    ray_dir = normalize_vector(ray_direction)
    plane_n = normalize_vector(plane_normal)

    denom = np.dot(ray_dir, plane_n)
    if abs(denom) < 1e-8:
        return None

    t = np.dot(plane_point - ray_origin, plane_n) / denom
    if t < 0:
        return None  # Intersection is behind the ray origin

    return ray_origin + t * ray_dir


def project_point_to_image(
    point_3d: np.ndarray, camera_matrix: np.ndarray
) -> np.ndarray:
    """Project a 3D point onto the image plane using a camera intrinsic matrix."""
    if point_3d[2] < 1e-8:
        return np.array([0.0, 0.0])

    proj = camera_matrix @ point_3d
    return (proj[:2] / proj[2]).astype(np.float32)


def direction_to_azimuth_elevation(direction: np.ndarray) -> tuple[float, float]:
    """Convert a 3D direction vector to azimuth and elevation angles.

    Convention: azimuth from +Z axis (positive=right), elevation from XZ plane (positive=up).
    """
    d = normalize_vector(direction)
    if np.linalg.norm(d) < 1e-8:
        return 0.0, 0.0

    azimuth = np.degrees(np.arctan2(d[0], d[2]))
    elevation = np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0)))
    return float(azimuth), float(elevation)
