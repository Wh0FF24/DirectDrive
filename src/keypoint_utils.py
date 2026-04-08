"""Geometric computations on pose keypoints for spatial intent analysis.

Copied from spatial-intent/src/pose/keypoint_utils.py with imports
adjusted for CyberTruck flat module layout.

All functions work with the landmark format from MediaPipePoseExtractor:
    landmarks: np.ndarray of shape (33, 4) with columns [x, y, z, visibility]
"""

import numpy as np

# Key joint indices (MediaPipe convention)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_INDEX = 19
RIGHT_INDEX = 20
NOSE = 0


def get_joint_position(landmarks: np.ndarray, joint_idx: int) -> np.ndarray:
    """Get the (x, y, z) position of a joint."""
    return landmarks[joint_idx, :3].copy()


def compute_arm_vector(landmarks: np.ndarray, side: str = "right") -> np.ndarray:
    """Compute the arm direction vector (elbow -> wrist), normalized."""
    if side == "right":
        elbow = landmarks[RIGHT_ELBOW, :3]
        wrist = landmarks[RIGHT_WRIST, :3]
    else:
        elbow = landmarks[LEFT_ELBOW, :3]
        wrist = landmarks[LEFT_WRIST, :3]

    vec = wrist - elbow
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def compute_pointing_ray(landmarks: np.ndarray, side: str = "right") -> np.ndarray:
    """Compute a pointing ray from shoulder through wrist (full arm line)."""
    if side == "right":
        shoulder = landmarks[RIGHT_SHOULDER, :3]
        wrist = landmarks[RIGHT_WRIST, :3]
    else:
        shoulder = landmarks[LEFT_SHOULDER, :3]
        wrist = landmarks[LEFT_WRIST, :3]

    vec = wrist - shoulder
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def compute_finger_pointing_ray(landmarks: np.ndarray, side: str = "right") -> np.ndarray:
    """Compute pointing ray from wrist through index finger tip."""
    if side == "right":
        wrist = landmarks[RIGHT_WRIST, :3]
        index = landmarks[RIGHT_INDEX, :3]
    else:
        wrist = landmarks[LEFT_WRIST, :3]
        index = landmarks[LEFT_INDEX, :3]

    vec = index - wrist
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def compute_joint_angle(
    landmarks: np.ndarray, joint_a: int, joint_b: int, joint_c: int
) -> float:
    """Compute the angle at joint_b formed by joints a-b-c, in degrees."""
    a = landmarks[joint_a, :3]
    b = landmarks[joint_b, :3]
    c = landmarks[joint_c, :3]

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_elbow_angle(landmarks: np.ndarray, side: str = "right") -> float:
    """Compute elbow flexion angle (shoulder-elbow-wrist) in degrees. 180 = fully extended."""
    if side == "right":
        return compute_joint_angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    return compute_joint_angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)


def compute_arm_extension_ratio(landmarks: np.ndarray, side: str = "right") -> float:
    """Compute how extended the arm is (0 = fully bent, 1 = fully extended)."""
    if side == "right":
        shoulder = landmarks[RIGHT_SHOULDER, :3]
        elbow = landmarks[RIGHT_ELBOW, :3]
        wrist = landmarks[RIGHT_WRIST, :3]
    else:
        shoulder = landmarks[LEFT_SHOULDER, :3]
        elbow = landmarks[LEFT_ELBOW, :3]
        wrist = landmarks[LEFT_WRIST, :3]

    upper_arm_len = np.linalg.norm(elbow - shoulder)
    forearm_len = np.linalg.norm(wrist - elbow)
    full_arm_len = upper_arm_len + forearm_len

    if full_arm_len < 1e-8:
        return 0.0

    direct_dist = np.linalg.norm(wrist - shoulder)
    return float(np.clip(direct_dist / full_arm_len, 0.0, 1.0))


def detect_pointing_arm(landmarks: np.ndarray, extension_threshold: float = 0.7) -> str | None:
    """Detect which arm (if any) is extended in a pointing gesture.

    Returns "left", "right", or None.
    """
    left_ext = compute_arm_extension_ratio(landmarks, "left")
    right_ext = compute_arm_extension_ratio(landmarks, "right")

    left_pointing = left_ext >= extension_threshold
    right_pointing = right_ext >= extension_threshold

    if right_pointing and left_pointing:
        return "right" if right_ext >= left_ext else "left"
    elif right_pointing:
        return "right"
    elif left_pointing:
        return "left"
    return None
