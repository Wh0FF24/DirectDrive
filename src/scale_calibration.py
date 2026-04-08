#!/usr/bin/env python3
"""Scale stereo calibration from 640x480 to 1280x720 and save to calibration/.

Usage:
    python3 scale_calibration.py [--input PATH] [--output PATH]

The 640x480 calibration from the Stereo Calibration assignment is loaded,
intrinsics are scaled to 1280x720, and the result is saved for main_pointing.py.
"""

import argparse
import os
import numpy as np

# Default paths
DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Assignments", "Stereo Calibration and Rectification",
    "outputs", "stereo_calib.npz"
)
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "calibration", "stereo_calib.npz")

CALIB_SIZE = (640, 480)
TARGET_SIZE = (1280, 720)


def scale_intrinsics(K, from_size, to_size):
    """Scale a camera intrinsic matrix from one resolution to another."""
    K_scaled = K.copy()
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    K_scaled[0, 0] *= sx  # fx
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[1, 2] *= sy  # cy
    return K_scaled


def main():
    parser = argparse.ArgumentParser(description="Scale stereo calibration to 1280x720")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Input stereo_calib.npz path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output scaled .npz path")
    args = parser.parse_args()

    # Resolve paths
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        print("  Make sure the Stereo Calibration assignment outputs exist.")
        return

    print(f"[Input]  {input_path}")
    data = np.load(input_path)

    K_L = data["K_L"]
    K_R = data["K_R"]
    dist_L = data["dist_L"]
    dist_R = data["dist_R"]
    R = data["R"]
    T = data["T"]
    E = data["E"]
    F = data["F"]

    print(f"\nOriginal calibration ({CALIB_SIZE[0]}x{CALIB_SIZE[1]}):")
    print(f"  K_L focal: fx={K_L[0,0]:.1f}, fy={K_L[1,1]:.1f}")
    print(f"  K_L center: cx={K_L[0,2]:.1f}, cy={K_L[1,2]:.1f}")
    print(f"  K_R focal: fx={K_R[0,0]:.1f}, fy={K_R[1,1]:.1f}")
    print(f"  Baseline (T): [{T.ravel()[0]:.2f}, {T.ravel()[1]:.2f}, {T.ravel()[2]:.2f}] cm")

    # Scale intrinsics
    K_L_scaled = scale_intrinsics(K_L, CALIB_SIZE, TARGET_SIZE)
    K_R_scaled = scale_intrinsics(K_R, CALIB_SIZE, TARGET_SIZE)

    print(f"\nScaled calibration ({TARGET_SIZE[0]}x{TARGET_SIZE[1]}):")
    print(f"  K_L focal: fx={K_L_scaled[0,0]:.1f}, fy={K_L_scaled[1,1]:.1f}")
    print(f"  K_L center: cx={K_L_scaled[0,2]:.1f}, cy={K_L_scaled[1,2]:.1f}")
    print(f"  K_R focal: fx={K_R_scaled[0,0]:.1f}, fy={K_R_scaled[1,1]:.1f}")

    # Save scaled calibration
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path,
             K_L=K_L_scaled,
             K_R=K_R_scaled,
             dist_L=dist_L,
             dist_R=dist_R,
             R=R,
             T=T,
             E=E,
             F=F)

    print(f"\n[Output] {output_path}")
    print("[Done] Calibration scaled and saved.")


if __name__ == "__main__":
    main()
