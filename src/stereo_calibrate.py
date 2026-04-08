#!/usr/bin/env python3
"""Stereo camera calibration using checkerboard captures.

Usage:
    source ~/rv/bin/activate
    cd ~/Desktop/DirectDrive/src
    python3 stereo_calibrate.py

Controls:
    SPACE  — capture a pair (only saves if corners found in both)
    C      — run calibration with captured pairs
    Q      — quit

Aim for 15-20 captures with the board at varied positions, angles, and distances.
"""

import os
import cv2
import numpy as np
import config as cfg

# === Checkerboard config ===
BOARD_ROWS = 5       # internal corner rows
BOARD_COLS = 7       # internal corner columns
SQUARE_SIZE_MM = 31  # mm per square

# Output path
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "calibration", "stereo_calib.npz")

# Corner refinement criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def make_object_points():
    """3D points of checkerboard corners in board frame (Z=0)."""
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # in mm
    return objp


def find_corners(gray):
    """Find checkerboard corners in a grayscale image."""
    found, corners = cv2.findChessboardCorners(
        gray, (BOARD_COLS, BOARD_ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
    )
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return found, corners


def draw_status(frame, text, color=(0, 255, 0)):
    """Draw status text on frame."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def run_calibration(obj_points_list, img_points_l, img_points_r, image_size):
    """Run stereo calibration and return results."""
    print(f"\nRunning stereo calibration with {len(obj_points_list)} image pairs...")
    print(f"  Image size: {image_size}")
    print(f"  Board: {BOARD_COLS}x{BOARD_ROWS} inner corners, {SQUARE_SIZE_MM}mm squares")

    # Individual camera calibrations first (better initial estimates)
    print("  Calibrating left camera...")
    ret_l, K_L, dist_L, _, _ = cv2.calibrateCamera(
        obj_points_list, img_points_l, image_size, None, None
    )
    print(f"    RMS reprojection error: {ret_l:.4f}")

    print("  Calibrating right camera...")
    ret_r, K_R, dist_R, _, _ = cv2.calibrateCamera(
        obj_points_list, img_points_r, image_size, None, None
    )
    print(f"    RMS reprojection error: {ret_r:.4f}")

    # Stereo calibration
    print("  Running stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC  # use individual calibration results
    ret_stereo, K_L, dist_L, K_R, dist_R, R, T, E, F = cv2.stereoCalibrate(
        obj_points_list, img_points_l, img_points_r,
        K_L, dist_L, K_R, dist_R,
        image_size,
        criteria=CRITERIA,
        flags=flags,
    )
    print(f"  Stereo RMS reprojection error: {ret_stereo:.4f}")
    print(f"  Baseline (T): [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}] mm")
    print(f"  Baseline magnitude: {np.linalg.norm(T):.2f} mm")

    return K_L, K_R, dist_L, dist_R, R, T, E, F, ret_stereo


def load_saved_images():
    """Load previously saved calibration image pairs."""
    calib_img_dir = os.path.join(os.path.dirname(__file__), "calibration", "images")
    if not os.path.isdir(calib_img_dir):
        return [], [], []

    objp = make_object_points()
    obj_points_list = []
    img_points_l = []
    img_points_r = []

    idx = 1
    while True:
        lp = os.path.join(calib_img_dir, f"left_{idx:02d}.png")
        rp = os.path.join(calib_img_dir, f"right_{idx:02d}.png")
        if not os.path.exists(lp) or not os.path.exists(rp):
            break
        left = cv2.imread(lp)
        right = cv2.imread(rp)
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_corners(gray_l)
        found_r, corners_r = find_corners(gray_r)

        if found_l and found_r:
            obj_points_list.append(objp)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
        else:
            print(f"  WARNING: Pair {idx} — corners not found (L={found_l} R={found_r}), skipped")
        idx += 1

    print(f"  Loaded {len(obj_points_list)} pairs from {calib_img_dir}")
    return obj_points_list, img_points_l, img_points_r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
                        help="Calibrate from saved images (no cameras needed)")
    args = parser.parse_args()

    if args.load:
        obj_points_list, img_points_l, img_points_r = load_saved_images()
        if len(obj_points_list) < 5:
            print(f"  Need at least 5 pairs (have {len(obj_points_list)})")
            return
        image_size = (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
        K_L, K_R, dist_L, dist_R, R, T, E, F, rms = run_calibration(
            obj_points_list, img_points_l, img_points_r, image_size
        )
        out = os.path.abspath(OUTPUT_PATH)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, K_L=K_L, K_R=K_R, dist_L=dist_L, dist_R=dist_R,
                 R=R, T=T, E=E, F=F)
        print(f"\n  Saved to {out}")
        print(f"  RMS error: {rms:.4f}")
        print(f"  fx/fy ratio: {K_L[0,0]/K_L[1,1]:.3f} (should be ~1.0)")
        print(f"  K_L: fx={K_L[0,0]:.1f}  fy={K_L[1,1]:.1f}")
        print(f"  T: [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}]")
        print(f"  Baseline: {np.linalg.norm(T):.2f} mm")
        return

    # Open cameras
    cap_l = cv2.VideoCapture(cfg.CAMERA_LEFT)
    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    cap_r = cv2.VideoCapture(cfg.CAMERA_RIGHT)
    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("Failed to open cameras")
        return

    print("Stereo Calibration")
    print(f"  Board: {BOARD_COLS}x{BOARD_ROWS} inner corners, {SQUARE_SIZE_MM}mm squares")
    print(f"  Resolution: {cfg.FRAME_WIDTH}x{cfg.FRAME_HEIGHT}")
    print()
    print("  SPACE = capture | C = calibrate | Q = quit")
    print("  Aim for 15-20 captures at varied positions/angles.")
    print()

    objp = make_object_points()
    obj_points_list = []
    img_points_l = []
    img_points_r = []

    while True:
        ret_l, left = cap_l.read()
        ret_r, right = cap_r.read()
        if not ret_l or not ret_r:
            continue

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Try to find corners (for live preview)
        found_l, corners_l = find_corners(gray_l)
        found_r, corners_r = find_corners(gray_r)

        # Draw corners on display copies
        disp_l = left.copy()
        disp_r = right.copy()

        if found_l:
            cv2.drawChessboardCorners(disp_l, (BOARD_COLS, BOARD_ROWS), corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(disp_r, (BOARD_COLS, BOARD_ROWS), corners_r, found_r)

        # Status
        n = len(obj_points_list)
        both = found_l and found_r
        status_color = (0, 255, 0) if both else (0, 0, 255)
        status = f"Pairs: {n} | {'READY - press SPACE' if both else 'Move board into view'}"
        draw_status(disp_l, status, status_color)
        draw_status(disp_r, f"L:{'OK' if found_l else '--'}  R:{'OK' if found_r else '--'}", status_color)

        combined = np.hstack([disp_l, disp_r])
        display = cv2.resize(combined, (combined.shape[1] // 2, combined.shape[0] // 2))
        cv2.imshow("Stereo Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if both:
                obj_points_list.append(objp)
                img_points_l.append(corners_l)
                img_points_r.append(corners_r)
                idx = len(obj_points_list)
                # Save raw images
                calib_img_dir = os.path.join(os.path.dirname(__file__), "calibration", "images")
                os.makedirs(calib_img_dir, exist_ok=True)
                cv2.imwrite(os.path.join(calib_img_dir, f"left_{idx:02d}.png"), left)
                cv2.imwrite(os.path.join(calib_img_dir, f"right_{idx:02d}.png"), right)
                print(f"  Captured pair #{idx} (saved to calibration/images/)")
            else:
                print("  Corners not found in both images — skipped")

        elif key == ord('c'):
            if len(obj_points_list) < 5:
                print(f"  Need at least 5 pairs (have {len(obj_points_list)})")
                continue

            image_size = (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
            K_L, K_R, dist_L, dist_R, R, T, E, F, rms = run_calibration(
                obj_points_list, img_points_l, img_points_r, image_size
            )

            # Save
            out = os.path.abspath(OUTPUT_PATH)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out,
                     K_L=K_L, K_R=K_R,
                     dist_L=dist_L, dist_R=dist_R,
                     R=R, T=T, E=E, F=F)

            # Verify save
            verify = np.load(out)
            print(f"\n  Saved to {out}")
            print(f"  RMS error: {rms:.4f} (good if < 1.0)")
            print(f"  fx/fy ratio: {K_L[0,0]/K_L[1,1]:.3f} (should be ~1.0 for native res)")
            print(f"  K_L: fx={K_L[0,0]:.1f}  fy={K_L[1,1]:.1f}")
            print(f"  T: [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}]")
            print(f"  Baseline: {np.linalg.norm(T):.2f} mm")
            print(f"  Verified: {list(verify.keys())}")
            print("\n  Press Q to quit or keep capturing for a re-run.")

        elif key == ord('q'):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
