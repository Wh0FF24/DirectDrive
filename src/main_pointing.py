#!/usr/bin/env python3
"""Point-and-Drive: stereo vision pointing detection + autonomous navigation.

Usage (on Radxa):
    source ~/rv/bin/activate
    python3 main_pointing.py [--debug] [--no-drive] [--calib PATH]

Usage (on desktop with webcam, for testing vision pipeline):
    python3 main_pointing.py --no-drive --webcam
"""

import argparse
import signal
import sys
import time

import cv2
import numpy as np

# Local flat imports (CyberTruck src/ layout)
import config as cfg
from car_controller import CarController
from stereo_vision import StereoVision
from pointing_detector import PointingDetector
from ground_target import GroundTargetEstimator
from navigator import Navigator


def parse_args():
    parser = argparse.ArgumentParser(description="Point-and-Drive CyberTruck")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug windows with HUD overlay")
    parser.add_argument("--no-drive", action="store_true",
                        help="Vision-only mode — don't send motor commands")
    parser.add_argument("--calib", type=str, default=cfg.STEREO_CALIB_PATH,
                        help="Path to stereo calibration .npz file")
    parser.add_argument("--webcam", action="store_true",
                        help="Use single webcam (split into fake stereo) for desktop testing")
    parser.add_argument("--record", type=str, default=None,
                        help="Record video to file (e.g. 'demo.avi')")
    parser.add_argument("--flip-left", action="store_true",
                        help="Flip left camera 180 degrees (if mounted upside-down)")
    parser.add_argument("--flip-right", action="store_true",
                        help="Flip right camera 180 degrees")
    return parser.parse_args()


def open_cameras(args):
    """Open stereo cameras or single webcam for testing."""
    if args.webcam:
        # Desktop testing: split single webcam into left/right
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            sys.exit(1)
        print("[Camera] Webcam mode — using single camera as fake stereo")
        return cap, None  # Single camera, will split later
    else:
        # Radxa: open both MIPI CSI cameras
        cap_left = cv2.VideoCapture(cfg.CAMERA_LEFT)
        cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

        cap_right = cv2.VideoCapture(cfg.CAMERA_RIGHT)
        cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

        if not cap_left.isOpened():
            print(f"[ERROR] Cannot open left camera (video{cfg.CAMERA_LEFT})")
            sys.exit(1)
        if not cap_right.isOpened():
            print(f"[ERROR] Cannot open right camera (video{cfg.CAMERA_RIGHT})")
            sys.exit(1)

        print(f"[Camera] Stereo: left=video{cfg.CAMERA_LEFT}, right=video{cfg.CAMERA_RIGHT}")
        return cap_left, cap_right


def read_stereo_pair(cap_left, cap_right, args):
    """Read a frame from both cameras. Returns (left, right) or (None, None)."""
    if cap_right is None:
        # Webcam mode: use same frame for both (no real stereo)
        ret, frame = cap_left.read()
        if not ret:
            return None, None
        return frame.copy(), frame.copy()
    else:
        ret_l, left = cap_left.read()
        ret_r, right = cap_right.read()
        if not ret_l or not ret_r:
            return None, None

        # Flip if cameras are mounted upside-down
        if args.flip_left:
            left = cv2.flip(left, -1)
        if args.flip_right:
            right = cv2.flip(right, -1)

        return left, right


def main():
    args = parse_args()
    print("=" * 60)
    print("  CyberTruck Point-and-Drive")
    print(f"  Mode: {'VISION ONLY' if args.no_drive else 'FULL DRIVE'}")
    print(f"  Debug: {args.debug}")
    print(f"  Webcam: {args.webcam}")
    print("=" * 60)

    # --- Initialize car controller ---
    car = None
    if not args.no_drive:
        car = CarController(cfg.SERIAL_PORT, cfg.BAUD_RATE)
        if not car.connect():
            print("[WARN] Car not connected — switching to no-drive mode")
            car = None

    # --- Emergency stop handler ---
    def signal_handler(sig, frame):
        print("\n[!] Ctrl+C — Emergency Stop!")
        if car:
            car.emergency_stop()
            car.disconnect()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # --- Initialize stereo vision ---
    calib_size = (640, 480) if not args.webcam else None
    try:
        stereo = StereoVision(
            calib_path=args.calib,
            image_size=(cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT),
            calib_size=calib_size,
        )
    except FileNotFoundError:
        print(f"[ERROR] Calibration file not found: {args.calib}")
        print("  Run: python3 scale_calibration.py   (or recalibrate at 1280x720)")
        if car:
            car.disconnect()
        sys.exit(1)

    # --- Initialize pointing detectors (one per camera) ---
    pointing_left = PointingDetector(
        extension_threshold=cfg.POINTING_EXTENSION_THRESH,
        hand_open_threshold=cfg.POINTING_HAND_OPEN_THRESH,
        min_detection_confidence=cfg.POINTING_CONFIDENCE_GATE,
    )
    pointing_right = PointingDetector(
        extension_threshold=cfg.POINTING_EXTENSION_THRESH,
        hand_open_threshold=cfg.POINTING_HAND_OPEN_THRESH,
        min_detection_confidence=cfg.POINTING_CONFIDENCE_GATE,
    )

    # --- Initialize ground target estimator ---
    estimator = GroundTargetEstimator(
        stereo=stereo,
        ground_y=cfg.GROUND_PLANE_Y,
        smooth_alpha=cfg.GROUND_SMOOTH_ALPHA,
        stable_frames_needed=cfg.POINTING_STABLE_FRAMES,
    )

    # --- Initialize navigator ---
    navigator = Navigator(
        car=car,
        pointing_left=pointing_left,
        pointing_right=pointing_right,
        stereo=stereo,
        estimator=estimator,
        config=cfg,
    )

    # --- Open cameras ---
    cap_left, cap_right = open_cameras(args)

    # --- Video writer ---
    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.record, fourcc, 20.0,
                                 (cfg.FRAME_WIDTH * 2, cfg.FRAME_HEIGHT))
        print(f"[Record] Saving to {args.record}")

    # --- Main loop ---
    print("\n[Ready] Point at the ground to direct the car. Ctrl+C to stop.\n")
    frame_count = 0
    fps_time = time.time()

    try:
        while True:
            left, right = read_stereo_pair(cap_left, cap_right, args)
            if left is None:
                print("[WARN] Failed to read frames")
                time.sleep(0.01)
                continue

            # Run navigator update
            debug = navigator.update(left, right)

            frame_count += 1

            # FPS calculation
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - fps_time)
                fps_time = now
                if not args.debug:
                    print(f"\r[{debug['state']}] FPS: {fps:.1f}  ", end="", flush=True)

            # Debug display
            if args.debug:
                left_hud = navigator.draw_hud(left, side="left")
                right_hud = navigator.draw_hud(right, side="right")

                # FPS overlay
                now = time.time()
                if frame_count % 30 == 0:
                    fps = 30.0 / (now - fps_time + 1e-6)
                cv2.putText(left_hud, f"FPS: {fps:.0f}" if frame_count > 30 else "",
                            (left_hud.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                combined = np.hstack([left_hud, right_hud])
                cv2.imshow("Point-and-Drive", combined)

                if writer:
                    writer.write(combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset state machine
                    navigator.state = Navigator.WAITING
                    navigator.target = None
                    estimator.reset()
                    print("[Reset] Back to WAITING")

    finally:
        print("\n[Cleanup] Shutting down...")
        if car:
            car.emergency_stop()
            car.disconnect()
        pointing_left.close()
        pointing_right.close()
        cap_left.release()
        if cap_right is not None:
            cap_right.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[Done]")


if __name__ == "__main__":
    main()
