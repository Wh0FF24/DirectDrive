"""Navigator: state machine for point-and-drive behavior.

States: WAITING -> DRIVING -> ARRIVED -> TURNING -> WAITING (repeat)
"""

import time
import numpy as np
import cv2

from car_controller import CarController
from pointing_detector import PointingDetector, PointingResult
from stereo_vision import StereoVision
from ground_target import GroundTargetEstimator, GroundTarget
from geometry import project_point_to_image


class Navigator:
    WAITING = 0
    DRIVING = 1
    ARRIVED = 2
    TURNING = 3

    STATE_NAMES = {0: "WAITING", 1: "DRIVING", 2: "ARRIVED", 3: "TURNING"}

    def __init__(self, car: CarController | None,
                 pointing_left: PointingDetector,
                 pointing_right: PointingDetector,
                 stereo: StereoVision,
                 estimator: GroundTargetEstimator,
                 config):
        """
        Args:
            car: CarController instance (None for --no-drive mode).
            pointing_left: PointingDetector for left camera.
            pointing_right: PointingDetector for right camera.
            stereo: StereoVision instance.
            estimator: GroundTargetEstimator instance.
            config: Config module with NAV_* parameters.
        """
        self.car = car
        self.pointing_left = pointing_left
        self.pointing_right = pointing_right
        self.stereo = stereo
        self.estimator = estimator
        self.cfg = config

        self.state = self.WAITING
        self.target: GroundTarget | None = None
        self.drive_start_time = 0.0
        self.turn_start_time = 0.0

        # Debug info updated each frame
        self.debug_info = {}

    def update(self, left_frame: np.ndarray, right_frame: np.ndarray) -> dict:
        """Called every frame. Returns debug info dict."""
        self.debug_info = {
            "state": self.STATE_NAMES[self.state],
            "target": None,
            "pointing_left": None,
            "pointing_right": None,
        }

        if self.state == self.WAITING:
            self._handle_waiting(left_frame, right_frame)
        elif self.state == self.DRIVING:
            self._handle_driving(left_frame, right_frame)
        elif self.state == self.ARRIVED:
            self._handle_arrived()
        elif self.state == self.TURNING:
            self._handle_turning()

        return self.debug_info

    def _handle_waiting(self, left_frame, right_frame):
        """Detect pointing in both frames, estimate ground target."""
        # Detect pointing in both cameras
        result_left = self.pointing_left.detect(left_frame)
        result_right = self.pointing_right.detect(right_frame)

        self.debug_info["pointing_left"] = result_left
        self.debug_info["pointing_right"] = result_right

        if result_left is None or result_right is None:
            self.estimator.reset()
            return

        # Both cameras see pointing — estimate ground target
        target = self.estimator.estimate(
            left_frame, right_frame,
            result_left, result_right,
        )

        if target is None:
            return

        self.debug_info["target"] = target

        # Wait for stable detection before committing
        if self.estimator.is_stable:
            self.target = target
            self.state = self.DRIVING
            self.drive_start_time = time.time()
            self.estimator.reset()
            print(f"[Navigator] TARGET LOCKED: dist={target.distance:.0f}cm "
                  f"heading={target.heading_angle:.1f}deg")
            print(f"[Navigator] -> DRIVING")

    def _handle_driving(self, left_frame, right_frame):
        """Visual servo: re-estimate target, steer proportionally, stop when close."""
        elapsed = time.time() - self.drive_start_time

        # Safety timeout
        if elapsed > self.cfg.NAV_MAX_DRIVE_TIME:
            print(f"[Navigator] Drive timeout ({elapsed:.1f}s) -> ARRIVED")
            self.state = self.ARRIVED
            return

        # Re-estimate target for visual servo
        result_left = self.pointing_left.detect(left_frame)
        result_right = self.pointing_right.detect(right_frame)

        self.debug_info["pointing_left"] = result_left
        self.debug_info["pointing_right"] = result_right

        current_target = self.target  # fallback to last known

        if result_left is not None and result_right is not None:
            new_target = self.estimator.estimate(
                left_frame, right_frame,
                result_left, result_right,
            )
            if new_target is not None:
                current_target = new_target
                self.target = new_target

        self.debug_info["target"] = current_target

        if current_target is None:
            # Lost target — keep driving with last heading briefly, then stop
            if self.car:
                self.car.drive(self.cfg.NAV_DRIVE_SPEED)
            return

        # Check arrival
        if current_target.distance < self.cfg.ARRIVE_DISTANCE_CM:
            print(f"[Navigator] ARRIVED (dist={current_target.distance:.0f}cm)")
            self.state = self.ARRIVED
            return

        # Proportional steering
        steer_angle = self.cfg.NAV_STEER_KP * current_target.heading_angle
        steer_angle = max(-30, min(30, steer_angle))

        if self.car:
            self.car.steer(steer_angle)
            self.car.drive(self.cfg.NAV_DRIVE_SPEED)

        self.debug_info["steer_cmd"] = steer_angle
        self.debug_info["drive_elapsed"] = elapsed

    def _handle_arrived(self):
        """Stop car and transition to turning."""
        if self.car:
            self.car.stop()
            self.car.steer(0)
        time.sleep(0.5)

        self.state = self.TURNING
        self.turn_start_time = time.time()
        print("[Navigator] -> TURNING (180 deg)")

    def _handle_turning(self):
        """Turn ~180 degrees using timed full-lock steering."""
        elapsed = time.time() - self.turn_start_time

        if elapsed > self.cfg.NAV_TURN_DURATION:
            # Done turning
            if self.car:
                self.car.stop()
                self.car.steer(0)
            self.state = self.WAITING
            self.target = None
            print("[Navigator] -> WAITING (ready for next point)")
            return

        # Full steering lock + slow speed for turn
        if self.car:
            self.car.steer(30)  # full right lock
            self.car.drive(self.cfg.NAV_DRIVE_SPEED)

    def draw_hud(self, frame: np.ndarray, side: str = "left") -> np.ndarray:
        """Draw debug HUD overlay on a frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # State indicator
        state_name = self.STATE_NAMES[self.state]
        color = {
            self.WAITING: (0, 255, 255),   # yellow
            self.DRIVING: (0, 255, 0),     # green
            self.ARRIVED: (255, 0, 0),     # blue
            self.TURNING: (0, 165, 255),   # orange
        }[self.state]

        cv2.putText(overlay, f"STATE: {state_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw target info
        target = self.debug_info.get("target")
        if target is not None:
            cv2.putText(overlay, f"Dist: {target.distance:.0f}cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Heading: {target.heading_angle:.1f}deg", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            stable_text = "STABLE" if self.estimator.is_stable else f"Stabilizing ({self.estimator._stable_count})"
            cv2.putText(overlay, stable_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.estimator.is_stable else (0, 0, 255), 2)

        # Draw skeleton if pointing detected
        pointing_key = f"pointing_{side}"
        pointing_result = self.debug_info.get(pointing_key)
        if pointing_result is not None:
            # Draw shoulder -> wrist line
            sh = tuple(pointing_result.shoulder_px.astype(int))
            wr = tuple(pointing_result.wrist_px.astype(int))
            idx = tuple(pointing_result.index_px.astype(int))

            cv2.circle(overlay, sh, 8, (0, 255, 0), -1)
            cv2.circle(overlay, wr, 8, (0, 0, 255), -1)
            cv2.circle(overlay, idx, 6, (255, 0, 255), -1)
            cv2.line(overlay, sh, wr, (0, 255, 255), 2)
            cv2.arrowedLine(overlay, wr, idx, (255, 0, 255), 2)

            # Confidence
            cv2.putText(overlay, f"Conf: {pointing_result.confidence:.2f}",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Steer command
        steer = self.debug_info.get("steer_cmd")
        if steer is not None:
            cv2.putText(overlay, f"Steer: {steer:.1f}deg", (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return overlay
