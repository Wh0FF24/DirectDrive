# DirectDrive — Point-and-Drive with Stereo Vision

## What This Is

ECEN 631 (Robotic Vision) final project at BYU. An RC car (CyberTruck platform on Radxa RK3588)
uses dual stereo cameras to detect a person pointing at a spot on the ground, triangulates
the 3D ground location, drives there, turns 180 degrees, and waits for the next command.

**Team name:** Direct Drive (Team 1)
**Student:** Will W — BYU senior, ECEN 631 Robotic Vision, familiar with Python/OpenCV/stereo vision.
This also serves as thesis RQ3 demo (stereo depth for pointing) in his spatial-intent research.

## Project Structure

All source code lives in `src/` with **flat imports** (no package hierarchy — just `from config import ...`).

```
src/
├── main_pointing.py      # Entry point — run this
├── config.py             # All tunable parameters (ONLY file with magic numbers)
├── car_controller.py     # Arduino serial: steer(angle), drive(speed), emergency_stop()
├── stereo_vision.py      # Loads stereo calibration, rectify(), triangulate()
├── pointing_detector.py  # MediaPipe pose → PointingResult (shoulder/wrist pixel coords)
├── ground_target.py      # Stereo triangulation → ray-plane intersection → GroundTarget
├── navigator.py          # State machine: WAITING → DRIVING → ARRIVED → TURNING → WAITING
├── mediapipe_extractor.py  # MediaPipe Pose wrapper (from spatial-intent research)
├── keypoint_utils.py       # Arm extension ratio, pointing ray math (from spatial-intent)
├── geometry.py             # ray_plane_intersection, normalize_vector (from spatial-intent)
├── scale_calibration.py    # One-shot script: scale 640x480 calibration → 1280x720
└── calibration/
    └── stereo_calib.npz  # Scaled stereo calibration (K_L, K_R, dist_L, dist_R, R, T)
```

## How to Run

```bash
# On Radxa:
source ~/rv/bin/activate
cd src/
python3 main_pointing.py --debug              # stereo cameras + driving
python3 main_pointing.py --debug --no-drive   # vision only, no motor commands

# Desktop testing (single webcam, fake stereo — no real 3D):
python3 main_pointing.py --debug --no-drive --webcam
```

### CLI Flags
- `--debug` — Show OpenCV windows with HUD overlay (skeleton, target info, state)
- `--no-drive` — Don't send serial commands to Arduino (ALWAYS use this first)
- `--webcam` — Single-webcam mode for desktop testing (duplicates frame as fake stereo)
- `--calib PATH` — Override path to stereo calibration .npz
- `--record FILE` — Record side-by-side video to .avi
- `--flip-left` / `--flip-right` — Flip camera 180° if mounted upside-down

### Keyboard Controls (in --debug mode)
- `q` — Quit
- `r` — Reset state machine back to WAITING

## Hardware

| What | Value |
|------|-------|
| Platform | Radxa Rock 5T, RK3588 SoC, aarch64 Linux |
| Left camera | `/dev/video31` MIPI CSI |
| Right camera | `/dev/video22` MIPI CSI |
| Frame size | 1280 × 720 |
| Stereo baseline | ~11.7 cm |
| Serial port | `/dev/ttyUSB0` @ 115200 baud |
| Steering range | -30° to +30° (integer) |
| Min rolling speed | ~75% throttle (car stalls below) |
| Safe test speed | 40% throttle (NAV_DRIVE_SPEED in config) |
| Python env | `source ~/rv/bin/activate` (Python 3.11, OpenCV, pyserial) |

## Architecture & Data Flow

```
Left cam (video31)   →  MediaPipe Pose  →  PointingDetector  → shoulder_px, wrist_px (left)
                                                                          ↓
Right cam (video22)  →  MediaPipe Pose  →  PointingDetector  → shoulder_px, wrist_px (right)
                                                                          ↓
                               StereoVision.triangulate(left_pts, right_pts)
                                                                          ↓
                               GroundTargetEstimator.estimate()
                               → 3D pointing ray (shoulder→wrist)
                               → ray_plane_intersection with ground
                               → GroundTarget { distance_cm, heading_deg }
                                                                          ↓
                               Navigator.update()
                               → State machine drives car to target
                               → CarController.steer() / .drive()
```

### State Machine (navigator.py)
1. **WAITING** — Detect pointing in both cameras. If stable for 5 frames → lock target → DRIVING
2. **DRIVING** — Visual servo: re-estimate target each frame, steer proportionally (KP * heading), drive at NAV_DRIVE_SPEED. If distance < 30cm → ARRIVED. Safety timeout after 10s.
3. **ARRIVED** — Stop car → TURNING
4. **TURNING** — Full steering lock + slow speed for NAV_TURN_DURATION seconds → WAITING

### Key Design Decisions
- **No feature matching.** MediaPipe gives consistent joint indices (e.g., joint 12 = right_shoulder) in both views. We triangulate by joint index, not by visual correspondence.
- **Ground plane assumption.** Flat floor at `GROUND_PLANE_Y` cm below camera. Valid for indoor lab. Must calibrate this value on the actual car.
- **No odometry.** Car has no wheel encoders. Uses visual servo (re-estimates target every frame while driving) instead of dead reckoning.
- **Two separate PointingDetector instances** — one per camera. Each runs MediaPipe independently.

## Stereo Calibration

The stereo calibration was done at **640×480** (from the Stereo Calibration assignment using a 10×7 checkerboard). `scale_calibration.py` scales the intrinsics (fx, fy, cx, cy) to **1280×720** — R, T, distortion coefficients are resolution-independent.

The scaled calibration lives at `src/calibration/stereo_calib.npz` with keys:
- `K_L`, `K_R`: (3,3) intrinsic matrices (scaled to 1280×720)
- `dist_L`, `dist_R`: (1,5) distortion coefficients
- `R`: (3,3) rotation between cameras
- `T`: (3,1) translation = [-11.58, -0.38, -1.70] cm (~11.7 cm baseline)
- `E`, `F`: essential and fundamental matrices

If accuracy is poor, recalibrate at native 1280×720 with new checkerboard captures.

## What Needs Calibration on the Car

These values MUST be tuned in `config.py` on the actual hardware:

| Parameter | Default | How to calibrate |
|-----------|---------|-----------------|
| `GROUND_PLANE_Y` | 50.0 | Measure camera height above floor in cm |
| `NAV_TURN_DURATION` | 3.0s | Time at full steering lock + 40% throttle for 180° turn |
| `NAV_DRIVE_SPEED` | 40 | Increase if car doesn't move (min rolling is ~75%, but 40% is safe crawl start) |
| `NAV_STEER_KP` | 1.0 | If car oscillates, decrease. If sluggish, increase. |
| `ARRIVE_DISTANCE_CM` | 30 | How close is "close enough" |
| `POINTING_EXTENSION_THRESH` | 0.55 | Lower if pointing isn't detected, raise if false positives |
| Camera flips | off | If camera image is upside-down, add `--flip-left` or `--flip-right` |

## Dependencies

```bash
pip install mediapipe numpy opencv-python pyserial
# Note: mediapipe requires protobuf>=5.28.0
# On Radxa aarch64, mediapipe may need: pip install mediapipe or build from source
```

## Development Guidelines

- **ALWAYS test with `--no-drive` first** before enabling motor commands
- Config parameters go in `config.py`, not inline
- All files use flat imports: `from stereo_vision import StereoVision`
- Don't add lane detection, sign detection, or speed controller code — that lives in the separate CyberTruck repo
- The car_controller.py is shared with CyberTruck — don't modify its API
- Safety: `emergency_stop()` is called on Ctrl+C via signal handler in main_pointing.py

## Likely Next Steps

1. **SSH to Radxa**, clone this repo, install mediapipe in rv venv
2. **Test vision pipeline** with `--no-drive --debug` to verify pointing detection + ground target
3. **Calibrate GROUND_PLANE_Y** by measuring camera height
4. **Calibrate NAV_TURN_DURATION** by testing turning with motor commands
5. **Increase NAV_DRIVE_SPEED** if 40% doesn't move the car (min rolling speed is ~75%)
6. **Full integration test** — point → drive → arrive → turn → wait
7. **Record demo video** for final project submission
8. If stereo accuracy is poor, recalibrate at 1280×720

## Connection to Other Projects

- **CyberTruck repo** (`cybertruck-autonomous`): Lane-following autonomous driving. Shares `car_controller.py` interface. Don't modify that repo.
- **spatial-intent** (thesis research): `mediapipe_extractor.py`, `keypoint_utils.py`, `geometry.py` were copied from there. Stereo code here should eventually port back to `spatial-intent/src/stereo/`.
