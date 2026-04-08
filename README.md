# DirectDrive

ECEN 631 Final Project -- Point-and-Drive with Stereo Vision.

A person points at a spot on the ground. The car sees the pointing gesture
with stereo cameras, triangulates the 3D ground target, drives there,
turns 180 degrees, and waits for the next command.

## Architecture

```
left cam  -->  MediaPipe Pose  -->  PointingDetector  --\
                                                         |--> GroundTargetEstimator --> Navigator --> CarController
right cam -->  MediaPipe Pose  -->  PointingDetector  --/
                    |
              StereoVision (rectify + triangulate)
```

## Quick Start

```bash
# Desktop testing (single webcam, no driving)
cd src
python main_pointing.py --debug --no-drive --webcam

# On Radxa (full stereo + driving)
source ~/rv/bin/activate
python main_pointing.py --debug
```

## Dependencies

- Python 3.11+
- OpenCV
- MediaPipe
- NumPy

## Files

| File | Purpose |
|------|---------|
| `main_pointing.py` | Entry point |
| `config.py` | All tunable parameters |
| `car_controller.py` | Arduino serial interface (steer, drive) |
| `stereo_vision.py` | Load stereo calibration, rectify, triangulate |
| `pointing_detector.py` | MediaPipe pose + arm extension detection |
| `ground_target.py` | Ray-ground plane intersection for 3D target |
| `navigator.py` | State machine: WAITING -> DRIVING -> ARRIVED -> TURNING |
| `mediapipe_extractor.py` | MediaPipe Pose wrapper |
| `keypoint_utils.py` | Arm vector / extension ratio math |
| `geometry.py` | Ray-plane intersection, vector utils |
| `scale_calibration.py` | Scale 640x480 calibration to 1280x720 |

## Calibration

Stereo calibration was done at 640x480 (from Stereo Calibration assignment).
`scale_calibration.py` scales it to 1280x720 for the runtime cameras.
The scaled `.npz` lives at `src/calibration/stereo_calib.npz`.
