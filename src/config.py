# config.py — All tunable parameters for DirectDrive (Point-and-Drive)

# === Camera Settings ===
CAMERA_LEFT = 31          # /dev/video31
CAMERA_RIGHT = 22         # /dev/video22
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# === Serial / Car Control ===
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# === Pointing Detection ===
POINTING_EXTENSION_THRESH = 0.50    # min arm extension ratio to count as pointing
POINTING_STABLE_FRAMES = 5         # frames of stable detection before committing
POINTING_CONFIDENCE_GATE = 0.3     # min MediaPipe detection confidence
POINTING_HAND_OPEN_THRESH = 0.35   # min hand openness ratio (open hand = pointing, fist = ignore)

# === Ground Target ===
GROUND_PLANE_Y = 170.5             # ground plane Y in camera frame (mm below camera)
GROUND_SMOOTH_ALPHA = 0.5          # EMA smoothing for ground target
ARRIVE_DISTANCE_CM = 300           # mm — consider "arrived" when this close

# === Navigation ===
NAV_DRIVE_SPEED = 75               # % throttle — slow and safe
NAV_STEER_KP = 1.0                 # proportional gain: heading_deg -> steering_deg
NAV_MAX_DRIVE_TIME = 10.0          # seconds — safety timeout
NAV_TURN_DURATION = 3.0            # seconds at full lock for ~180 turn (CALIBRATE THIS)

# === Stereo ===
STEREO_CALIB_PATH = "calibration/stereo_calib.npz"
