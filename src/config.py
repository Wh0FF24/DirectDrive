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
GROUND_PLANE_Y = 170.5             # mm — ground plane Y below camera (measure camera height above floor)
GROUND_SMOOTH_ALPHA = 0.3          # EMA smoothing for ground target (lower = smoother)
GROUND_MAX_JUMP_MM = 1000.0        # mm — outlier rejection: ignore targets jumping more than this
ARRIVE_DISTANCE_MM = 1000          # mm — consider "arrived" when this close (accounts for braking coast)
MIN_RAY_DOWN_DEG = 5.0             # degrees — ray must point at least this far below horizontal

# === Pointing Stability ===
POINTING_STABLE_HEADING_TOL = 15.0   # degrees — max heading change to count as stable
POINTING_STABLE_DISTANCE_TOL = 300.0 # mm — max distance change to count as stable

# === Navigation ===
NAV_DRIVE_SPEED = 75               # % throttle — slow and safe
NAV_STEER_KP = 1.0                 # proportional gain: heading_deg -> steering_deg
NAV_MAX_DRIVE_TIME = 10.0          # seconds — safety timeout
NAV_TURN_DURATION = 5.0            # seconds at full lock for ~180 turn (calibrated on hardware)

# === Stereo ===
STEREO_CALIB_PATH = "calibration/stereo_calib.npz"
