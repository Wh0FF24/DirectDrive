#!/usr/bin/env python3
"""Quick stereo capture — press SPACE to save, Q to quit."""

import cv2
import numpy as np
import config as cfg

cap_l = cv2.VideoCapture(cfg.CAMERA_LEFT)
cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

cap_r = cv2.VideoCapture(cfg.CAMERA_RIGHT)
cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

if not cap_l.isOpened() or not cap_r.isOpened():
    print("Failed to open cameras")
    exit(1)

print("SPACE = save test_left.png + test_right.png | Q = quit")

while True:
    ret_l, left = cap_l.read()
    ret_r, right = cap_r.read()
    if not ret_l or not ret_r:
        continue

    combined = np.hstack([left, right])
    # Scale down for display
    display = cv2.resize(combined, (combined.shape[1] // 2, combined.shape[0] // 2))
    cv2.imshow("Left | Right", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite("test_left.png", left)
        cv2.imwrite("test_right.png", right)
        print("Saved test_left.png and test_right.png")
    elif key == ord('q'):
        break

cap_l.release()
cap_r.release()
cv2.destroyAllWindows()
