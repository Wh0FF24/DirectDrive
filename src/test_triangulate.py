#!/usr/bin/env python3
"""Click a point in both views to triangulate its 3D position.

Usage: python3 test_triangulate.py
  - Click the SAME point in the LEFT window, then the RIGHT window
  - Press R to reset clicks
  - Press Q to quit
"""

import cv2
import numpy as np
from stereo_vision import StereoVision
import config as cfg

stereo = StereoVision("calibration/stereo_calib.npz", (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))

cap_l = cv2.VideoCapture(cfg.CAMERA_LEFT)
cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
cap_r = cv2.VideoCapture(cfg.CAMERA_RIGHT)
cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

click_l = None
click_r = None

def on_click_l(event, x, y, flags, param):
    global click_l
    if event == cv2.EVENT_LBUTTONDOWN:
        click_l = (x, y)
        print(f"  Left click: ({x}, {y})")

def on_click_r(event, x, y, flags, param):
    global click_r
    if event == cv2.EVENT_LBUTTONDOWN:
        click_r = (x, y)
        print(f"  Right click: ({x}, {y})")

cv2.namedWindow("Left")
cv2.namedWindow("Right")
cv2.setMouseCallback("Left", on_click_l)
cv2.setMouseCallback("Right", on_click_r)

print("Click the SAME point in Left window, then Right window")
print("SPACE = freeze | R = reset | Q = quit")

frozen = None
result_text = ""

while True:
    if frozen is None:
        ret_l, left = cap_l.read()
        ret_r, right = cap_r.read()
        if not ret_l or not ret_r:
            continue
    else:
        left, right = frozen

    disp_l = left.copy()
    disp_r = right.copy()

    if click_l is not None:
        cv2.drawMarker(disp_l, click_l, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    if click_r is not None:
        cv2.drawMarker(disp_r, click_r, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    if click_l is not None and click_r is not None:
        pt_l = np.array([click_l], dtype=np.float64)
        pt_r = np.array([click_r], dtype=np.float64)
        pt_3d = stereo.triangulate(pt_l, pt_r)[0]
        depth = pt_3d[2]
        dist = np.linalg.norm(pt_3d)
        result_text = f"[{pt_3d[0]:.0f}, {pt_3d[1]:.0f}, {pt_3d[2]:.0f}]mm  depth={depth/10:.0f}cm"
        cv2.putText(disp_l, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_r, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Left", disp_l)
    cv2.imshow("Right", disp_r)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        click_l = None
        click_r = None
        frozen = None
        result_text = ""
        print("  Reset")
    elif key == ord(' '):
        frozen = (left.copy(), right.copy())
        print("  Frozen")

cap_l.release()
cap_r.release()
cv2.destroyAllWindows()
