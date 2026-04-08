#!/usr/bin/env python3
"""Quick rectification test — draws horizontal lines to verify epipolar alignment."""

import cv2
import numpy as np
from stereo_vision import StereoVision
import config as cfg

stereo = StereoVision("calibration/stereo_calib.npz", (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))

left = cv2.imread("test_left.png")
right = cv2.imread("test_right.png")

left_rect, right_rect = stereo.rectify(left, right)

# Draw horizontal lines across both images to check epipolar alignment
combined = np.hstack([left_rect, right_rect])
for y in range(0, combined.shape[0], 40):
    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

# Save full-res and display scaled
cv2.imwrite("test_rectified.png", combined)
print("Saved test_rectified.png")

display = cv2.resize(combined, (combined.shape[1] // 2, combined.shape[0] // 2))
cv2.imshow("Rectified (green lines should align features)", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
