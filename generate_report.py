#!/usr/bin/env python3
"""Generate ECEN 631 DirectDrive Final Project Report PDF."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
import os

OUTPUT = "ECEN631_DirectDrive_Report_Team1.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    topMargin=0.8*inch,
    bottomMargin=0.8*inch,
    leftMargin=1*inch,
    rightMargin=1*inch,
)

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    'ReportTitle', parent=styles['Title'],
    fontSize=22, spaceAfter=6, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    'Subtitle', parent=styles['Normal'],
    fontSize=12, alignment=TA_CENTER, textColor=HexColor('#555555'),
    spaceAfter=20,
))
styles.add(ParagraphStyle(
    'SectionHead', parent=styles['Heading1'],
    fontSize=14, spaceBefore=18, spaceAfter=8,
    textColor=HexColor('#1a1a2e'),
))
styles.add(ParagraphStyle(
    'SubHead', parent=styles['Heading2'],
    fontSize=12, spaceBefore=12, spaceAfter=6,
    textColor=HexColor('#2d2d44'),
))
styles.add(ParagraphStyle(
    'Body', parent=styles['Normal'],
    fontSize=10.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=8,
))
styles.add(ParagraphStyle(
    'Caption', parent=styles['Normal'],
    fontSize=9, alignment=TA_CENTER, textColor=HexColor('#666666'),
    spaceBefore=4, spaceAfter=12,
))

TABLE_STYLE = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8f8f8')]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
])

story = []

# ============================================================
# TITLE PAGE
# ============================================================
story.append(Spacer(1, 1.5*inch))
story.append(Paragraph("ECEN 631 - Robotic Vision", styles['ReportTitle']))
story.append(Spacer(1, 8))
story.append(Paragraph("DirectDrive: Point-and-Drive with Stereo Vision", styles['ReportTitle']))
story.append(Spacer(1, 20))
story.append(Paragraph("Team 1: Direct Drive", styles['Subtitle']))
story.append(Paragraph("William Weigeshoff &amp; Matthew Watler", styles['Subtitle']))
story.append(Spacer(1, 12))
story.append(Paragraph("Brigham Young University", styles['Subtitle']))
story.append(Paragraph("April 2026", styles['Subtitle']))
story.append(PageBreak())

# ============================================================
# 1. INTRODUCTION
# ============================================================
story.append(Paragraph("1. Introduction", styles['SectionHead']))
story.append(Paragraph(
    "For our final project we built a gesture-controlled RC car. You point at a spot "
    "on the ground, and the car drives there. It uses two cameras in a stereo setup to "
    "figure out where in 3D space you're pointing, then drives to that spot on its own.",
    styles['Body']
))
story.append(Paragraph(
    "We reused the Radxa RK3588 board and Traxxas RC car from our CyberTruck "
    "lane-following project. Both MIPI CSI cameras feed into MediaPipe for pose "
    "estimation, stereo triangulation gives us a 3D ground target, and the car steers "
    "toward it in a closed loop. Everything runs in real time on the Radxa.",
    styles['Body']
))

# ============================================================
# 2. SYSTEM ARCHITECTURE
# ============================================================
story.append(Paragraph("2. System architecture", styles['SectionHead']))
story.append(Paragraph(
    "The code is split into modules that each do one thing. Frames come in from both "
    "cameras, go through pose estimation, then stereo triangulation figures out the "
    "3D target. A state machine decides what the car does next. All the tunable "
    "numbers live in config.py.",
    styles['Body']
))

arch_data = [
    ['Module', 'File', 'What it does'],
    ['Pointing Detector', 'pointing_detector.py', 'MediaPipe pose + arm extension check'],
    ['Stereo Vision', 'stereo_vision.py', 'Load calibration, undistort, triangulate'],
    ['Ground Target', 'ground_target.py', 'Ray-plane intersection to find ground point'],
    ['Navigator', 'navigator.py', 'State machine: WAITING / DRIVING / ARRIVED'],
    ['Car Controller', 'car_controller.py', 'Serial commands to Arduino (steer, drive)'],
    ['Config', 'config.py', 'All tunable parameters'],
    ['Main Loop', 'main_pointing.py', 'Cameras, main loop, debug HUD'],
]
t = Table(arch_data, colWidths=[1.3*inch, 1.5*inch, 3.2*inch])
t.setStyle(TABLE_STYLE)
story.append(t)
story.append(Paragraph("Table 1: Software modules.", styles['Caption']))

story.append(Paragraph("2.1 Hardware", styles['SubHead']))
hw_data = [
    ['Component', 'Details'],
    ['Compute', 'Radxa Rock 5T, RK3588 (aarch64, 8-core)'],
    ['Left Camera', 'MIPI CSI @ 1280x720 (/dev/video31)'],
    ['Right Camera', 'MIPI CSI @ 1280x720 (/dev/video22)'],
    ['Stereo Baseline', '~11.7 cm'],
    ['Vehicle', 'Traxxas Splash 4x4 RC car'],
    ['Motor Control', 'Arduino via USB serial (115200 baud)'],
    ['Steering', '-30 to +30 degrees'],
    ['Pose Estimation', 'MediaPipe Pose (CPU)'],
]
t2 = Table(hw_data, colWidths=[1.6*inch, 4.4*inch])
t2.setStyle(TABLE_STYLE)
story.append(t2)
story.append(Paragraph("Table 2: Hardware.", styles['Caption']))

# ============================================================
# 3. POINTING DETECTION
# ============================================================
story.append(Paragraph("3. Pointing detection", styles['SectionHead']))
story.append(Paragraph(
    "We didn't train a custom gesture model. MediaPipe Pose already gives 33 body "
    "landmarks per frame, so we just check whether the arm is extended enough to count "
    "as pointing.",
    styles['Body']
))

story.append(Paragraph("3.1 How it works", styles['SubHead']))
story.append(Paragraph(
    "Each camera has its own MediaPipe instance. We compute an arm extension ratio "
    "(how straight the arm is, from 0 to 1) and if it is above 0.50, we call it a "
    "point. Both cameras have to agree on which arm is extended before we accept it. "
    "That requirement alone gets rid of most false positives.",
    styles['Body']
))

story.append(Paragraph("3.2 Stereo correspondence by joint index", styles['SubHead']))
story.append(Paragraph(
    "We didn't need feature matching between the two views. MediaPipe gives consistent "
    "joint indices (joint 12 is always the right shoulder, for example), so we "
    "triangulate by index rather than by visual similarity. This works well even when "
    "the person looks quite different from each camera's angle.",
    styles['Body']
))

# ============================================================
# 4. STEREO VISION AND GROUND TARGET
# ============================================================
story.append(Paragraph("4. Stereo vision and ground target", styles['SectionHead']))

story.append(Paragraph("4.1 Calibration", styles['SubHead']))
story.append(Paragraph(
    "We calibrated the stereo pair at 1280x720 with a 10x7 checkerboard. That gave "
    "us intrinsic matrices, distortion coefficients, and the rotation/translation "
    "between cameras. The baseline is about 11.7 cm.",
    styles['Body']
))

story.append(Paragraph("4.2 Triangulation", styles['SubHead']))
story.append(Paragraph(
    "We take matching pixel coordinates from both cameras (shoulder, wrist, index "
    "fingertip), undistort them, and pass them to OpenCV's triangulatePoints. That "
    "gives us 3D positions in the left camera's coordinate frame.",
    styles['Body']
))

story.append(Paragraph("4.3 Finding the ground target", styles['SubHead']))
story.append(Paragraph(
    "With the shoulder and index fingertip in 3D, we form a ray from shoulder through "
    "fingertip and intersect it with the ground plane (170.5 mm below the camera). "
    "We originally used the wrist as the ray endpoint, but the fingertip turned out "
    "to be much better. The wrist only tells you where the arm is aimed; the fingertip "
    "tells you where the finger is actually pointing.",
    styles['Body']
))
story.append(Paragraph(
    "To keep things stable we run an EMA filter (alpha = 0.3), reject outliers that "
    "jump more than 1 meter between frames, and require the ray to be aimed at least "
    "5 degrees downward. The target also has to hold steady for 5 frames before the "
    "car will commit to driving. Without that, quick arm movements and noisy "
    "detections would trigger false starts.",
    styles['Body']
))

# ============================================================
# 5. NAVIGATION
# ============================================================
story.append(Paragraph("5. Navigation", styles['SectionHead']))
story.append(Paragraph(
    "A state machine handles the drive cycle:",
    styles['Body']
))

states_data = [
    ['State', 'What happens', 'Next state'],
    ['WAITING', 'Looking for a stable pointing gesture', 'Stable for 5 frames -> DRIVING'],
    ['DRIVING', 'Steering toward the target', 'Close enough or timeout -> ARRIVED'],
    ['ARRIVED', 'Stop the car', 'Immediately -> WAITING'],
]
t3 = Table(states_data, colWidths=[1.0*inch, 2.4*inch, 2.6*inch])
t3.setStyle(TABLE_STYLE)
story.append(t3)
story.append(Paragraph("Table 3: Navigation states.", styles['Caption']))

story.append(Paragraph(
    "While driving, the car re-estimates the target every frame. Steering is "
    "proportional (steer = K<sub>P</sub> * heading_angle, clamped to +/-30 degrees) "
    "and speed is fixed at 75% throttle. A 10 second timeout stops the car if "
    "something goes wrong.",
    styles['Body']
))
story.append(Paragraph(
    "The car has no wheel encoders or IMU, so we can't dead reckon. Instead it just "
    "keeps looking at the target through the cameras. This works fine, but it means "
    "the person has to keep pointing while the car drives.",
    styles['Body']
))

# ============================================================
# 6. DEVELOPMENT PROCESS
# ============================================================
story.append(Paragraph("6. How we built it", styles['SectionHead']))
story.append(Paragraph(
    "We started with the stereo calibration (checkerboard at 1280x720) and verified "
    "triangulation by measuring objects at known distances. Then we built the pointing "
    "detector on a laptop with a single webcam using a --webcam flag that duplicates "
    "the frame as fake stereo. We tuned the extension threshold by testing with a "
    "few different people at different distances.",
    styles['Body']
))
story.append(Paragraph(
    "After that we wired up the stereo triangulation with the ray-plane intersection. "
    "We were using the wrist as the ray endpoint at first, but it was too inaccurate, "
    "so we switched to the index fingertip. That was a big improvement. We added "
    "EMA smoothing and outlier rejection on top to deal with noise.",
    styles['Body']
))
story.append(Paragraph(
    "For driving, we tested in --no-drive mode first to make sure state transitions "
    "fired correctly, then turned on motor commands and tuned speed, steering gain, "
    "and arrival distance on the actual car.",
    styles['Body']
))

# ============================================================
# 7. RESULTS AND VIDEO
# ============================================================
story.append(Paragraph("7. Results and video", styles['SectionHead']))
story.append(Paragraph(
    "The car detects pointing and drives toward the target. The visual servo "
    "keeps it roughly on course while moving. The person does need to stay visible "
    "and pointing while the car drives, since there is no odometry. If they move "
    "out of frame, the car holds its last heading until the timeout stops it.",
    styles['Body']
))
story.append(Paragraph(
    "Demo video: <link href=\"https://drive.google.com/file/d/1Vibrvtulc27LQvVIg_UyHWadJ3H6KQDe/view?usp=drive_link\">"
    "https://drive.google.com/file/d/1Vibrvtulc27LQvVIg_UyHWadJ3H6KQDe/view</link>",
    styles['Body']
))

# ============================================================
# 8. LESSONS LEARNED
# ============================================================
story.append(Paragraph("8. Lessons learned", styles['SectionHead']))
story.append(Paragraph(
    "We tried scaling a lower-res calibration up to 1280x720 at first, and "
    "triangulation was way off. Recalibrating at the actual runtime resolution "
    "fixed it. Use the fingertip as the ray endpoint, not the wrist. The wrist "
    "only tells you arm direction, not where the finger is aimed. Also, camera "
    "height above the floor is surprisingly sensitive. A few centimeters off "
    "shifts the ground target by meters at longer distances. We had to measure "
    "it carefully and verify by pointing at known spots on the floor.",
    styles['Body']
))

# ============================================================
# BUILD
# ============================================================
doc.build(story)
print(f"Report generated: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT) / 1024:.0f} KB")
