"""
yolo_phone_tracker.py

Detects a phone in a webcam feed using YOLOv8's pretrained COCO model
(class 67 = "cell phone"), converts the detected pixel position into
real-world desk coordinates via a one-time homography calibration, and
prints/sends the target position. Meant to run on your laptop first
(webcam index 0), then move unchanged onto the Raspberry Pi later.

Install:
    pip install ultralytics opencv-python numpy pyserial

Run:
    python yolo_phone_tracker.py
    python yolo_phone_tracker.py --calibrate   (run this first, once)
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
# Logitech C270. Use a 4:3 mode: the 18x24 in pane is also 4:3, so the full
# frame maps onto the pane with nothing wasted. 720p (16:9) crops the top and
# bottom of the sensor, which would force the camera much higher to see the
# 18 in side. 640x480 is ~1 mm per pixel across the pane, and YOLO resizes
# to 640 internally anyway.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_NAME = "yolov8n.pt"          # nano model: smallest, fastest, good first choice
CELL_PHONE_CLASS_ID = 67           # COCO class index for "cell phone"
CONFIDENCE_THRESHOLD = 0.4
CALIBRATION_FILE = "calibration.json"

# Real-world size in cm of the area you click during calibration.
# Set to the full 24 x 18 in plexiglass pane (24 in along the camera's
# width, 18 in along its height). Once the gantry is built, shrink these to
# the area the coil can actually reach, and click that area's corners instead.
DESK_WIDTH_CM = 60.96   # 24 in
DESK_DEPTH_CM = 45.72   # 18 in

# Smoothing: how many recent positions to average
SMOOTH_WINDOW = 5

# Only send a new target if the phone moved more than this many cm
MOVE_TOLERANCE_CM = 1.5

# If a serial connection to the motion controller is available, set this to
# the right port (e.g. "/dev/ttyUSB0" or "COM5") and set SEND_SERIAL = True.
SEND_SERIAL = False
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200


