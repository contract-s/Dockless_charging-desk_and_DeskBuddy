# charging-desk

A plexiglass pane with a wireless (Qi) charging coil underneath that moves to wherever you set your phone down.

An overhead camera finds the phone, a 2-axis gantry under the pane slides the coil beneath it, and charging starts automatically.

## How it works

1. **Detect** — a webcam looks down at the pane; YOLOv8 (pretrained COCO "cell phone" class) finds the phone.
2. **Map** — a one-time 4-point homography converts pixel coordinates into real desk coordinates (cm).
3. **Move** — the target position is sent over serial to an Arduino running GRBL, which drives the two stepper axes.

## Files

- `yolo_phone_tracker.py` — camera → phone detection → desk coordinates → serial target

## Setup

```bash
pip install ultralytics opencv-python numpy pyserial
python yolo_phone_tracker.py --calibrate   # click the 4 corners of the charging area, once
python yolo_phone_tracker.py               # start tracking
```

The script is set up for an 18 x 24 in plexiglass pane; mount the camera about 80 cm (31 in) above it. Change `DESK_WIDTH_CM` / `DESK_DEPTH_CM` if your pane or reachable area differs. On the Raspberry Pi, run with `--no-display` and set `SEND_SERIAL = True`.
