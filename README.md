# SnapFire AI

## On-Device Multimodal Fire Detection and False-Alarm Reduction

SnapFire AI is an edge-AI prototype that combines camera-based fire/smoke detection with environmental sensor evidence to support more informed fire-event decisions.

## Key Features

- YOLO-based fire and smoke detection
- Real-time camera processing using OpenCV
- Temporal confirmation of visual detections
- Sensor evidence using temperature, humidity, TVOC, eCO2 and particulate measurements
- Multimodal fusion of vision and sensor evidence
- NORMAL, WARNING and FIRE event states
- PySide6 graphical monitoring dashboard
- Audible warning and fire alarm feedback
- Designed for local/on-device edge inference

## Technology Stack

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- pandas
- scikit-learn
- PySide6
- Pygame

## Running the Prototype

Create and activate a Python virtual environment and install the required packages.

Run the dashboard with:

    python dashboard/app_v4.py

The current prototype uses the YOLO V2 model located at:

    models/fire_smoke_yolo_v2/weights/best.pt

## Project Structure

    dashboard/    - graphical dashboard applications
    data/         - sensor datasets
    models/       - trained model files
    src/          - detection, sensor, fusion and training code
    assets/       - alarm audio files
    docs/         - project documentation

## Edge Deployment

The architecture is intended for on-device inference and is proposed for optimization and deployment on Snapdragon-powered HP PCs.

## Limitations

This is a research and engineering prototype and is NOT a safety-certified fire alarm system.

The current vision model can produce false-positive detections in some normal scenes. Sensor-model experiments also showed variation across different dataset sessions, so the live prototype uses transparent rule-based sensor evidence rather than claiming a general high-accuracy sensor ML model.

Further work includes Snapdragon hardware validation, model optimization, broader real-world testing and additional validation.

## Project Status

Working prototype with integrated vision, sensor evidence, temporal confirmation, multimodal fusion, dashboard visualization and alarm feedback.
