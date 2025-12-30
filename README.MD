# Computer Vision Safety Zone Monitor (YOLOv8)

Real-time webcam monitoring system that detects when a person enters a defined “Safety Zone”.
Uses **edge-triggered events** (ENTER/EXIT) to prevent alert spam, logs events, plays an alarm, and saves violation snapshots.

## Why this matters
This project demonstrates:
- Real-time computer vision inference (YOLOv8 / Ultralytics)
- Event-driven / edge-trigger logic (state change detection)
- Rate limiting (cooldowns for snapshots + alarms)
- Logging and evidence capture (timestamped logs + images)

## Demo
**Video:** (add link here)  ← EDIT  
**Screenshots:** (optional)

## Features
- Detects people via YOLOv8
- Draws configurable safety zone rectangle
- **ENTER** event triggers:
  - log entry
  - snapshot (rate-limited)
  - alarm sound (rate-limited)
- **EXIT** event triggers:
  - log entry
- On-screen overlay:
  - “ALERT: ZONE VIOLATION”
  - violation counter

## Project Structure
cv_safety_system/
├── main.py
├── README.md
├── yolov8n.pt
├── logs/
│   └── events.log
├── violations/
│   └── *_violation.jpg
├── sounds/
│   └── ALARM.WAV

## Author
Ian Hernandez
