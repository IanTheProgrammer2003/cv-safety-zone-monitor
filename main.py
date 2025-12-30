from ultralytics import YOLO
import cv2
import datetime
import time
import pygame
# --------------------------
# SETTINGS
# --------------------------
SAFETY_ZONE = (400, 20, 630, 200)

SAVE_INTERVAL = 5
ALARM_INTERVAL = 5
last_save_time = 0
last_alarm_time = 0
violation_count = 0

was_inside = False
alert_active = False

# --------------------------
# MODEL + CAMERA
# --------------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

pygame.mixer.init()

# --------------------------
# FUNCTIONS
# --------------------------
def log_event(event):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/events.log", "a") as f:
        f.write(f"[{timestamp}] {event}\n")
    print(event)

def play_alarm():
    pygame.mixer.music.load("sounds/ALARM.WAV")
    pygame.mixer.music.play()

# --------------------------
# MAIN LOOP
# --------------------------
while True:
    # 1) Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # 2) Time for rate-limits
    current_time = time.time()

    # 3) Detect objects
    results = model(frame)
    annotated = results[0].plot()

    # 4) Draw safety zone
    x1, y1, x2, y2 = SAFETY_ZONE
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(annotated, "Safety Zone", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 5) Decide: is ANY person inside this frame?
    inside_zone = False
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        if label == "person":
            px1, py1, px2, py2 = map(int, box.xyxy[0])

            # overlap check
            if px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1:
                inside_zone = True
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)
                break  # one person is enough

    # 6) EDGE LOGIC (ONE TIME PER FRAME)
    if inside_zone and not was_inside:
        log_event(">>> ENTER: PERSON ENTERED SAFETY ZONE <<<")

        if current_time - last_save_time >= SAVE_INTERVAL:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(f"violations/{timestamp}_violation.jpg", annotated)
            last_save_time = current_time
            violation_count += 1

        if current_time - last_alarm_time >= ALARM_INTERVAL:
            play_alarm()
            last_alarm_time = current_time

    elif (not inside_zone) and was_inside:
        log_event("<<< EXIT: PERSON LEFT SAFETY ZONE >>>")

    # 7) Update memory
    was_inside = inside_zone
    alert_active = inside_zone

    # 8) UI overlays
    if alert_active:
        cv2.putText(annotated, "ALERT: ZONE VIOLATION",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.putText(annotated, f"Violations: {violation_count}",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 9) Show window + quit
    cv2.imshow("Safety Detection System - Press Q to Quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
