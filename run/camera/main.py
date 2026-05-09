from ultralytics import YOLO
import cv2
import time
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from collections import defaultdict

#pip install ultralytics opencv-python mediapipe numpy pandas streamlit plotly
#python main.py
#streamlit run dashboard.py
# =========================
# LOAD MODELS
# =========================
model_behavior = YOLO(r"best.pt")
model_persons  = YOLO("yolov8n.pt")

# =========================
# MEDIAPIPE
# =========================
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("✅ Downloaded!")

base_options = python.BaseOptions(model_asset_path=model_path)
options      = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    running_mode=vision.RunningMode.IMAGE
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

cv2.namedWindow("Classroom AI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classroom AI", 1280, 720)

FRAME_W = 720
FRAME_H = 480


PROCESS_EVERY = 3
frame_count = 0
last_drawn_frame = None

# =========================
# LESSON TIMER
# =========================
lesson_start = time.time()

# =========================
# SEAT-BASED STABLE ID
# =========================
seats         = {}
track_to_seat = {}
next_seat     = [1]
SEAT_DIST     = 80  

def find_nearest_seat(cx, cy):
    best_id   = None
    best_dist = float("inf")
    for sid, (sx, sy) in seats.items():
        dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        if dist < SEAT_DIST and dist < best_dist:
            best_dist = dist
            best_id   = sid
    return best_id

def get_seat_id(track_id, cx, cy):
    if track_id in track_to_seat:
        sid = track_to_seat[track_id]
        return sid
    existing = find_nearest_seat(cx, cy)
    if existing is not None:
        track_to_seat[track_id] = existing
        return existing
    new_sid = next_seat[0]
    next_seat[0]           += 1
    seats[new_sid]          = (cx, cy)
    track_to_seat[track_id] = new_sid
    return new_sid

# =========================
# DATA STORAGE
# =========================
students = defaultdict(lambda: {
    "status":    "Attentive",
    "behavior":  None,
    "last_seen": time.time(),
})

# =========================
# HEAD POSE
# =========================
def get_head_pose(frame, x1, y1, x2, y2):
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return None

    rgb    = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_mesh.detect(mp_img)

    if not result.facial_transformation_matrixes:
        return None

    matrix = result.facial_transformation_matrixes[0]
    pitch  = np.arcsin(-matrix[2][1]) * 180 / np.pi
    yaw    = np.arctan2(matrix[2][0], matrix[2][2]) * 180 / np.pi

    return pitch, yaw

# =========================

# =========================
def get_attention(pose):
    if pose is None:
        return None

    pitch, yaw = pose

    if abs(yaw) > 25:    
        return "Distracted"

    elif pitch > 20:     
        return "Distracted"
    else:
        return "Attentive"

# =========================

# =========================
def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union == 0:
        return 0
    return inter / union

# =========================
# DRAW INFO
# =========================
def draw_info(frame, x1, y1, x2, y2,
              stable_id, status, behavior, color):

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    corner_len = 15
    corner_t   = 4
    cv2.line(frame, (x1, y1), (x1+corner_len, y1), color, corner_t)
    cv2.line(frame, (x1, y1), (x1, y1+corner_len), color, corner_t)
    cv2.line(frame, (x2, y1), (x2-corner_len, y1), color, corner_t)
    cv2.line(frame, (x2, y1), (x2, y1+corner_len), color, corner_t)
    cv2.line(frame, (x1, y2), (x1+corner_len, y2), color, corner_t)
    cv2.line(frame, (x1, y2), (x1, y2-corner_len), color, corner_t)
    cv2.line(frame, (x2, y2), (x2-corner_len, y2), color, corner_t)
    cv2.line(frame, (x2, y2), (x2, y2-corner_len), color, corner_t)

    texts = [
        f"S{stable_id}",
        f"{status}",
    ]
    if behavior:
        texts.append(f"{behavior}")

    line_h  = 22
    padding = 5
    box_w   = 180
    total_h = len(texts) * line_h + padding * 2
    box_top = max(0, y1 - total_h - 5)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, box_top), (x1+box_w, y1-5), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x1, box_top), (x1+box_w, y1-5), color, 1)

    for i, text in enumerate(texts):
        y_pos = box_top + padding + (i+1) * line_h
        cv2.putText(frame, text, (x1+5, y_pos+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x1+5, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame = cv2.flip(frame, 1)  # 

    frame_count += 1

    # إ
    if frame_count % PROCESS_EVERY != 0:
        if last_drawn_frame is not None:
            cv2.imshow("Classroom AI", last_drawn_frame)
        else:
            cv2.imshow("Classroom AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    current_time    = time.time()
    lesson_duration = current_time - lesson_start

    # =========================
    # STEP 1: best.pt
    #  track
    # =========================
    behavior_results = model_behavior(frame, verbose=False)
    behavior_zones   = []

    for r in behavior_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.6:
                continue
            cls   = int(box.cls[0])
            label = model_behavior.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            behavior_zones.append((cx, cy, label, x1, y1, x2, y2))

    # =========================
    
    # =========================
    filtered_behavior = []
    used_behavior = [False] * len(behavior_zones)
    for i in range(len(behavior_zones)):
        if used_behavior[i]:
            continue
        best = behavior_zones[i]
        for j in range(i + 1, len(behavior_zones)):
            if used_behavior[j]:
                continue
            iou = calc_iou(
                (behavior_zones[i][3], behavior_zones[i][4], behavior_zones[i][5], behavior_zones[i][6]),
                (behavior_zones[j][3], behavior_zones[j][4], behavior_zones[j][5], behavior_zones[j][6])
            )
            if iou > 0.3:
                used_behavior[j] = True
        filtered_behavior.append(best)
    behavior_zones = filtered_behavior

    # =========================
    # STEP 2: yolov8n
    #  track
    # =========================
    person_results = model_persons.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    #  seat_id
    drawn_seats = set()
    # behavior_zone 
    used_bz = set()

    for r in person_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            cls = int(box.cls[0])
            if model_persons.names[cls] != "person":
                continue

            track_id = int(box.id[0]) if box.id is not None else -1
            if track_id < 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            stable_id = get_seat_id(track_id, cx, cy)

            #  seat_id 
            if stable_id in drawn_seats:
                continue
            drawn_seats.add(stable_id)

            # =========================
            # 
            # =========================
            matched_behavior = None

            best_bz_dist = float("inf")
            best_bz_idx  = -1
            for idx, (bcx, bcy, blabel, bx1, by1, bx2, by2) in enumerate(behavior_zones):
                if idx in used_bz:
                    continue
                dist = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5
                if dist < SEAT_DIST and dist < best_bz_dist:
                    best_bz_dist = dist
                    best_bz_idx  = idx
                    matched_behavior = blabel

            if best_bz_idx >= 0:
                used_bz.add(best_bz_idx)

            if matched_behavior is not None:
                # =========================
                # Engaged 
                # =========================
                status = "Engaged"
                color  = (0, 255, 0) if matched_behavior == "reading_writing" else (0, 255, 255)

            else:
                # =========================
                
                # =========================
                pose      = get_head_pose(frame, x1, y1, x2, y2)
                attention = get_attention(pose)

                if attention is None:
                    continue  # 

                status = attention

                if attention == "Attentive":
                    color = (255, 165, 0)    
                elif attention == "Distracted":
                    color = (0, 0, 255)      


            students[stable_id]["status"]    = status
            students[stable_id]["behavior"]  = matched_behavior
            students[stable_id]["last_seen"] = current_time

            draw_info(frame, x1, y1, x2, y2,
                      stable_id, status, matched_behavior, color)

    # =========================
    # SAVE FOR DASHBOARD
    # =========================
    save_data = {}
    for sid, d in students.items():
        save_data[str(sid)] = {
            "status":          d["status"],
            "behavior":        d["behavior"],
            "last_seen":       d["last_seen"],
            "lesson_duration": round(lesson_duration, 1)
        }

    with open("data.json", "w") as f:
        json.dump(save_data, f)

    # =========================
    # LEGEND
    # =========================
    legends = [
        ("Engaged - Reading",   (0, 255, 0)),
        ("Engaged - RaiseHand", (0, 255, 255)),
        ("Attentive",           (255, 165, 0)),
        ("Distracted",          (0, 0, 255)),

    ]
    for i, (text, color) in enumerate(legends):
        cv2.putText(frame, text,
                    (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)

    last_drawn_frame = frame.copy()  
    cv2.imshow("Classroom AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# FINAL REPORT
# =========================
lesson_duration = time.time() - lesson_start
lesson_minutes  = round(lesson_duration / 60, 1)

print(f"\n===== FINAL LESSON REPORT =====")
print(f"Lesson Duration: {lesson_minutes} minutes\n")

for sid, d in students.items():
    print(f"Seat {sid}: "
          f"Status={d['status']} | "
          f"Action={d['behavior']}")

cap.release()
cv2.destroyAllWindows()
