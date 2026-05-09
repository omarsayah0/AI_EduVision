import os
import cv2
import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_DIR = os.path.join(BASE_DIR, "camera")

STUDENT_IDS = ["S1", "S2", "S3"]
STUDENT_NAMES = {"S1": "abdallah", "S2": "anas", "S3": "omar"}
KNOWN_SUBJECTS = {"math", "arabic", "english"}

ACADEMIC_LEVEL_LABELS = {
    1: ("Good", "Low Academic Risk"),
    2: ("Needs Monitoring", "Medium Academic Risk"),
    3: ("Needs Support", "High Academic Risk"),
}

_camera_models_cache = None


def load_camera_models():
    """Load YOLO behavior + person models and MediaPipe face landmarker.
    Returns (models_dict, error_string). error_string is None on success."""
    global _camera_models_cache
    if _camera_models_cache is not None:
        return _camera_models_cache, None

    try:
        from ultralytics import YOLO
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        behavior_path = os.path.join(CAMERA_DIR, "best.pt")
        person_path = os.path.join(CAMERA_DIR, "yolov8n.pt")
        landmarker_path = os.path.join(CAMERA_DIR, "face_landmarker.task")

        for p, label in [
            (behavior_path, "best.pt"),
            (person_path, "yolov8n.pt"),
            (landmarker_path, "face_landmarker.task"),
        ]:
            if not os.path.exists(p):
                return None, f"Missing model file: {label}"

        model_behavior = YOLO(behavior_path)
        model_persons = YOLO(person_path)

        base_options = mp_python.BaseOptions(model_asset_path=landmarker_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
        )
        face_mesh = vision.FaceLandmarker.create_from_options(options)

        _camera_models_cache = {
            "behavior": model_behavior,
            "persons": model_persons,
            "face_mesh": face_mesh,
        }
        return _camera_models_cache, None

    except Exception as e:
        return None, str(e)


def _get_head_pose(face_mesh, frame, x1, y1, x2, y2):
    import mediapipe as mp

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_mesh.detect(mp_img)
    if not result.facial_transformation_matrixes:
        return None
    m = result.facial_transformation_matrixes[0]
    pitch = np.arcsin(-m[2][1]) * 180 / np.pi
    yaw = np.arctan2(m[2][0], m[2][2]) * 180 / np.pi
    return pitch, yaw


def _attention_from_pose(pose):
    if pose is None:
        return None
    pitch, yaw = pose
    if abs(yaw) > 25 or pitch > 20:
        return "Distracted"
    return "Attentive"


def parse_session_filename(filename):
    """
    Extract (day, subject, session_id) from a filename like day1_math.mp4.

    Handles underscores, hyphens, spaces, and mixed case.
    Returns ('unknown', 'unknown', stem) only if parsing completely fails.
    """
    # Work only with the bare filename, no directory, no extension
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.lower().strip()
    # Normalize separators so "day1-math" and "day1 math" both become "day1_math"
    stem_norm = stem.replace("-", "_").replace(" ", "_")
    parts = [p for p in stem_norm.split("_") if p]  # drop empty parts

    # Strategy 1: look for dayX immediately followed by a known subject
    for i in range(len(parts) - 1):
        p = parts[i]
        q = parts[i + 1]
        if p.startswith("day") and p[3:].isdigit() and q in KNOWN_SUBJECTS:
            return p, q, f"{p}_{q}"

    # Strategy 2: dayX followed by anything (unknown subject but valid day)
    for i in range(len(parts) - 1):
        p = parts[i]
        if p.startswith("day") and p[3:].isdigit():
            return p, parts[i + 1], f"{p}_{parts[i + 1]}"

    # Strategy 3: first two parts as-is (original simple approach)
    if len(parts) >= 2 and parts[0].startswith("day"):
        return parts[0], parts[1], f"{parts[0]}_{parts[1]}"

    return "unknown", "unknown", stem_norm or stem


def process_video(video_path, session_id, day, subject, process_every=5, progress_cb=None):
    """
    Process a combined session video (S1=left, S2=middle, S3=right).
    Returns (list_of_result_dicts, warning_str).
    warning_str is None when real inference ran; otherwise describes the fallback used.
    """
    # Always normalise incoming metadata
    day = (day or "unknown").lower().strip()
    subject = (subject or "unknown").lower().strip()
    session_id = (session_id or "unknown").lower().strip()

    models, err = load_camera_models()
    if models is None:
        warning = f"Camera models unavailable ({err}). Using fallback placeholder data."
        return _fallback_results(session_id, day, subject), warning

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _fallback_results(session_id, day, subject), "Could not open video file. Using fallback data."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = max(1, total_frames // process_every)

    student_stats = {
        sid: {"attentive": 0, "distracted": 0, "engaged": 0,
              "reading": 0, "raise_hand": 0, "total": 0}
        for sid in STUDENT_IDS
    }

    model_behavior = models["behavior"]
    model_persons = models["persons"]
    face_mesh = models["face_mesh"]

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % process_every != 0:
            continue

        processed += 1
        if progress_cb:
            progress_cb(min(processed / max(frames_to_process, 1), 1.0))

        frame = cv2.resize(frame, (720, 480))

        # Behavior detections (reading_writing, raise_hand, …)
        beh_results = model_behavior(frame, verbose=False)
        behavior_zones = []
        for r in beh_results:
            for box in r.boxes:
                if float(box.conf[0]) < 0.5:
                    continue
                cls = int(box.cls[0])
                label = model_behavior.names[cls]
                x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                cx = (x1b + x2b) // 2
                behavior_zones.append((cx, label))

        # Person detections — sort left→right to assign S1, S2, S3
        per_results = model_persons(frame, verbose=False)
        persons = []
        for r in per_results:
            for box in r.boxes:
                if float(box.conf[0]) < 0.4:
                    continue
                if model_persons.names[int(box.cls[0])] != "person":
                    continue
                x1p, y1p, x2p, y2p = map(int, box.xyxy[0])
                cx = (x1p + x2p) // 2
                persons.append((cx, x1p, y1p, x2p, y2p))

        if not persons:
            continue

        persons.sort(key=lambda p: p[0])
        persons = persons[:3]

        for idx, (cx, x1p, y1p, x2p, y2p) in enumerate(persons):
            sid = STUDENT_IDS[idx]
            sf = student_stats[sid]

            matched_behavior = None
            for bcx, blabel in behavior_zones:
                if abs(bcx - cx) < 120:
                    matched_behavior = blabel
                    break

            if matched_behavior:
                sf["engaged"] += 1
                if matched_behavior == "reading_writing":
                    sf["reading"] += 1
                elif matched_behavior == "raise_hand":
                    sf["raise_hand"] += 1
            else:
                pose = _get_head_pose(face_mesh, frame, x1p, y1p, x2p, y2p)
                attention = _attention_from_pose(pose)
                if attention == "Attentive":
                    sf["attentive"] += 1
                elif attention == "Distracted":
                    sf["distracted"] += 1

            sf["total"] += 1

    cap.release()

    results = []
    for sid in STUDENT_IDS:
        sf = student_stats[sid]
        total = sf["total"] if sf["total"] > 0 else 1
        focused = sf["attentive"] + sf["engaged"]
        focus_score = round(focused / total * 100)

        reading_ratio = sf["reading"] / total
        raise_ratio = sf["raise_hand"] / total
        if reading_ratio > 0.2 or raise_ratio > 0.1:
            engagement = "Good"
        elif focus_score >= 60:
            engagement = "Acceptable"
        else:
            engagement = "Low"

        counts = {
            "Engaged": sf["engaged"],
            "Attentive": sf["attentive"],
            "Distracted": sf["distracted"],
        }
        camera_status = max(counts, key=counts.get)

        results.append({
            "student_id": sid,
            "day": day,
            "subject": subject,
            "session_id": session_id,
            "focus_score": focus_score,
            "engagement_status": engagement,
            "camera_status": camera_status,
            "confidence": round(focused / total, 2),
        })

    return results, None


def _fallback_results(session_id, day, subject):
    """Deterministic placeholder results when camera models are unavailable."""
    base_focus = {"S1": 45, "S2": 65, "S3": 80}
    seed = abs(hash(session_id)) % 20
    results = []
    for sid in STUDENT_IDS:
        focus = base_focus[sid] + (seed % 11) - 5
        focus = max(15, min(92, focus))
        engagement = "Good" if focus >= 70 else ("Acceptable" if focus >= 50 else "Low")
        results.append({
            "student_id": sid,
            "day": day,
            "subject": subject,
            "session_id": session_id,
            "focus_score": focus,
            "engagement_status": engagement,
            "camera_status": "Attentive" if focus >= 55 else "Distracted",
            "confidence": round(focus / 100, 2),
        })
    return results


def read_watch_csv(csv_file):
    """
    Read watch CSV with columns: student_id, day, subject, session_id, stress_level, bpm.
    Raises ValueError with a clear message if required columns are missing.
    """
    required = {"student_id", "day", "subject", "session_id", "stress_level", "bpm"}
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Watch CSV is missing required columns: {sorted(missing)}")
    df["student_id"] = df["student_id"].astype(str).str.strip().str.upper()
    df["stress_level"] = df["stress_level"].astype(str).str.strip().str.lower()
    df["session_id"] = df["session_id"].astype(str).str.strip().str.lower()
    df["day"] = df["day"].astype(str).str.strip().str.lower()
    df["subject"] = df["subject"].astype(str).str.strip().str.lower()
    return df


def read_academic_csv(csv_file):
    """
    Read academic CSV with columns:
      student_id, subject, homework_commitment, month1_exam, month2_exam, absence, academic_level
    Raises ValueError with a clear message if required columns are missing.
    """
    required = {
        "student_id", "subject",
        "homework_commitment", "month1_exam", "month2_exam",
        "absence", "academic_level",
    }
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Academic CSV is missing required columns: {sorted(missing)}")
    df["student_id"] = df["student_id"].astype(str).str.strip().str.upper()
    df["subject"] = df["subject"].astype(str).str.strip().str.lower()
    return df


def get_interpreted_status(focus_score, engagement, stress_level):
    """Map focus / engagement / stress to a teacher-friendly status label."""
    stress = str(stress_level).lower().strip()
    if focus_score >= 70 and stress == "high":
        return "Focused but stressed"
    if focus_score >= 70 and engagement == "Good":
        return "Stable / Good engagement"
    if focus_score >= 70:
        return "Focused"
    if focus_score >= 50 and stress == "high":
        return "Needs attention (stressed)"
    if focus_score >= 50:
        return "Acceptable / Needs monitoring"
    return "Low engagement / Needs attention"


def fuse_data(camera_results, watch_df):
    """
    Merge camera results with watch data on student_id + session_id.
    Falls back to student_id + day + subject if session_id merge yields no matches.
    Returns merged DataFrame with stress_level, bpm, and interpreted_status columns.
    """
    df_cam = pd.DataFrame(camera_results)
    # Normalise camera columns
    df_cam["student_id"] = df_cam["student_id"].astype(str).str.strip().str.upper()
    df_cam["session_id"] = df_cam["session_id"].astype(str).str.strip().str.lower()
    df_cam["day"] = df_cam["day"].astype(str).str.strip().str.lower()
    df_cam["subject"] = df_cam["subject"].astype(str).str.strip().str.lower()

    df_watch = watch_df[["student_id", "session_id", "stress_level", "bpm"]].copy()

    merged = df_cam.merge(df_watch, on=["student_id", "session_id"], how="left")

    # If most rows are unmatched, try day+subject as fallback key
    unmatched = merged["stress_level"].isna().sum()
    if unmatched > len(merged) * 0.5:
        df_watch2 = watch_df[["student_id", "day", "subject", "stress_level", "bpm"]].copy()
        merged2 = df_cam.merge(df_watch2, on=["student_id", "day", "subject"], how="left")
        if merged2["stress_level"].isna().sum() < unmatched:
            merged = merged2

    merged["stress_level"] = merged["stress_level"].fillna("normal")
    merged["bpm"] = merged["bpm"].fillna(0).astype(int)
    merged["interpreted_status"] = merged.apply(
        lambda r: get_interpreted_status(r["focus_score"], r["engagement_status"], r["stress_level"]),
        axis=1,
    )
    return merged


def generate_mistral_report(all_results_df, academic_df, api_key, student_names):
    """Call Mistral AI and return a weekly teacher report as a string."""
    summaries = []

    for sid in STUDENT_IDS:
        name = student_names.get(sid, sid)
        s_data = all_results_df[all_results_df["student_id"] == sid].copy()
        if s_data.empty:
            continue

        avg_focus = round(s_data["focus_score"].mean())

        # Build per-subject behavioral summary
        subject_lines = []
        for subj in ["math", "arabic", "english"]:
            sub = s_data[s_data["subject"] == subj]
            if sub.empty:
                continue
            sf = round(sub["focus_score"].mean())
            eng = sub["engagement_status"].value_counts().index[0]
            stress_counts = sub["stress_level"].value_counts().to_dict()
            dom_stress = max(stress_counts, key=stress_counts.get)
            high_count = stress_counts.get("high", 0)
            subject_lines.append(
                f"    {subj.capitalize()}: avg_focus={sf}%, engagement={eng}, "
                f"dominant_stress={dom_stress}, high_stress_sessions={high_count}"
            )

        # Build per-subject academic summary
        acad_lines = []
        acad_student = academic_df[academic_df["student_id"] == sid] if academic_df is not None else pd.DataFrame()
        if not acad_student.empty:
            for subj in ["math", "arabic", "english"]:
                row = acad_student[acad_student["subject"] == subj]
                if row.empty:
                    continue
                r = row.iloc[0]
                level = int(r.get("academic_level", 2))
                level_lbl = ACADEMIC_LEVEL_LABELS.get(level, ("Unknown", "Unknown"))
                hw = "Yes" if int(r.get("homework_commitment", 0)) else "No"
                acad_lines.append(
                    f"    {subj.capitalize()}: homework={hw}, "
                    f"month1={r.get('month1_exam','N/A')}/100, "
                    f"month2={r.get('month2_exam','N/A')}/100, "
                    f"absences={r.get('absence','N/A')}, "
                    f"level={level} ({level_lbl[0]} / {level_lbl[1]})"
                )
        else:
            acad_lines = ["    Academic data not available"]

        total_high_stress = int((s_data["stress_level"] == "high").sum())

        summaries.append(
            f"Student {sid} ({name}):\n"
            f"  Overall average focus: {avg_focus}%\n"
            f"  Behavioral data by subject (watch stress is an indicator only, not medical):\n"
            + "\n".join(subject_lines) + "\n"
            f"  Total high-stress sessions: {total_high_stress}/9\n"
            f"  Academic data by subject:\n"
            + "\n".join(acad_lines)
        )

    prompt = (
        "You are an AI assistant generating a concise weekly classroom report for a teacher.\n\n"
        "Context: Controlled classroom simulation — 3 students, 3 days, 3 subjects (Math, Arabic, English). "
        "Stress data is from smartwatch heart-rate sensors and is an INDICATOR only — not a medical diagnosis. "
        "Do not make medical claims.\n\n"
        "Student data:\n"
        + "\n\n".join(summaries)
        + "\n\n"
        "Write a professional, teacher-friendly weekly report. For each student include a clear heading, then:\n"
        "- Subject-level focus and engagement patterns (mention which subject is strongest/weakest)\n"
        "- Brief stress note with appropriate caveats\n"
        "- Academic risk per subject and key contributing factors\n"
        "- 1–2 clear, practical recommendations for the teacher\n\n"
        "Keep total report under 700 words. Use a heading per student."
    )

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1400,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
