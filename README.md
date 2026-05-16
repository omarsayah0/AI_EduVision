# EduVision 

A graduation-level AI research and engineering project that integrates Computer Vision, Physiological Signal Processing, and Academic Performance Prediction into a unified real-time classroom monitoring platform.

---

## Table of Contents

1. [Description](#description)
2. [Introduction](#introduction)
3. [Project Overview](#project-overview)
   - [Camera Module](#1-camera-module)
   - [Smartwatch Module](#2-smartwatch-module)
   - [Academic Performance Module](#3-academic-performance-module)
4. [Datasets](#datasets)
5. [Project Structure](#project-structure)
6. [File Descriptions](#file-descriptions)
7. [Model Performance](#model-performance)
8. [Ground Truth Validation](#ground-truth-validation)
9. [Requirements](#requirements)
10. [Setup and Installation](#setup-and-installation)
11. [How to Use](#how-to-use)
12. [Example Outputs](#example-outputs)
13. [Resources and References](#resources-and-references)
14. [Conclusion](#conclusion)
15. [Contributors](#-contributors)
16. [License](#-license)

---

## Description

This project presents a **multi-modal AI system** designed for **real-time student monitoring** in classroom environments. The system integrates three independently trained and deployed AI modules:

- A **YOLO-based Computer Vision module** that monitors student engagement and classroom behavior through live or recorded video, achieving **77% mAP** on a custom classroom dataset.
- A **MediaPipe Face Mesh pipeline** that analyzes student focus and distraction by estimating head orientation and detecting off-task facial movements in real time.
- A **Smartwatch module** trained on the WESAD physiological dataset that classifies student stress levels from heart-rate features, achieving **91% accuracy**.
- An **Academic Performance module** trained on the UCI Student Performance dataset that predicts academic risk levels from demographic and academic attributes, achieving **89% accuracy**.
- A **Streamlit-based deployment platform** that provides live dashboards, historical session analytics, per-student profiling, and AI-generated weekly reports powered by the Mistral AI language model.
- End-to-end **ground-truth validation** performed on real classroom recordings at the university with 6 students, achieving an **overall framework accuracy of ≈82.33%** across the three pipeline stages, evaluated against a dedicated ground truth file.

The system is designed to assist educators in identifying at-risk students early, understanding behavioral and physiological engagement patterns, and making data-informed pedagogical decisions — all from a single integrated interface.

---

## Introduction

### The Problem

Traditional classroom assessment relies almost entirely on periodic examinations and subjective teacher observation. These methods are inherently delayed — by the time a student's academic risk is identified, valuable intervention time has already been lost. Moreover, examinations capture a single dimension of student performance while ignoring the rich behavioral and physiological signals that occur continuously during every lesson.

### The Motivation

Modern AI capabilities — including real-time object detection, facial landmark estimation, physiological signal processing, and gradient-boosted classification — have matured to a point where they can be meaningfully combined to produce a holistic picture of student engagement. This project was motivated by the question: *Can we build an AI system that gives teachers actionable, per-student insights across behavioral, physiological, and academic dimensions simultaneously?*

### The Solution

This system answers that question by fusing three data modalities — visual behavior, heart-rate-derived stress, and academic metrics — into a unified dashboard. The platform supports:

- **Real-time behavioral classification** of individual students in a shared classroom frame
- **Stress-level indicators** derived from smartwatch heart-rate signals, providing physiological context that visual monitoring alone cannot capture
- **Academic risk prediction** that contextualizes behavioral findings with historical performance data
- **AI-generated teacher reports** that synthesize all three data streams into actionable, subject-level student summaries

### Scope

The system was validated using real classroom videos recorded at the university, involving 6 students across multiple class sessions and subjects. Stage 1 comprises 30 historical session videos; Stage 2 comprises 1 final session video used for end-to-end ground-truth validation.

---

## Project Overview

### 1. Camera Module

The Camera module is the perceptual core of the system. It combines two complementary computer vision pipelines to classify each student's engagement state in every video frame.

#### YOLOv8 Behavior Detection

A **YOLOv8s** model was fine-tuned on a custom classroom dataset to detect two engagement-positive behavior classes:

| Class | Description |
|:---|:---|
| `reading_writing` | Student is actively reading or writing — highest engagement signal |
| `raise_hand` | Student raises hand to participate — strong engagement signal |

Training configuration:
- **Base model:** YOLOv8s (small variant for accuracy-efficiency balance)
- **Epochs:** 70
- **Image size:** 640 × 640
- **Batch size:** 16
- **Optimizer:** SGD with momentum 0.937
- **Learning rate:** 0.01
- **Augmentation:** HSV jitter, translation, scaling, horizontal flip, mosaic (1.0), mixup (0.2)

A separate **YOLOv8n** (nano) model handles person detection, sorting detected persons left-to-right to assign stable student identities (S1, S2, ..., S6) within a shared classroom frame.

#### MediaPipe Face Mesh — Head Pose Estimation

When no engagement behavior is detected for a student, the pipeline falls back to **MediaPipe Face Landmarker** for head pose analysis. The model outputs a 4×4 facial transformation matrix from which **pitch** (vertical tilt) and **yaw** (horizontal rotation) are extracted:

| Condition | Classification |
|:---|:---|
| `abs(yaw) > 25°` or `pitch > 20°` | **Distracted** |
| Within thresholds | **Attentive** |

#### Focus Score and Engagement Classification

Each student receives a per-session **focus score** (0–100%) computed as the proportion of frames where the student was classified as Attentive or Engaged. The engagement status is derived from reading and hand-raise ratios:

| Threshold | Engagement Status |
|:---|:---|
| Reading ratio > 20% or raise-hand ratio > 10% | **Good** |
| Focus score ≥ 60% | **Acceptable** |
| Below thresholds | **Low** |

#### Live Inference Pipeline

The standalone `camera/main.py` script supports real-time webcam monitoring with:
- ByteTrack multi-object tracking for persistent person IDs
- Seat-based stable ID assignment to handle track ID re-assignment
- IoU-based deduplication of overlapping behavior detections
- JSON output (`data.json`) written every inference cycle for dashboard consumption

---

### 2. Smartwatch Module

The Smartwatch module classifies student physiological stress from heart-rate data. The design deliberately prioritizes **practical real-world deployability** over laboratory completeness.

#### Dataset: WESAD

The model was trained on the **WESAD (Wearable Stress and Affect Detection)** dataset, which provides multi-signal physiological recordings from an **Empatica E4** wrist-worn device for 15 subjects under controlled stress and baseline conditions. Available signals include:

| Signal | Description |
|:---|:---|
| Heart Rate (HR) | Blood volume pulse-derived BPM at 1 Hz |
| BVP | Blood Volume Pulse at 64 Hz |
| EDA | Electrodermal Activity (skin conductance) at 4 Hz |
| ACC | 3-axis Accelerometer at 32 Hz |
| TEMP | Skin Temperature at 4 Hz |

#### Design Decision: Heart Rate Only

Despite the rich multi-signal nature of WESAD, this system was intentionally designed to use **Heart Rate only**. The rationale is straightforward: BVP, EDA, and TEMP require medical-grade wearable sensors (such as the E4, costing several hundred dollars), making them impractical for real school deployments. Heart rate, by contrast, is measured by virtually every consumer-grade smartwatch (Apple Watch, Samsung Galaxy Watch, Garmin, Fitbit, etc.).

This design choice means that the trained model can be used with any BPM-capable wearable, without requiring specialized hardware.

#### Preprocessing Pipeline (`preprocess.py`)

The preprocessing script aligns WESAD labels (sampled at 700 Hz) with HR values (sampled at 1 Hz):

1. Load `HR.csv` from each subject's Empatica E4 recording (format: start timestamp, sample rate, values)
2. Load the subject `.pkl` file containing label arrays
3. Map each HR sample to its nearest label index: `label_index = round(i × 700 / hr_hz)`
4. Filter to labels 1 (baseline → non-stress), 2 (stress → stress), 3 (amusement → non-stress)
5. Produce a `{subject}_hr_labeled.csv` per subject ready for training

#### Feature Engineering

A 30-sample rolling window is applied to extract temporal HR features:

| Feature | Description |
|:---|:---|
| `hr` | Raw heart rate (BPM) |
| `hr_mean` | Rolling mean over 30 samples |
| `hr_std` | Rolling standard deviation |
| `hr_min` | Rolling minimum |
| `hr_max` | Rolling maximum |
| `hr_range` | `hr_max - hr_min` |
| `hr_diff` | First-order difference |
| `hr_diff_abs` | Absolute first-order difference |
| `hr_diff_std` | Rolling std of first-order difference |
| `hr_dev` | Deviation from rolling mean |

#### Model Training (`train.py`)

- **Algorithm:** XGBoost binary classifier (`binary:logistic`)
- **Hyperparameter search:** GridSearchCV with 5-fold cross-validation, scoring `f1_macro`
- **Class imbalance handling:** `scale_pos_weight` parameter tuned in grid search
- **Grid parameters:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `scale_pos_weight`
- **Output:** Binary label — `0 = non-stress`, `1 = stress`
- **Saved artifact:** `hr_stress_xgb_model.pkl` (model + feature columns + window size)

---

### 3. Academic Performance Module

The Academic Performance module predicts a student's academic risk level from historical performance indicators.

#### Dataset: UCI Student Performance

The model was trained on the **UCI Machine Learning Repository Student Performance Dataset**, which records Portuguese secondary-school student attributes including demographic information, family background, study habits, and exam scores. The prediction target is `G3` — the final period grade — which is discretized into three ordinal risk classes:

| Level | Label | Interpretation |
|:---:|:---|:---|
| 1 | Good | Low Academic Risk |
| 2 | Needs Monitoring | Medium Academic Risk |
| 3 | Needs Support | High Academic Risk |

#### Model Training (`prefor_module/train.py`)

- **Algorithm:** XGBoost multi-class classifier (`mlogloss` evaluation metric)
- **Preprocessing:** One-hot encoding of categorical features via `pd.get_dummies`
- **Hyperparameter search:** GridSearchCV with 3-fold cross-validation, scoring `accuracy`
- **Grid parameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- **Feature alignment:** `reindex` ensures inference-time feature columns always match training columns
- **Saved artifact:** `xgb_g3_model.pkl` (model + feature column list + label offset)

#### Input Features (deployment)

In the deployment context, the academic module ingests a structured CSV with the following columns per student per subject:

| Column | Type | Description |
|:---|:---|:---|
| `student_id` | String | Student identifier (S1–S6) |
| `subject` | String | Subject name (math, arabic, english) |
| `homework_commitment` | Binary | 1 = homework submitted, 0 = not |
| `month1_exam` | Integer | First monthly exam score (0–100) |
| `month2_exam` | Integer | Second monthly exam score (0–100) |
| `absence` | Integer | Number of absences |
| `academic_level` | Integer | Ground-truth label for validation (1/2/3) |

---

## Datasets

> **Note:** All datasets listed below are **not included** in this repository due to their large file sizes. Users must download them from their official sources before running training scripts.

### Camera Dataset — SCB05

The YOLOv8 behavior detection model was trained on the **SCB05** dataset — a classroom behavior dataset annotated in YOLO format covering two engagement-relevant action classes: `reading_writing` and `raise_hand`.

| Property | Details |
|:---|:---|
| **Name** | SCB05 (Student Classroom Behavior Dataset) |
| **Format** | YOLO annotation format |
| **Classes** | `reading_writing`, `raise_hand` |
| **Config** | `classroom.yaml` — specifies class names, train/val/test split paths |
| **Usage** | Fine-tuning YOLOv8s from the `yolov8s.pt` pretrained checkpoint |

**Download:** [SCB05 Dataset — Kaggle](https://www.kaggle.com/datasets/shreyasudaya/scb-05-dataset)

### Smartwatch Dataset — WESAD

| Property | Details |
|:---|:---|
| **Full Name** | Wearable Stress and Affect Detection (WESAD) |
| **Source** | UCI Machine Learning Repository / Physionet |
| **Subjects** | 15 participants |
| **Device** | Empatica E4 wrist-worn biosensor |
| **Signals Used** | HR (Heart Rate) at 1 Hz |
| **Available Signals** | HR, BVP (64 Hz), EDA (4 Hz), ACC (32 Hz), TEMP (4 Hz) |
| **Labels** | 0=Transient, 1=Baseline (non-stress), 2=Stress, 3=Amusement (non-stress), 4=Meditation |
| **Label Rate** | 700 Hz (per-subject `.pkl` file) |
| **Format** | Per-subject folder with `HR.csv` and `{subject}.pkl` |

**Download:** [WESAD Dataset — UCI Repository](https://archive.ics.uci.edu/dataset/465/wesad)

After downloading, place each subject folder (e.g., `S2/`, `S3/`, ...) under `training/watch_module/data/` and run `preprocess.py` before `train.py`.

### Academic Dataset — UCI Student Performance

| Property | Details |
|:---|:---|
| **Full Name** | Student Performance Dataset |
| **Source** | UCI Machine Learning Repository |
| **Instances** | 649 students (Math: 395, Portuguese: 649) |
| **Features** | 33 attributes (demographic, family, academic) |
| **Target** | G3 — Final period grade (0–20), discretized into risk levels |
| **Format** | CSV (`student-mat.csv` and/or `student-por.csv`) |

**Download:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

After downloading, rename or preprocess the CSV as `neww_data.csv` (with a discretized `G3` column encoding the three risk levels) and place it in `training/prefor_module/` before running `train.py`.

---

## Project Structure

```
graduate_project/
│
├── README.md                            ← This file — project-level documentation
│
├── training/                            ← Model training pipelines (all three modules)
│   │
│   ├── camera_module/                   ← YOLO behavior detection training
│   │   ├── train.py                     ← YOLOv8s fine-tuning script
│   │   ├── data.json                    ← Sample inference output (status snapshot)
│   │   ├── best.pt                      ← Trained YOLOv8 behavior model checkpoint
│   │   └── yolov8n.pt                   ← YOLOv8 nano base checkpoint (person detector)
│   │
│   ├── watch_module/                    ← Smartwatch stress classification training
│   │   ├── preprocess.py                ← WESAD HR alignment and label extraction
│   │   ├── train.py                     ← XGBoost training with GridSearchCV
│   │   ├── main.py                      ← Inference script for new HR data
│   │   └── hr_stress_xgb_model.pkl      ← Trained stress classification model
│   │
│   └── prefor_module/                   ← Academic performance prediction training
│       ├── train.py                     ← XGBoost multi-class training with GridSearchCV
│       ├── main.py                      ← Inference script for new student records
│       └── xgb_g3_model.pkl             ← Trained academic risk prediction model
│
└── deployment/                          ← Full integrated deployment application
    │
    ├── app.py                           ← Main Streamlit application entry point
    ├── utils.py                         ← Core logic: model loading, video processing,
    │                                       data fusion, Mistral API integration
    ├── requirements.txt                 ← Python package dependencies
    ├── README.md                        ← Deployment-specific setup and demo guide
    ├── app.zip                          ← Packaged application archive
    │
    ├── camera/                          ← Camera module assets and scripts
    │   ├── best.pt                      ← Deployed YOLO behavior detection model
    │   ├── yolov8n.pt                   ← Deployed YOLO person detection model
    │   ├── face_landmarker.task         ← MediaPipe Face Landmarker model file
    │   ├── main.py                      ← Standalone real-time webcam inference
    │   ├── dashboard.py                 ← Live Streamlit dashboard (standalone mode)
    │   ├── data.json                    ← Live inference output consumed by dashboard
    │   └── test_model.py                ← Multi-mode model evaluation utility
    │
    ├── watch/                           ← Smartwatch module (deployment copy)
    │   ├── hr_stress_xgb_model.pkl      ← Deployed stress classification model
    │   └── main.py                      ← Stress inference for new HR readings
    │
    └── perfor/                          ← Academic module (deployment copy)
        ├── xgb_g3_model.pkl             ← Deployed academic risk prediction model
        └── main.py                      ← Academic inference for new student data
```

---

## File Descriptions

### Training — Camera Module

| File | Description |
|:---|:---|
| `training/camera_module/train.py` | Fine-tunes YOLOv8s on the classroom behavior dataset. Configures training for 70 epochs at 640px with SGD, data augmentation (mosaic, mixup, HSV jitter), and GPU device. Requires `classroom.yaml` and the raw dataset. |
| `training/camera_module/best.pt` | The best YOLO model checkpoint produced by training — selected by highest validation mAP across all epochs. This is the model deployed in production. |
| `training/camera_module/yolov8n.pt` | Pre-trained YOLOv8 nano weights used as the person detection backbone. Not fine-tuned; used directly for generic person localization. |
| `training/camera_module/data.json` | A JSON snapshot of engagement inference output demonstrating the runtime data schema: `student_id → {status, behavior, last_seen, lesson_duration}`. |

### Training — Watch Module

| File | Description |
|:---|:---|
| `training/watch_module/preprocess.py` | Loads Empatica E4 `HR.csv` and WESAD label `.pkl` files, aligns label timestamps to HR sample timestamps (700 Hz → 1 Hz mapping), and writes per-subject `_hr_labeled.csv` files. Must be run before `train.py`. |
| `training/watch_module/train.py` | Loads preprocessed HR labeled CSVs from all subjects, applies 30-sample rolling-window feature extraction (10 features), trains an XGBoost classifier with 5-fold GridSearchCV, prints classification report, and saves the final model as `hr_stress_xgb_model.pkl`. |
| `training/watch_module/main.py` | Standalone inference script demonstrating how to load the saved stress model and apply it to a new CSV containing a `hr` column. Outputs per-sample predictions and stress probabilities. |
| `training/watch_module/hr_stress_xgb_model.pkl` | Serialized model artifact containing the trained XGBoost classifier, the list of 10 feature column names, and the rolling window size. Required for all downstream inference. |

### Training — Academic Performance Module

| File | Description |
|:---|:---|
| `training/prefor_module/train.py` | Loads the academic dataset (`neww_data.csv`), drops the `G3` target column to form feature matrix `X`, applies one-hot encoding, trains an XGBoost multi-class classifier with 3-fold GridSearchCV, prints accuracy and classification report, and saves `xgb_g3_model.pkl`. |
| `training/prefor_module/main.py` | Inference script for new student records. Loads the saved model package, aligns input features to training columns using `reindex`, runs prediction, and prints the predicted `G3` grade class (with label offset restored). |
| `training/prefor_module/xgb_g3_model.pkl` | Serialized model package containing the trained XGBoost classifier, the ordered list of training feature column names, and the label offset value (1). |

### Deployment — Core Application

| File | Description |
|:---|:---|
| `deployment/app.py` | Main Streamlit application. Implements the two-stage data upload flow, session state management, data validation, camera result processing, data fusion orchestration, and the three-tab dashboard UI (Live Classroom, Student Profile, Weekly AI Report). |
| `deployment/utils.py` | Core utility module. Contains: `load_camera_models()` (lazy model loading with caching), `process_video()` (frame-by-frame YOLO + MediaPipe inference on session videos), `_get_head_pose()` and `_attention_from_pose()` (head orientation classification), `parse_session_filename()` (robust filename metadata extraction), `read_watch_csv()` and `read_academic_csv()` (CSV ingestion with column validation), `fuse_data()` (camera-watch merge with session_id and day+subject fallback), `get_interpreted_status()` (multi-signal status label generation), and `generate_mistral_report()` (Mistral AI API call for teacher report synthesis). |
| `deployment/requirements.txt` | Pinned Python dependency list for the deployment environment. |
| `deployment/README.md` | Deployment-specific documentation covering installation steps, demo flow, expected input file formats, and project structure. |
| `deployment/app.zip` | Compressed archive of the deployment application for distribution. |

### Deployment — Camera Module

| File | Description |
|:---|:---|
| `deployment/camera/main.py` | Standalone real-time camera inference script. Opens a webcam feed, applies ByteTrack tracking, assigns students to stable seat IDs, runs YOLO behavior + person detection and MediaPipe head pose every 3 frames, draws color-coded overlays on the live frame, and writes `data.json` for dashboard consumption. Operates independently of the Streamlit app. |
| `deployment/camera/dashboard.py` | Standalone Streamlit dashboard that reads `data.json` (written by `main.py`) and displays real-time student status with glass-morphism dark UI, status distribution pie chart, student detail table, and engagement statistics. Designed to run concurrently with `main.py`. |
| `deployment/camera/best.pt` | Deployed copy of the trained YOLO behavior detection model. |
| `deployment/camera/yolov8n.pt` | Deployed copy of the YOLOv8 nano person detection model. |
| `deployment/camera/face_landmarker.task` | MediaPipe Face Landmarker model binary (float16). Used by both the standalone pipeline and `utils.py` for head pose estimation. If absent, `main.py` auto-downloads it from the MediaPipe CDN. |
| `deployment/camera/data.json` | Shared state file written by `main.py` and read by `dashboard.py`. Contains per-student status, detected behavior, last-seen timestamp, and lesson duration. |
| `deployment/camera/test_model.py` | Multi-mode evaluation utility for `best.pt`. Supports four modes: `info` (print model metadata), `val` (validate on a YOLO-format labeled dataset and print mAP/precision/recall), `images` (run inference on an image folder and generate a confidence report), `video` (run inference on a video file with visual overlay). Used for model quality assessment. |

### Deployment — Watch Module

| File | Description |
|:---|:---|
| `deployment/watch/main.py` | Deployment inference script for the stress classification model. Loads the saved XGBoost model, applies rolling-window feature extraction to a new HR CSV, and outputs per-row stress predictions and probabilities. |
| `deployment/watch/hr_stress_xgb_model.pkl` | Deployed copy of the trained stress classification model artifact. |

### Deployment — Academic Performance Module

| File | Description |
|:---|:---|
| `deployment/perfor/main.py` | Deployment inference script for the academic performance model. Loads the model package, one-hot encodes a new student record, aligns columns to the training schema, runs prediction, and outputs the predicted academic level. |
| `deployment/perfor/xgb_g3_model.pkl` | Deployed copy of the trained academic risk prediction model artifact. |

---

## Model Performance

### Summary Table

| Module | Metric | Result |
|:---|:---:|:---:|
| YOLO Behavior Detection (Camera) | Binary Accuracy | **67.60%** |
| Smartwatch Stress Classification | Accuracy | **91%** |
| Academic Performance Prediction | Accuracy | **89%** |
| End-to-End Ground Truth Validation | Overall Accuracy | **≈82.33%** |

### Individual Model Performance

The system consists of three main models, each handling a different part of the pipeline. Their performance is summarized below:

| Model   | Type                  | Task                                | Accuracy | mAP@50 | mAP@50–95 | Precision | Recall | F1-score |
|---------|-----------------------|-------------------------------------|----------|--------|-----------|-----------|--------|----------|
| Model 1 | Detection (YOLO)      | Classroom behavior (camera input)   | 67.60%   | —      | —         | 62.91%    | 97.94% | 76.61%   |
| Model 2 | Classification        | Academic performance analysis       | 88.52%   | —      | —         | 0.88      | 0.89   | 0.88     |
| Model 3 | Classification        | Stress detection (smartwatch data)  | 91.0%    | —      | —         | 0.91      | 0.91   | 0.91     |

---

## Ground Truth Validation

### Overview

The system was validated end-to-end using real classroom videos recorded at the university, with 6 students participating. All three AI modules were evaluated jointly to verify that their combined predictions are consistent with pre-defined expected outcomes per student and per session. The complete per-student, per-session ground truth labels used for evaluation are provided in [`ground_truth_correct_reformatted.xlsx`](ground_truth_correct_reformatted.xlsx).

### Validation Methodology

1. **Define ground truth** — Expected behavioral, physiological, and academic outcomes are documented before running the system.
2. **Run the full pipeline** — Upload all videos and CSVs through the Streamlit app to complete both Stage 1 and Stage 2 processing.
3. **Compare outputs** — The Student Profile Dashboard and Live Classroom Dashboard outputs are compared against the ground truth file.
4. **Generate and verify AI report** — The Mistral AI weekly report is generated and its per-student subject-level characterizations are verified against expected phrasing.
5. **Result** — Across real classroom test runs, the system's combined predictions were evaluated against the pre-defined ground truth, yielding an overall framework accuracy of ≈82.33% across the three pipeline stages.

---

## Requirements

### Runtime Environment

| Requirement | Specification |
|:---|:---|
| **Python** | 3.10 or higher (3.12 recommended) |
| **Operating System** | Windows (camera models use compiled `.pyd` binaries in MediaPipe) |
| **API Key** | Mistral AI API key (required only for the Weekly Report feature) |

### Python Dependencies (Deployment)

| Package | Version | Purpose |
|:---|:---:|:---|
| `streamlit` | 1.57.0 | Web dashboard framework |
| `pandas` | 3.0.2 | Data manipulation and CSV processing |
| `numpy` | 2.4.4 | Numerical computation |
| `opencv-python` | 4.13.0.92 | Video capture and image processing |
| `ultralytics` | 8.4.46 | YOLOv8 model loading and inference |
| `mediapipe` | 0.10.35 | Face Landmarker for head pose estimation |
| `requests` | 2.33.1 | Mistral AI REST API calls |
| `plotly` | 6.7.0 | Interactive charts in the dashboard |
| `torch` | 2.11.0 | PyTorch backend for YOLO/MediaPipe |
| `torchvision` | 0.26.0 | PyTorch vision utilities |

### Additional Dependencies (Training Only)

These packages are required for running the training scripts but are not included in the deployment `requirements.txt`:

| Package | Purpose |
|:---|:---|
| `scikit-learn` | GridSearchCV, train-test split, classification metrics |
| `xgboost` | XGBoost classifier for stress and academic models |
| `joblib` | Model serialization and loading |
| `pickle` | Loading WESAD `.pkl` subject files |

---

## Setup and Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/omarsayah0/AI_EduVision.git
cd AI_EduVision
```

### Step 2 — Navigate to the Deployment Folder

```bash
cd deployment
```

### Step 3 — Create a Virtual Environment

```bash
py -3.12 -m venv venv
```

### Step 4 — Activate the Virtual Environment

```bash
# Windows
venv\Scripts\activate
```

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

> `torch` and `torchvision` are large packages (~2 GB). The first install may take several minutes depending on your internet connection.

### Step 6 — Set the Mistral API Key (optional but recommended)

```bash
set MISTRAL_API_KEY=your_key_here
```

Alternatively, paste the key directly in the sidebar when the Streamlit app is running.

---

### Running the Training Pipelines

#### Camera Module Training

Requires the classroom dataset in YOLO format with a `classroom.yaml` config file.

```bash
cd training/camera_module
python train.py
```

The trained model will be saved as `runs/detect/train/weights/best.pt`.

#### Smartwatch Module Training

Requires the WESAD dataset placed under `training/watch_module/data/` (one folder per subject containing `HR.csv` and `{subject}.pkl`).

```bash
cd training/watch_module

# Step 1: Preprocess WESAD data
python preprocess.py

# Step 2: Train the stress classifier
python train.py
```

The model will be saved as `models/hr_stress_xgb_model.pkl`.

#### Academic Performance Module Training

Requires the preprocessed UCI dataset saved as `neww_data.csv` in `training/prefor_module/`.

```bash
cd training/prefor_module
python train.py
```

The model will be saved as `xgb_g3_model.pkl` in the same directory.

---

### Running the Deployment Application

#### Integrated Streamlit App (Recommended)

```bash
cd deployment
streamlit run app.py
```

The app opens in your default browser at `http://localhost:8501`.

#### Standalone Real-Time Camera Mode

Run both scripts concurrently — `main.py` writes inference results and `dashboard.py` displays them live:

```bash
# Terminal 1 — run camera inference
cd deployment/camera
python main.py

# Terminal 2 — run live dashboard
cd deployment/camera
streamlit run dashboard.py
```

#### Model Evaluation

```bash
cd deployment/camera

# Print model information
python test_model.py --mode info

# Validate on a labeled YOLO dataset
python test_model.py --mode val --data path/to/data.yaml

# Run inference on an image folder
python test_model.py --mode images --source path/to/images/

# Run inference on a video file
python test_model.py --mode video --source path/to/video.mp4
```

---

## How to Use

### Demo Flow (Integrated Streamlit App)

#### Stage 1 — Process Historical Sessions

1. Open the app with `streamlit run app.py`
2. In the sidebar, enter student display names for S1–S6 (optional)
3. Upload the **30 historical session videos**
4. Upload the **historical watch CSV**
5. Click **Process Historical Sessions** — the system processes each video with YOLO + MediaPipe inference and stores per-student, per-session results

#### Stage 2 — Process Final Session and Academic Data

6. Upload the **final session video**
7. Upload the **final watch CSV**
8. Upload the **academic results CSV**
9. Click **Process Final Session & Academic Data**

#### Explore the Dashboard

- **Live Classroom Dashboard tab** — View the final session video replay alongside per-student focus score, engagement badge, and stress indicator
- **Student Profile Dashboard tab** — Select any student to view behavioral analytics by subject, stress patterns across all sessions, academic risk per subject, and a combined behavioral-academic summary
- **Weekly AI Report tab** — Click **Generate Weekly Report** to invoke the Mistral AI API and produce a structured teacher-friendly narrative covering focus patterns, stress indicators, academic risk, and recommendations per student

### Video Format Requirements

- Videos must be named using the pattern `dayX_subject.mp4` (e.g., `day1_math.mp4`)
- Each video must show all 6 students in a single combined frame, arranged left to right
- Supported formats: `.mp4`, `.avi`, `.mov`

### CSV Format Requirements

#### Watch CSV

```
student_id,day,subject,session_id,stress_level,bpm
S1,day1,math,day1_math,normal,86
S2,day1,math,day1_math,normal,82
S3,day1,math,day1_math,normal,84
```

#### Academic CSV

```
student_id,subject,homework_commitment,month1_exam,month2_exam,absence,academic_level
S1,math,1,61,64,2,2
S1,arabic,0,54,57,3,3
S1,english,1,90,94,1,1
```

`academic_level`: 1 = Good / Low Risk, 2 = Needs Monitoring / Medium Risk, 3 = Needs Support / High Risk

### Understanding the Output

| Output | Source | Interpretation |
|:---|:---:|:---|
| Focus Score (%) | Camera | Proportion of frames where student was Attentive or Engaged |
| Engagement Status | Camera | Good / Acceptable / Low — based on behavior ratios and focus score |
| Stress Level | Smartwatch | Normal / High — derived from BPM-based XGBoost prediction |
| Interpreted Status | Fusion | Teacher-friendly label combining focus, engagement, and stress |
| Academic Risk | Academic Model | Low / Medium / High — XGBoost prediction from exam and attendance data |
| Weekly Report | Mistral AI | Subject-level narrative summary with practical teacher recommendations |

---

## Example Outputs

### Main page
Stage 1:

<img width="1919" height="946" alt="image" src="https://github.com/user-attachments/assets/e7aecc44-330d-4e36-b29b-d003a1b70422" />

Stage 2: 
<img width="1919" height="949" alt="image" src="https://github.com/user-attachments/assets/b0107cdc-c57e-41c0-8ba0-fe689883bd0d" />

---

### Live Classroom Dashboard

<img width="1645" height="902" alt="image" src="https://github.com/user-attachments/assets/1f088dff-61d1-43fb-9958-4b289ebc94b1" />

---

### Student Profile — Behavioral Analytics Table

<img width="1919" height="944" alt="image" src="https://github.com/user-attachments/assets/276cce11-3c0c-460c-b2c3-eaf5f7ec694c" />

---

### Student Profile — Combined Subject Summary

<img width="1919" height="936" alt="image" src="https://github.com/user-attachments/assets/94335fbf-f071-4e19-9966-2a2b4a302cab" />

---

### Weekly AI Report — Mistral Generated Summary

Data sent to LLM:

<img width="1919" height="949" alt="image" src="https://github.com/user-attachments/assets/239608c8-c60b-4e5e-860a-efa2e0d8982b" />

The generated report:

<img width="1080" height="952" alt="image" src="https://github.com/user-attachments/assets/7e4220d2-3ef9-42b4-9966-a17ab4079e59" />

---

## Resources and References

### Frameworks and Libraries

| Resource | URL |
|:---|:---|
| YOLOv8 / Ultralytics | https://github.com/ultralytics/ultralytics |
| MediaPipe | https://developers.google.com/mediapipe |
| Streamlit | https://streamlit.io |
| OpenCV | https://opencv.org |
| PyTorch | https://pytorch.org |
| XGBoost | https://xgboost.readthedocs.io |
| scikit-learn | https://scikit-learn.org |
| Plotly | https://plotly.com/python |
| Mistral AI API | https://docs.mistral.ai |

### Datasets

| Dataset | Citation / Source |
|:---|:---|
| SCB-Dataset | Yang, Y., Wang, Y., & Wang, Y. (2023). *SCB-Dataset: A Dataset for Detecting Student and Teacher Classroom Behavior*. arXiv preprint, arXiv:2304.02488. |
| WESAD (Stress Detection) | Schmidt, P., et al. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*. Proceedings of the 2018 International Conference on Multimodal Interaction (ICMI ’18), ACM, 400–408. |
| Student Academic Performance | Alamri, L. H., et al. (2020). *Predicting Student Academic Performance Using Support Vector Machine and Random Forest*. Proceedings of the 3rd International Conference on Education Technology Management (ICETM 2020), 1–8. |
### Academic References

- Yang, Y., Wang, Y., & Wang, Y. (2023). *SCB-Dataset: A Dataset for Detecting Student and Teacher Classroom Behavior*. arXiv preprint, arXiv:2304.02488.

- Schmidt, P., et al. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*. Proceedings of the 2018 International Conference on Multimodal Interaction (ICMI ’18), ACM, 400–408.

- Alamri, L. H., et al. (2020). *Predicting Student Academic Performance Using Support Vector Machine and Random Forest*. Proceedings of the 3rd International Conference on Education Technology Management (ICETM 2020), 1–8.

---

## Conclusion

This project demonstrates that a multi-modal AI approach — combining computer vision, physiological signal analysis, and academic performance prediction — can provide a substantially richer and more actionable view of student engagement than any single modality alone.

The three modules are complementary by design: the Camera module captures visible behavioral engagement, the Smartwatch module reveals physiological stress that has no visual manifestation, and the Academic Performance module provides the longitudinal outcome context that gives both other signals their pedagogical meaning. When fused together and presented through an AI-generated narrative, these signals equip teachers with the kind of per-student, per-subject insight that would otherwise require hours of manual observation and record review.

The ground-truth validation — conducted using real classroom recordings at the university with 6 students, achieving an overall framework accuracy of ≈82.33% across the three pipeline stages — confirms that the system's combined predictions are meaningful and aligned with real-world behavioral patterns.

Looking ahead, the architecture is designed for incremental extension:

- **Larger student cohorts** can be accommodated by updating the student ID mapping and extending the video layout assumptions
- **Additional sensor modalities** (e.g., microphone-based speech activity, EDA wristbands) can be fused through the same `fuse_data` infrastructure
- **Longitudinal dashboards** tracking engagement trends across full academic terms are a natural extension of the existing historical analytics
- **School-scale deployment** would require integration with existing student information systems and appropriate privacy-preserving data handling

The ultimate goal of this system is not to automate teacher judgment, but to enhance it — giving educators timely, evidence-based signals so that interventions happen when they can still make a difference, rather than after the fact.

---

## 👥 Contributors

- **Omar Al ethamat** – *AI Engineer*

Feel free to open issues or pull requests to contribute.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
