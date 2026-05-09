import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR = "data"
LABEL_HZ = 700  # WESAD label sampling rate

for subject in sorted(os.listdir(DATA_DIR)):
    subject_path = os.path.join(DATA_DIR, subject)
    if not os.path.isdir(subject_path):
        continue

    hr_path = os.path.join(subject_path, "HR.csv")
    pkl_path = os.path.join(subject_path, f"{subject}.pkl")
    out_path = os.path.join(subject_path, f"{subject}_hr_labeled.csv")

    if not os.path.exists(hr_path) or not os.path.exists(pkl_path):
        print(f"[SKIP] {subject}: missing HR.csv or pkl")
        continue

    # Load HR.csv (Empatica E4 format: row0=start_time, row1=sample_rate, rest=values)
    hr_raw = pd.read_csv(hr_path, header=None).squeeze()
    hr_start = float(hr_raw.iloc[0])
    hr_hz = float(hr_raw.iloc[1])
    hr_values = hr_raw.iloc[2:].reset_index(drop=True).astype(float)

    n_hr = len(hr_values)
    hr_times = hr_start + np.arange(n_hr) / hr_hz  # Unix timestamps for each HR sample

    # Load pkl labels (sampled at LABEL_HZ)
    with open(pkl_path, "rb") as f:
        wesad = pickle.load(f, encoding="latin1")

    labels = wesad["label"]  # shape: (N,) at 700 Hz
    n_labels = len(labels)
    # Labels share the same start time as the HR file (both from E4 / study start)
    label_times = hr_start + np.arange(n_labels) / LABEL_HZ

    # For each HR sample, find nearest label index and assign that label
    # HR is 1 Hz so each sample maps to label_index = round(i * LABEL_HZ / hr_hz)
    hr_label_indices = np.round(np.arange(n_hr) * LABEL_HZ / hr_hz).astype(int)
    hr_label_indices = np.clip(hr_label_indices, 0, n_labels - 1)
    hr_labels = labels[hr_label_indices]

    df = pd.DataFrame({
        "timestamp": hr_times,
        "hr": hr_values.values,
        "label": hr_labels,
    })

    df.to_csv(out_path, index=False)
    counts = pd.Series(hr_labels).value_counts().sort_index().to_dict()
    print(f"[OK] {subject}: {n_hr} samples, label counts: {counts} -> {out_path}")

print("\nDone. Run train.py now.")
