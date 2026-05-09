import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


DATA_DIR = "data"
WINDOW = 30

all_data = []

for subject in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject)

    if not os.path.isdir(subject_path):
        continue

    file_path = os.path.join(subject_path, f"{subject}_hr_labeled.csv")

    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)

    df = df[df["label"].isin([1, 2, 3])].copy()

    df["label"] = df["label"].map({
        1: 0,
        2: 1,
        3: 0
    })

    hr = df["hr"]

    df["hr_mean"] = hr.rolling(WINDOW, min_periods=1).mean()
    df["hr_std"] = hr.rolling(WINDOW, min_periods=1).std().fillna(0)
    df["hr_min"] = hr.rolling(WINDOW, min_periods=1).min()
    df["hr_max"] = hr.rolling(WINDOW, min_periods=1).max()
    df["hr_range"] = df["hr_max"] - df["hr_min"]

    df["hr_diff"] = hr.diff().fillna(0)
    df["hr_diff_abs"] = df["hr_diff"].abs()
    df["hr_diff_std"] = df["hr_diff"].rolling(WINDOW, min_periods=1).std().fillna(0)

    df["hr_dev"] = hr - df["hr_mean"]

    all_data.append(df)


data = pd.concat(all_data, ignore_index=True)


FEATURE_COLS = [
    "hr",
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_range",
    "hr_diff",
    "hr_diff_abs",
    "hr_diff_std",
    "hr_dev",
]

X = data[FEATURE_COLS]
y = data["label"]

print(f"Class distribution:\n{y.value_counts()}\n")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


n_stress = (y_train == 1).sum()
n_nonstress = (y_train == 0).sum()

scale_pos_weight = n_nonstress / n_stress


model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)


param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "scale_pos_weight": [1, scale_pos_weight]
}


grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)


grid.fit(X_train, y_train)


print("\nBest Params:")
print(grid.best_params_)

print("\nBest CV Score (f1_macro):")
print(round(grid.best_score_, 4))


best_model = grid.best_estimator_


# =========================
# Evaluate final model
# =========================

y_pred = best_model.predict(X_test)

print("\nFinal Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["non-stress", "stress"]
))


# =========================
# Save model
# =========================

os.makedirs("models", exist_ok=True)

model_data = {
    "model": best_model,
    "feature_cols": FEATURE_COLS,
    "window": WINDOW
}

joblib.dump(model_data, "models/hr_stress_xgb_model.pkl")

print("\nModel saved successfully:")
print("models/hr_stress_xgb_model.pkl")