import joblib
import pandas as pd


# Load saved model
model_data = joblib.load("models/hr_stress_xgb_model.pkl")

model = model_data["model"]
FEATURE_COLS = model_data["feature_cols"]
WINDOW = model_data["window"]


def extract_hr_features(df):
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

    return df


# Example:
# لازم الملف الجديد يكون فيه عمود اسمه hr
df_new = pd.read_csv("new_hr_data.csv")

df_new = extract_hr_features(df_new)

X_new = df_new[FEATURE_COLS]

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

df_new["prediction"] = predictions
df_new["stress_probability"] = probabilities[:, 1]

print(df_new[["hr", "prediction", "stress_probability"]])