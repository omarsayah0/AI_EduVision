import pandas as pd
import joblib



model_package = joblib.load("xgb_g3_model.pkl")

model = model_package["model"]
feature_columns = model_package["feature_columns"]
label_offset = model_package["label_offset"]


# =========================
# New Input Example
# =========================
new_data = pd.DataFrame([{
    # ever thing expect the G3

    # example:
    # "school": "GP",
    # "sex": "F",
    # "age": 18,
    # "address": "U",
    # "famsize": "GT3",
    # ...
}])



new_data = pd.get_dummies(new_data, drop_first=True)

new_data = new_data.reindex(columns=feature_columns, fill_value=0)



prediction = model.predict(new_data)

prediction_original = prediction + label_offset

print("Predicted G3:", prediction_original[0])