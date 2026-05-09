import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier



df = pd.read_csv("neww_data.csv")



X = df.drop("G3", axis=1)


y = df["G3"] - 1



X = pd.get_dummies(X, drop_first=True)

feature_columns = X.columns.tolist()



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)


param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}



grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1
)



grid_search.fit(X_train, y_train)



print("Best Parameters:")
print(grid_search.best_params_)



best_model = grid_search.best_estimator_



y_pred = best_model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


model_package = {
    "model": best_model,
    "feature_columns": feature_columns,
    "label_offset": 1
}

joblib.dump(model_package, "xgb_g3_model.pkl")

print("\nModel saved successfully as xgb_g3_model.pkl")