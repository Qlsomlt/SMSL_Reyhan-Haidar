import json
import os
from pathlib import Path

import dagshub
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from preprocessing_data.preprocessing import prepare_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================
# PATH SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "logistic_regression_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "predictions.csv"

CSV_PATH = BASE_DIR / "data_clean.csv"

mlflow.set_tracking_uri(
    "https://dagshub.com/Qlsomlt/SMSL_Reyhan-Haidar.mlflow"
)

print("Tracking URI:", mlflow.get_tracking_uri())

# Set experiment AFTER tracking URI
mlflow.set_experiment("Logistic_Regression_Experiment")

# Optional client (safe AFTER setup)
client = MlflowClient()
print(client.search_experiments())

# =========================
# DATA PREPARATION
# =========================
(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    vectorizer,
) = prepare_data(CSV_PATH)

# =========================
# TRAINING
# =========================
with mlflow.start_run():

    try:
        random_state = 42
        max_iter = 1000

        print("\nTraining Logistic Regression model...")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train unique: {pd.Series(y_train).unique()}")

        model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs"
        )

        model.fit(X_train, y_train)

        # =========================
        # PREDICTION
        # =========================
        y_pred = model.predict(X_test)

        # =========================
        # METRICS
        # =========================
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # =========================
        # LOG PARAMETERS
        # =========================
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_iter", max_iter)

        # =========================
        # LOG METRICS
        # =========================
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # =========================
        # SAVE LOCAL ARTIFACTS
        # =========================
        joblib.dump(
            {
                "model": model,
                "vectorizer": vectorizer,
                "classes": model.classes_.tolist(),
            },
            MODEL_PATH,
        )

        METRICS_PATH.write_text(
            json.dumps(
                {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        pd.DataFrame({
            "actual": y_test,
            "predicted": y_pred
        }).to_csv(PREDICTIONS_PATH, index=False)

        # =========================
        # LOG MODEL TO MLFLOW
        # =========================
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="logistic_regression_model"
        )

        # =========================
        # OUTPUT
        # =========================
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("=" * 60)

        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved metrics: {METRICS_PATH}")
        print(f"Saved predictions: {PREDICTIONS_PATH}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise