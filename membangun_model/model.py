from pathlib import Path

import pandas as pd

import mlflow
import mlflow.sklearn
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

# Load data
CSV_PATH = Path(__file__).resolve().parent.parent / "data_clean.csv"

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    vectorizer,
) = prepare_data(CSV_PATH)

mlflow.sklearn.autolog()


# Set experiment
mlflow.set_experiment("Logistic_Regression_Experiment")

with mlflow.start_run():

    # Parameters
    random_state = 42
    max_iter = 1000

    # Train model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log parameters
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_iter", max_iter)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logistic_regression_model"
    )

    # Print results
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

    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")