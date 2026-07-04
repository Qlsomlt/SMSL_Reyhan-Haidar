from pathlib import Path

import mlflow
import mlflow.sklearn

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Logistic_Regression_Tuning")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocessing_data import prepare_data




# =====================================
# Load data
# =====================================
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

# =====================================
# MLflow Experiment
# =====================================


# =====================================
# Hyperparameter Grid
# =====================================
param_grid = {
    "C": [0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}

# =====================================
# Start MLflow Run
# =====================================
with mlflow.start_run():

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    print("Running GridSearchCV...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters")
    print(grid_search.best_params_)

    print(f"Best CV Accuracy : {grid_search.best_score_:.4f}")

    # =====================================
    # Test Evaluation
    # =====================================
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # =====================================
    # MLflow Logging
    # =====================================
    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_logistic_regression_model",
    )

    print("\n" + "=" * 60)
    print("GRID SEARCH RESULT")
    print("=" * 60)

    print(f"Best Parameters : {grid_search.best_params_}")
    print(f"Best CV Score   : {grid_search.best_score_:.4f}")

    print("\nTest Performance")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("=" * 60)

    print(f"MLflow Run ID : {mlflow.active_run().info.run_id}")