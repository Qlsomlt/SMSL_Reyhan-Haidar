"""
MLflow Configuration and Tracking Guide
========================================

This project uses MLflow for experiment tracking and model management.

### How to use MLflow:

1. **Install MLflow** (if not already installed):
   pip install mlflow>=2.19.0

2. **Start MLflow UI**:
   Run this command from the project root directory:
   
   mlflow ui --host 0.0.0.0 --port 5000
   
   Then open http://localhost:5000 in your browser

3. **Run the model training script**:
   python membangun_model/model.py

4. **View experiments and runs**:
   - Open MLflow UI to see all experiments
   - Compare metrics across different runs
   - Download and deploy models

### MLflow Experiment: Monster_Hunter_Sentiment_Analysis

Currently tracking:
- Model parameters (test_size, random_state, max_iter, etc.)
- Metrics (accuracy, precision, recall, f1_score)
- Trained model artifacts

### Logged Metrics:
- accuracy: Overall accuracy score
- precision: Weighted precision across classes
- recall: Weighted recall across classes
- f1_score: Weighted F1-score across classes

### Logged Parameters:
- test_size: Train/test split ratio
- random_state: Random seed for reproducibility
- max_iter: Maximum iterations for LogisticRegression
- model_type: Type of model used
- training_samples: Number of training samples
- test_samples: Number of test samples

### View MLflow Backend:
- Experiment tracking: http://localhost:5000/#/experiments
- Model registry: http://localhost:5000/#/models
"""

if __name__ == "__main__":
    import subprocess
    import sys
    
    print(__doc__)
    print("\n" + "="*60)
    print("Starting MLflow UI...")
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
