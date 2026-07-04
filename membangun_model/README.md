# Model Training with MLflow

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Model Training
```bash
python membangun_model/model.py
```

### 3. View Results in MLflow UI
```bash
python run_mlflow_ui.py
```
Then open http://localhost:5000 in your browser

## What's Tracked

The model training automatically logs:
- **Experiment**: `Monster_Hunter_Sentiment_Analysis`
- **Parameters**: Model hyperparameters and data split ratios
- **Metrics**: accuracy, precision, recall, f1_score
- **Artifacts**: The trained model (Scikit-learn format)

## MLflow Features Used

1. **Experiment Tracking**: Automatically organized under "Monster_Hunter_Sentiment_Analysis"
2. **Run Management**: Each training run is automatically logged
3. **Metrics Logging**: Model performance metrics are tracked and visualized
4. **Model Logging**: Trained model is stored as an artifact for deployment

## Next Steps

- Compare multiple model runs in the MLflow UI
- Deploy models using MLflow Model Registry
- Track additional experiments with different algorithms
