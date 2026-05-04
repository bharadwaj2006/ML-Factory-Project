# FactoryGuard AI - IoT Predictive Maintenance System

Production-ready ML system for predicting machine failures using IoT sensor data.

## 🚀 Quick Start

```bash
# Clone & Setup
git clone <repo>
cd factoryguard-ai

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# (Future steps will include training and API commands)
```

## 📁 Project Structure

```
factoryguard-ai/
├── data/                 # Raw & processed datasets
├── src/                  # ML Pipeline
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── api/                  # Flask API
│   └── app.py
├── models/               # Trained models
├── requirements.txt      # Dependencies
└── README.md
```

## 📊 Features (Completed & Planned)

✅ **STEP 1: Project Setup** - Done!

⏳ **STEP 2:** Feature Engineering Pipeline
⏳ **STEP 3:** XGBoost Model Training  
⏳ **STEP 4:** Hyperparameter Tuning (Optuna)
⏳ **STEP 5:** SHAP Explainability
⏳ **STEP 6:** Flask Prediction API
⏳ **STEP 7:** Documentation

## Models & Metrics

- **XGBoost** (Primary)
- **Evaluation:** PR-AUC (Production metric for imbalanced data)
- **API Response:** <50ms

## 🏃 How to Run STEP 2 Demo

```cmd
cd factoryguard-ai
python -m venv venv
venv\\Scripts\\activate
pip install -r

---
*Built step-by-step for production reliability*
