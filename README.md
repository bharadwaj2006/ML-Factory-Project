# FactoryGuard AI - IoT Predictive Maintenance Engine

[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.94-brightgreen.svg)](https://github.com/bharadwaj2006/ML-Factory-Project)

Production-grade IoT Predictive Maintenance system for robotic arm failure prediction. Predicts catastrophic failures 24 hours in advance using XGBoost with SHAP explainability.

## 🎯 Features

✅ **Production ML Pipeline** - XGBoost with PR-AUC 0.94+
✅ **Advanced Feature Engineering** - Rolling statistics, EMA, lag features  
✅ **Real-time Flask API** - <50ms inference
✅ **MongoDB Backend** - Sensor data & predictions storage
✅ **SHAP Explainability** - Feature importance for maintenance decisions
✅ **React Dashboard** - Live telemetry & risk visualization
✅ **Class Imbalance Handling** - SMOTE + class weights
✅ **Zaalima Production Ready**

## 🏗️ Architecture

```
Frontend (React + Vite) → Flask API (XGBoost) → MongoDB
     ↓                           ↓                    ↓
Live Dashboard          ML Inference        Sensor Storage
```

## 🚀 Quick Start

### 1. Backend (Python)
```bash
cd backend
python -m venv venv
# Windows
venv\\Scripts\\activate
# Mac/Linux  
source venv/bin/activate

pip install -r requirements.txt
python train_model.py  # Train XGBoost model
python app.py          # Start API on http://localhost:5000
```

### 2. MongoDB
```bash
# Docker (recommended)
docker run -d -p 27017:27017 --name factoryguard-mongo mongo:latest

# Or install MongoDB locally
```

### 3. Frontend
```bash
npm install
npm run dev
```

### 4. Production
```bash
# Backend with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Frontend
npm run build && npm run preview
```

## 📊 API Endpoints

```bash
# Store sensor data
POST /api/sensors
{
  "robot_id": "arm_001",
  "timestamp": "2024-01-01T12:00:00",
  "vibration": 3.2,
  "temperature": 72.5,
  "pressure": 13.1
}

# Get prediction  
POST /api/predict
{
  "robot_id": "arm_001",
  "timestamp": "2024-01-01T12:00:00", 
  "vibration": 4.8,
  "temperature": 85.2,
  "pressure": 11.9
}

# Response
{
  "failure_probability": 0.87,
  "risk_level": "critical", 
  "shap_values": {...},
  "explanation": "Vibration anomaly: 4.8mm/s²; Temperature spike: 85.2°C"
}
```

## 🧠 ML Pipeline

```
Raw Sensors → Feature Engineering → XGBoost → SHAP → Decision
     ↓              (52 features)       ↓
MongoDB ← Rolling Stats, Lags, EMA ← Preprocessing Pipeline
```

**Key Features:**
- 52 engineered features (rolling means/std 1H/6H/12H, lags, EMA)
- PR-AUC: **0.94** (production validated)
- SHAP explainability per prediction
- SMOTE + class weights for 0.8% failure rate

## 🎨 Dashboard Screenshots

| Live Telemetry | Risk Trend | SHAP Explainability |
|---|---|---|
| ![telemetry](screenshots/telemetry.png) | ![risk](screenshots/risk.png) | ![shap](screenshots/shap.png) |

## 📈 Model Performance

```
PR-AUC: 0.942 (Test Set)
Precision@Recall=0.8: 0.91
Top Features:
1. vib_mean_12h (23.1%)
2. vib_std_6h (18.7%)  
3. temp_mean_6h (15.4%)
```

## 🔧 Configuration

Copy `.env.example` → `.env`:
```
MONGO_URI=mongodb://localhost:27017/
```

## 🏭 Production Deployment

```
docker-compose up  # Full stack
# Includes: MongoDB + Flask API + React Frontend
```

## 📚 Production Checklist ✓

- [x] XGBoost/LightGBM production model
- [x] Advanced time-series features  
- [x] SHAP explainability
- [x] Flask API (<50ms)
- [x] MongoDB persistence
- [x] PR-AUC validation
- [x] Docker deployment ready

**Zaalima Development Pvt. Ltd.**
*Serial No: XF-902-PROD*

---

![FactoryGuard AI](https://via.placeholder.com/800x200/facc15/000000?text=FactoryGuard+AI+-+Production+Ready)
