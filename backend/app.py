from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import traceback

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['factoryguard']
sensors = db['sensors']
predictions = db['predictions']
models = db['models']

class SensorData(BaseModel):
    timestamp: str
    vibration: float
    temperature: float
    pressure: float
    robot_id: str

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = SensorData(**request.json)
        
        # Fetch history
        history_df = fetch_sensor_history(data.robot_id)
        
        # Feature engineering
        features = create_features(history_df)
        
        # Load model and scaler
        model = joblib.load('models/xgb_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # Predict
        X_scaled = scaler.transform(features)
        prob = model.predict_proba(X_scaled)[-1][-1]
        prediction = model.predict(X_scaled)[-1]
        
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        feature_importance = dict(zip(features.columns, shap_values[-1]))
        
        # Store prediction
        pred_doc = {
            'robot_id': data.robot_id,
            'timestamp': data.timestamp,
            'sensors': {'vibration': data.vibration, 'temperature': data.temperature, 'pressure': data.pressure},
            'features': features.iloc[-1].to_dict(),
            'prediction': int(prediction),
            'probability': float(prob),
            'shap_importance': feature_importance,
            'created_at': datetime.now()
        }
        predictions.insert_one(pred_doc)
        
        return jsonify({
            'failure_probability': float(prob),
            'risk_level': get_risk_level(prob),
            'shap_values': feature_importance,
            'explanation': generate_explanation(feature_importance, data),
            'recommended_action': get_recommendation(prob),
            'model_version': 'v2.1.0'
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensors/<robot_id>', methods=['GET'])
def get_sensors(robot_id):
    docs = list(sensors.find({'robot_id': robot_id}).sort('timestamp', -1).limit(1000))
    return jsonify([doc for doc in docs])

@app.route('/api/sensors', methods=['POST'])
def store_sensor():
    data = request.json
    doc = {
        'robot_id': data['robot_id'],
        'timestamp': data['timestamp'],
        'vibration': data['vibration'],
        'temperature': data['temperature'],
        'pressure': data['pressure'],
        'created_at': datetime.now()
    }
    sensors.insert_one(doc)
    return jsonify({'status': 'stored'})

def fetch_sensor_history(robot_id: str, hours: int = 24) -> pd.DataFrame:
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    docs = list(sensors.find({
        'robot_id': robot_id,
        'timestamp': {'$gte': start_time.isoformat(), '$lte': end_time.isoformat()}
    }).sort('timestamp'))
    
    if not docs:
        # Generate synthetic history if no data
        df = generate_synthetic_history(robot_id, hours)
        for _, row in df.iterrows():
            sensors.insert_one(row.to_dict())
        return df
    
    df = pd.DataFrame(docs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df[['vibration', 'temperature', 'pressure']].sort_index()
    return df.resample('1H').mean().interpolate()

def generate_synthetic_history(robot_id: str, hours: int) -> pd.DataFrame:
    timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
    
    # Generate realistic drift towards failure
    vibration = 2.0 + np.cumsum(np.random.normal(0.02, 0.1, hours))
    temperature = 60.0 + np.cumsum(np.random.normal(0.3, 1.0, hours))
    pressure = 12.0 + np.random.normal(0, 0.5, hours)
    
    df = pd.DataFrame({
        'robot_id': robot_id,
        'timestamp': timestamps,
        'vibration': np.clip(vibration, 1, 8),
        'temperature': np.clip(temperature, 50, 95),
        'pressure': np.clip(pressure, 8, 18)
    }).set_index('timestamp')
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    
    # Raw sensor values
    features['vibration'] = df['vibration']
    features['temperature'] = df['temperature'] 
    features['pressure'] = df['pressure']
    
    # Rolling statistics (1H, 6H, 12H)
    for window in [3, 12, 24]:  # Approximate hours
        features[f'vib_mean_{window}'] = df['vibration'].rolling(window).mean()
        features[f'temp_mean_{window}'] = df['temperature'].rolling(window).mean()
        features[f'press_mean_{window}'] = df['pressure'].rolling(window).mean()
        
        features[f'vib_std_{window}'] = df['vibration'].rolling(window).std()
        features[f'temp_std_{window}'] = df['temperature'].rolling(window).std()
        
        # Lag features
        features[f'vib_lag1_{window}'] = df['vibration'].shift(1)
        features[f'temp_lag1_{window}'] = df['temperature'].shift(1)
    
    # Exponential moving averages
    features['vib_ema'] = df['vibration'].ewm(span=12).mean()
    features['temp_ema'] = df['temperature'].ewm(span=12).mean()
    
    # Ratios and differences
    features['vib_temp_ratio'] = features['vibration'] / (features['temperature'] + 1)
    features['temp_press_diff'] = features['temperature'] - features['pressure'] * 5
    
    # Recent change rates
    features['vib_trend'] = features['vibration'].diff(3)
    features['temp_trend'] = features['temperature'].diff(3)
    
    features = features.dropna()
    return features

def get_risk_level(prob: float) -> str:
    if prob >= 0.75: return 'critical'
    if prob >= 0.5: return 'high'
    if prob >= 0.25: return 'medium'
    return 'low'

def generate_explanation(shap_values: dict, data: SensorData) -> str:
    top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    reasons = []
    for feature, importance in top_features:
        if 'vib' in feature and abs(importance) > 0.1:
            reasons.append(f"Vibration anomaly: {data.vibration:.2f}mm/s²")
        elif 'temp' in feature and abs(importance) > 0.1:
            reasons.append(f"Temperature spike: {data.temperature:.1f}°C")
        elif 'press' in feature:
            reasons.append("Pressure system instability")
    return "; ".join(reasons) or "All metrics within normal ranges"

def get_recommendation(prob: float) -> str:
    if prob >= 0.75: return "EMERGENCY SHUTDOWN REQUIRED"
    if prob >= 0.5: return "IMMEDIATE MAINTENANCE SCHEDULED"
    if prob >= 0.25: return "INCREASE MONITORING FREQUENCY"
    return "NORMAL OPERATIONS"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
