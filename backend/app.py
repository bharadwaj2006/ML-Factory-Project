from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
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
import jwt
from passlib.context import CryptContext
import threading
import time

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'factoryguard-dev-secret')
CORS(app, supports_credentials=True)

# SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv('SECRET_KEY', 'factoryguard-dev-secret')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 30

users_db = {}  # In-memory users {username: {'password_hash': hash, 'email': email}}

# Mongo optional - in-memory users fallback
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
try:
  client = MongoClient(MONGO_URI)
  db = client['factoryguard']
  sensors = db['sensors']
  predictions = db['predictions']
  users = db['users']
  client.admin.command('ping')
except:
  # No Mongo - use in-memory
  sensors = predictions = users = None
  print('Mongo unavailable - using in-memory')

def verify_password(plain_password, hashed_password):
  return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
  return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta):
  to_encode = data.copy()
  expire = datetime.utcnow() + timedelta(minutes=expires_delta)
  to_encode.update({"exp": expire})
  encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
  return encoded_jwt

class SensorData(BaseModel):
  timestamp: str
  vibration: float
  temperature: float
  pressure: float
  robot_id: str

class LoginData(BaseModel):
  username: str
  password: str

class RegisterData(BaseModel):
  username: str
  email: str
  password: str

@app.route('/api/register', methods=['POST'])
def register():
  try:
    data = RegisterData(**request.json)
    if data.username in users_db:
      return jsonify({'error': 'User already exists'}), 400

    user_doc = {
      'username': data.username,
      'email': data.email,
      'password_hash': get_password_hash(data.password),
      'created_at': datetime.now()
    }
    if users:
      users.insert_one(user_doc)
    users_db[data.username] = {'password_hash': user_doc['password_hash'], 'email': data.email}

    access_token = create_access_token(
      data={"sub": data.username, "username": data.username, "email": data.email},
      expires_delta=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jsonify({
      'token': access_token,
      'user': {'id': data.username, 'username': data.username, 'email': data.email}
    })
  except Exception as e:
    return jsonify({'error': str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
  try:
    data = LoginData(**request.json)
    user_data = users.find_one({'username': data.username}) if users else None
    user_hash = users_db.get(data.username, {}).get('password_hash')
    if not user_data and not user_hash or not verify_password(data.password, user_hash or user_data['password_hash']):
      return jsonify({'error': 'Invalid credentials'}), 401

    access_token = create_access_token(
      data={"sub": data.username, "username": data.username, "email": user_data['email']},
      expires_delta=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jsonify({
      'token': access_token,
      'user': {'id': data.username, 'username': data.username, 'email': user_data['email']}
    })
  except Exception as e:
    return jsonify({'error': str(e)}), 400

def get_current_user(token: str = None):
  if not token:
    return None
  try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    username = payload.get("sub")
    if username in users_db:
      return {'id': username, 'username': payload.get('username'), 'email': payload.get('email')}
  except:
    pass
  return None

@app.route('/api/predict', methods=['POST'])
def predict():
  token = request.headers.get('Authorization', '').replace('Bearer ', '')
  user = get_current_user(token)
  if not user:
    return jsonify({'error': 'Unauthorized'}), 401

  try:
    data = SensorData(**request.json)
    
    # Fetch history
    history_df = fetch_sensor_history(data.robot_id)
    
    # Feature engineering
    features = create_features(history_df)
    
    # Mock model (since no joblib files)
    prob = np.random.uniform(0.05, 0.95)  # Mock
    risk_level = get_risk_level(prob)
    shap_values = {
      'vibration': np.random.uniform(-0.3, 0.3),
      'temperature': np.random.uniform(-0.3, 0.3),
      'pressure': np.random.uniform(-0.3, 0.3)
    }

    # Store prediction
    pred_doc = {
      'robot_id': data.robot_id,
      'user_id': user['id'],
      'timestamp': data.timestamp,
      'sensors': {'vibration': data.vibration, 'temperature': data.temperature, 'pressure': data.pressure},
      'prediction': 1 if prob > 0.5 else 0,
      'probability': prob,
      'shap_importance': shap_values,
      'created_at': datetime.now()
    }
    if predictions:
      predictions.insert_one(pred_doc)
    
    # Broadcast via socket
    socketio.emit('prediction', {
      'robot_id': data.robot_id,
      'failure_probability': prob,
      'risk_level': risk_level,
      'shap_values': shap_values
    }, namespace='/factory')

    return jsonify({
      'failure_probability': prob,
      'risk_level': risk_level,
      'shap_values': shap_values,
      'explanation': generate_explanation(shap_values, data),
      'recommended_action': get_recommendation(prob),
      'model_version': 'v2.1-mock'
    })
  except Exception as e:
    print(f"Prediction error: {str(e)}")
    return jsonify({'error': str(e)}), 500

@app.route('/api/sensors/<robot_id>', methods=['GET'])
def get_sensors(robot_id):
  token = request.headers.get('Authorization', '').replace('Bearer ', '')
  if not get_current_user(token):
    return jsonify({'error': 'Unauthorized'}), 401
  docs = list(sensors.find({'robot_id': robot_id}).sort('timestamp', -1).limit(1000))
  return jsonify([dict(doc) for doc in docs])

@app.route('/api/sensors', methods=['POST'])
def store_sensor():
  token = request.headers.get('Authorization', '').replace('Bearer ', '')
  user = get_current_user(token)
  if not user:
    return jsonify({'error': 'Unauthorized'}), 401
  data = request.json
  doc = {
    'robot_id': data['robot_id'],
    'user_id': user['id'],
    'timestamp': data['timestamp'],
    'vibration': data['vibration'],
    'temperature': data['temperature'],
    'pressure': data['pressure'],
    'created_at': datetime.now()
  }
  sensors.insert_one(doc)
  # Broadcast sensor update
  socketio.emit('sensor_update', doc, namespace='/factory')
  return jsonify({'status': 'stored'})

# SocketIO events
@socketio.on('connect', namespace='/factory')
def handle_connect():
  emit('status', {'msg': 'Connected to FactoryGuard RT'})

@socketio.on('join_robot', namespace='/factory')
def on_join(data):
  room = data['robot_id']
  join_room(room)
  emit('status', {'msg': f'Joined room {room}'})

@socketio.on('leave_robot', namespace='/factory')
def on_leave(data):
  room = data['robot_id']
  leave_room(room)
  emit('status', {'msg': f'Left room {room}'})

def background_sensor_simulator():
  """Simulate real-time sensor data every 2s"""
  while True:
    time.sleep(2)
    robot_id = 'robot-001'
    timestamp = datetime.now().isoformat()
    vibration = 2.5 + np.random.normal(0, 0.5)
    temperature = 65 + np.random.normal(0, 3)
    pressure = 12 + np.random.normal(0, 1)
    
    sensor_data = {
      'robot_id': robot_id,
      'timestamp': timestamp,
      'vibration': vibration,
      'temperature': temperature,
      'pressure': pressure
    }
    socketio.emit('sensor_update', sensor_data, namespace='/factory')

# Start simulator thread
threading.Thread(target=background_sensor_simulator, daemon=True).start()

# Existing functions (unchanged)
def fetch_sensor_history(robot_id: str, hours: int = 24) -> pd.DataFrame:
  end_time = datetime.now()
  start_time = end_time - timedelta(hours=hours)
  
  docs = list(sensors.find({
    'robot_id': robot_id,
    'timestamp': {'$gte': start_time.isoformat(), '$lte': end_time.isoformat()}
  }).sort('timestamp'))
  
  if not docs:
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
  
  features['vibration'] = df['vibration']
  features['temperature'] = df['temperature'] 
  features['pressure'] = df['pressure']
  
  for window in [3, 12, 24]:
    features[f'vib_mean_{window}'] = df['vibration'].rolling(window).mean()
    features[f'temp_mean_{window}'] = df['temperature'].rolling(window).mean()
    features[f'press_mean_{window}'] = df['pressure'].rolling(window).mean()
    
    features[f'vib_std_{window}'] = df['vibration'].rolling(window).std()
    features[f'temp_std_{window}'] = df['temperature'].rolling(window).std()
    
    features[f'vib_lag1_{window}'] = df['vibration'].shift(1)
    features[f'temp_lag1_{window}'] = df['temperature'].shift(1)
  
  features['vib_ema'] = df['vibration'].ewm(span=12).mean()
  features['temp_ema'] = df['temperature'].ewm(span=12).mean()
  
  features['vib_temp_ratio'] = features['vibration'] / (features['temperature'] + 1)
  features['temp_press_diff'] = features['temperature'] - features['pressure'] * 5
  
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
  socketio.run(app, debug=True, port=5000)

