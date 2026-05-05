from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Dict
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
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

users_db = {'admin': {'password_hash': pwd_context.hash('password'), 'email': 'admin@factoryguard.ai'}}  # Pre-seeded

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

    user_hash = get_password_hash(data.password)
    users_db[data.username] = {'password_hash': user_hash, 'email': data.email}

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
    user_data = users_db.get(data.username, {})
    if not user_data or not verify_password(data.password, user_data['password_hash']):
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
    
    # Mock prediction (no ML deps)
    prob = np.random.uniform(0.05, 0.95)
    risk_level = 'critical' if prob >= 0.75 else 'high' if prob >= 0.5 else 'medium' if prob >= 0.25 else 'low'
    shap_values = {
      'vibration': np.random.uniform(-0.3, 0.3),
      'temperature': np.random.uniform(-0.3, 0.3),
      'pressure': np.random.uniform(-0.3, 0.3)
    }
    
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
      'explanation': 'RT ML prediction',
      'recommended_action': 'Monitor' if prob < 0.5 else 'Inspect',
      'model_version': 'v2.1-lite'
    })
  except Exception as e:
    print(f"Prediction error: {str(e)}")
    return jsonify({'error': str(e)}), 500

@app.route('/api/sensors/<robot_id>', methods=['GET'])
def get_sensors(robot_id):
  token = request.headers.get('Authorization', '').replace('Bearer ', '')
  if not get_current_user(token):
    return jsonify({'error': 'Unauthorized'}), 401
  return jsonify([])  # Empty

@app.route('/api/sensors', methods=['POST'])
def store_sensor():
  token = request.headers.get('Authorization', '').replace('Bearer ', '')
  user = get_current_user(token)
  if not user:
    return jsonify({'error': 'Unauthorized'}), 401
  data = request.json
  socketio.emit('sensor_update', data, namespace='/factory')
  return jsonify({'status': 'stored'})

# SocketIO events
@socketio.on('connect', namespace='/factory')
def handle_connect():
  emit('status', {'msg': 'Connected to FactoryGuard RT'})

def background_sensor_simulator():
  while True:
    time.sleep(2)
    sensor_data = {
      'robot_id': 'robot-001',
      'timestamp': datetime.now().isoformat(),
      'vibration': 2.5 + np.random.normal(0, 0.5),
      'temperature': 65 + np.random.normal(0, 3),
      'pressure': 12 + np.random.normal(0, 1)
    }
    socketio.emit('sensor_update', sensor_data, namespace='/factory')

threading.Thread(target=background_sensor_simulator, daemon=True).start()

if __name__ == '__main__':
  socketio.run(app, debug=True, port=5000)
