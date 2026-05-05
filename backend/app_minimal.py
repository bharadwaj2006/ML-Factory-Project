from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'factoryguard-super-secret'
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")

# Simple auth - hardcoded
USERS = {'admin': 'password'}
SECRET_KEY = 'factoryguard-super-secret'

def create_token(username):
  return jwt.encode({'username': username}, SECRET_KEY, algorithm='HS256')

def verify_token(token):
  try:
    jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    return True
  except:
    return False

@app.route('/api/login', methods=['POST'])
def login():
  data = request.json
  if data.get('username') == 'admin' and data.get('password') == 'password':
    token = create_token('admin')
    return jsonify({'token': token, 'user': {'username': 'admin'}})
  return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/register', methods=['POST'])
def register():
  return jsonify({'error': 'Use login'}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
  token = request.headers.get('Authorization', '').replace('Bearer ', '')
  if not verify_token(token):
    return jsonify({'error': 'Unauthorized'}), 401
  
  data = request.json
  prob = 0.3 + (data.get('vibration', 0) / 10)
  risk = 'low' if prob < 0.25 else 'medium' if prob < 0.5 else 'high'
  
  socketio.emit('prediction', {'risk_level': risk, 'probability': prob})
  
  return jsonify({
    'failure_probability': prob,
    'risk_level': risk,
    'shap_values': {'vibration': 0.4, 'temp': 0.3},
    'explanation': 'Simple calc',
    'recommended_action': 'Monitor'
  })

@app.route('/api/sensors', methods=['POST'])
def sensors():
  token = request.headers.get('Authorization', '').replace('Bearer ', '')
  if not verify_token(token):
    return jsonify({'error': 'Unauthorized'}), 401
  
  data = request.json
  socketio.emit('sensor_update', data)
  return jsonify({'status': 'ok'})

@socketio.on('connect', namespace='/factory')
def connect():
  emit('status', 'Connected')

def simulator():
  while True:
    time.sleep(3)
    sensor_data = {
      'vibration': 2.5 + (time.time() % 10) / 10,
      'temperature': 65,
      'pressure': 12
    }
    socketio.emit('sensor_update', sensor_data, namespace='/factory')

threading.Thread(target=simulator, daemon=True).start()

if __name__ == '__main__':
  socketio.run(app, debug=True, port=5000)
