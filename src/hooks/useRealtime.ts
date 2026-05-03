import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuth } from '../context/AuthContext';

interface SensorUpdate {
  robot_id: string;
  timestamp: string;
  vibration: number;
  temperature: number;
  pressure: number;
}

interface PredictionUpdate {
  robot_id: string;
  failure_probability: number;
  risk_level: string;
  shap_values: Record<string, number>;
}

export function useRealtime(callbacks: {
  onSensorUpdate?: (data: SensorUpdate) => void;
  onPrediction?: (data: PredictionUpdate) => void;
}) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const { token, isAuthenticated } = useAuth();

  useEffect(() => {
    if (!isAuthenticated || !token) return;

    const newSocket = io('http://localhost:5000', {
      path: '/socket.io/',
      auth: { token }
    });

    newSocket.on('connect', () => {
      console.log('Connected to real-time server');
      newSocket.emit('join_robot', { robot_id: 'robot-001' });
    });

    newSocket.on('sensor_update', callbacks.onSensorUpdate);
    newSocket.on('prediction', callbacks.onPrediction);
    newSocket.on('status', (data) => console.log('RT status:', data.msg));

    newSocket.on('disconnect', () => console.log('Disconnected from RT server'));

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [isAuthenticated, token, callbacks]);

  return socket;
}

