/**
 * Local Prediction Engine
 * Provides fallback predictions when API quota is exhausted
 * Uses rule-based logic for industrial sensor analysis
 */

export interface LocalSensorData {
  vibration: number;
  temperature: number;
  pressure: number;
}

export interface LocalPredictionResult {
  failureProbability: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  shapValues: {
    vibration: number;
    temperature: number;
    pressure: number;
  };
  explanation: string;
  recommendedAction: string;
  isLocalMode: boolean;
}

/**
 * Analyzes sensor data locally without API calls
 * Uses industrial thresholds and rule-based heuristics
 */
export function predictFailureLocally(sensor: LocalSensorData, history: LocalSensorData[]): LocalPredictionResult {
  const { vibration, temperature, pressure } = sensor;
  
  // Calculate risk factors based on thresholds
  let vibrationRisk = 0;
  let temperatureRisk = 0;
  let pressureRisk = 0;
  
  // Vibration analysis (normal: 0-3, warning: 3-4.5, critical: >4.5)
  if (vibration > 4.5) {
    vibrationRisk = 0.8;
  } else if (vibration > 3.5) {
    vibrationRisk = 0.4;
  } else if (vibration > 3.0) {
    vibrationRisk = 0.2;
  } else {
    vibrationRisk = vibration / 15;
  }
  
  // Temperature analysis (normal: 50-75, warning: 75-85, critical: >85)
  if (temperature > 85) {
    temperatureRisk = 0.9;
  } else if (temperature > 75) {
    temperatureRisk = 0.5;
  } else if (temperature > 70) {
    temperatureRisk = 0.25;
  } else {
    temperatureRisk = Math.max(0, (temperature - 50) / 100);
  }
  
  // Pressure analysis (normal: 10-15, warning: 15-18, critical: >18 or <8)
  if (pressure > 18 || pressure < 8) {
    pressureRisk = 0.7;
  } else if (pressure > 15 || pressure < 10) {
    pressureRisk = 0.35;
  } else {
    pressureRisk = Math.abs(pressure - 12.5) / 25;
  }
  
  // Analyze trends from history
  let trendPenalty = 0;
  if (history.length >= 5) {
    const recentHistory = history.slice(-5);
    const avgVibration = recentHistory.reduce((a, b) => a + b.vibration, 0) / 5;
    const avgTemperature = recentHistory.reduce((a, b) => a + b.temperature, 0) / 5;
    
    // Check if values are increasing
    if (vibration > avgVibration * 1.1) trendPenalty += 0.1;
    if (temperature > avgTemperature * 1.05) trendPenalty += 0.1;
  }
  
  // Combined failure probability calculation
  const baseProbability = (vibrationRisk * 0.4 + temperatureRisk * 0.35 + pressureRisk * 0.25);
  const failureProbability = Math.min(0.99, baseProbability + trendPenalty);
  
  // Determine risk level
  let riskLevel: 'low' | 'medium' | 'high' | 'critical';
  if (failureProbability >= 0.75) {
    riskLevel = 'critical';
  } else if (failureProbability >= 0.5) {
    riskLevel = 'high';
  } else if (failureProbability >= 0.25) {
    riskLevel = 'medium';
  } else {
    riskLevel = 'low';
  }
  
  // Generate explanation based on highest risk factor
  let explanation = "";
  const maxRisk = Math.max(vibrationRisk, temperatureRisk, pressureRisk);
  
  if (vibrationRisk === maxRisk) {
    explanation = `Local engine: High vibration detected at ${vibration.toFixed(2)}mm/s² (threshold: 4.5mm/s²). Robotic arm bearings may be degrading. Immediate inspection recommended.`;
  } else if (temperatureRisk === maxRisk) {
    explanation = `Local engine: Elevated temperature at ${temperature.toFixed(1)}°C (threshold: 75°C). Motor insulation stress detected. Cooling system check required.`;
  } else {
    explanation = `Local engine: Abnormal pressure reading at ${pressure.toFixed(2)}bar. Hydraulic seal integrity may be compromised.`;
  }
  
  if (trendPenalty > 0) {
    explanation += " Trend analysis shows degrading conditions.";
  }
  
  // Generate recommended action
  let recommendedAction = "";
  switch (riskLevel) {
    case 'critical':
      recommendedAction = "EMERGENCY SHUTDOWN - Inspect robotic arm immediately. Potential catastrophic failure risk.";
      break;
    case 'high':
      recommendedAction = "Schedule immediate maintenance window. Reduce operational load until inspection.";
      break;
    case 'medium':
      recommendedAction = "Plan maintenance within 24-48 hours. Monitor sensor trends closely.";
      break;
    default:
      recommendedAction = "Continue normal operations. Continue monitoring telemetry.";
  }
  
  return {
    failureProbability,
    riskLevel,
    shapValues: {
      vibration: vibrationRisk,
      temperature: temperatureRisk,
      pressure: pressureRisk
    },
    explanation,
    recommendedAction,
    isLocalMode: true
  };
}

/**
 * Check if API should be used or skipped based on recent error history
 */
export function shouldUseLocalFallback(hasRecentQuotaError: boolean): boolean {
  return hasRecentQuotaError;
}
