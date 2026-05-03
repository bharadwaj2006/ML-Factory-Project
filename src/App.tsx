import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from './context/AuthContext';
import { LoginModal, RegisterModal } from './components/LoginModal';
import { 
  Activity, 
  Thermometer, 
  Gauge, 
  AlertTriangle, 
  ShieldCheck, 
  Info,
  Clock,
  Terminal,
  ChevronRight,
  Database,
  Upload,
  FileJson,
  Play,
  RotateCcw,
  User,
  LogOut
} from 'lucide-react';
import Papa from 'papaparse';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ReferenceLine
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from './lib/utils';
import { useRealtime } from './hooks/useRealtime';

// Types
interface SensorData {
  time: string;
  vibration: number;
  temperature: number;
  pressure: number;
  failureProbability?: number;
}

interface PredictionResult {
  failureProbability: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  shapValues: Record<string, number>;
  explanation: string;
  recommendedAction: string;
}

type ViewMode = 'dashboard' | 'model-lab' | 'stack';

export default function App() {
  const { user, logout, isAuthenticated, loading: authLoading } = useAuth();
  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [dataHistory, setDataHistory] = useState<SensorData[]>([]);
  const [currentPredict, setCurrentPredict] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [isAutoPilot, setIsAutoPilot] = useState(true);
  const [uploadedData, setUploadedData] = useState<SensorData[]>([]);
  const [uploadIndex, setUploadIndex] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [logs, setLogs] = useState<{msg: string, type: 'info' | 'warn' | 'error', time: string}[]>([]);
  const [error, setError] = useState<string | null>(null);

  const socketCallbacks = {
    onSensorUpdate: (data: any) => {
      const sensor: SensorData = {
        time: new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        vibration: data.vibration,
        temperature: data.temperature,
        pressure: data.pressure,
      };
      setDataHistory(prev => [...prev.slice(-39), sensor]);
      addLog(`[RT] Sensor update robot-${data.robot_id}: vib=${data.vibration.toFixed(2)}`, 'info');
    },
    onPrediction: (data: any) => {
      setCurrentPredict({
        failureProbability: data.failure_probability,
        riskLevel: data.risk_level as any,
        shapValues: data.shap_values,
        explanation: 'Real-time ML prediction received',
        recommendedAction: get_recommendation(data.failure_probability),
      });
      addLog(`[RT PREDICT] ${data.risk_level.toUpperCase()} (${(data.failure_probability*100).toFixed(1)}%)`, data.risk_level === 'critical' ? 'error' : data.risk_level === 'high' ? 'warn' : 'info');
    },
  };

  const socket = useRealtime(socketCallbacks);

  // Local model weights (simulated "trained" parameters)
  const [modelWeights, setModelWeights] = useState({
    vibration: 0.15,
    temperature: 0.12,
    pressure: 0.08,
    bias: -0.5
  });

  // Initial Data Generation
  useEffect(() => {
    const initialData = Array.from({ length: 40 }, (_, i) => ({
      time: `${i}:00`,
      vibration: 2.5 + Math.random() * 0.5,
      temperature: 65 + Math.random() * 5,
      pressure: 12 + Math.random() * 2,
    }));
    setDataHistory(initialData);
  }, []);

  // Simulation Loop (Simplified for clarity)
  useEffect(() => {
    if (!isAutoPilot) return;

    const interval = setInterval(() => {
      if (uploadedData.length > 0) {
        setUploadIndex(prev => {
          const next = (prev + 1) % uploadedData.length;
          const nextData = uploadedData[next];
          setDataHistory(history => [...history.slice(1), nextData]);
          return next;
        });
      } else {
        setDataHistory(prev => {
          const last = prev[prev.length - 1];
          const newTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          
          // Smoother, more realistic drift that naturally progresses to high/critical states
          const newVal: SensorData = {
            time: newTime,
            vibration: Math.max(1, last.vibration + (Math.random() - 0.4) * 0.5),
            temperature: Math.max(40, last.temperature + (Math.random() - 0.4) * 2),
            pressure: Math.max(5, last.pressure + (Math.random() - 0.4) * 0.5),
            failureProbability: last.failureProbability // Carry over last prediction for graph continuity
          };
          
          // Reset if it gets too theoretically impossible
          if (newVal.vibration > 8) newVal.vibration = 2.5 + Math.random();
          if (newVal.temperature > 110) newVal.temperature = 65 + Math.random() * 5;
          if (newVal.pressure > 20) newVal.pressure = 12 + Math.random();

          return [...prev.slice(1), newVal];
        });
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isAutoPilot, uploadedData]);

  // Trigger Prediction & Logs
  const lastProcessedTime = useRef<string | null>(null);

  useEffect(() => {
    const latest = dataHistory[dataHistory.length - 1];
    if (latest && isAutoPilot && lastProcessedTime.current !== latest.time) {
      lastProcessedTime.current = latest.time;
      // Run prediction every tick, but since it's local we don't need artificial visual delay 
      // that causes flickering.
      runPrediction(latest);

      if (latest.vibration > 4.5) {
        addLog(`[ANOMALY] High vibration detected: ${latest.vibration.toFixed(2)}mm/s²`, 'warn');
      }
    }
  }, [dataHistory, isAutoPilot]);

  const addLog = (msg: string, type: 'info' | 'warn' | 'error' = 'info') => {
    setLogs(prev => [{ msg, type, time: new Date().toLocaleTimeString() }, ...prev.slice(0, 49)]);
  };

  const runPrediction = (sensor: SensorData) => {
    setError(null);
    
    try {
      // Local Heuristic "ML Model"
      // score = (vibration * w1 + temp * w2 + pressure * w3) + bias
      // normalized around thresholds
      const vScore = (sensor.vibration / 5) * modelWeights.vibration;
      const tScore = (sensor.temperature / 100) * modelWeights.temperature;
      const pScore = (sensor.pressure / 20) * modelWeights.pressure;
      
      const rawScore = (vScore + tScore + pScore) * 25 - 6.5 + modelWeights.bias;
      const failProb = Math.min(0.99, Math.max(0.01, 1 / (1 + Math.exp(-rawScore))));

      let riskLevel: PredictionResult['riskLevel'] = 'low';
      if (failProb > 0.85) riskLevel = 'critical';
      else if (failProb > 0.65) riskLevel = 'high';
      else if (failProb > 0.35) riskLevel = 'medium';

      const shapValues = {
        vibration: (vScore / (vScore + tScore + pScore || 1)),
        temperature: (tScore / (vScore + tScore + pScore || 1)),
        pressure: (pScore / (vScore + tScore + pScore || 1))
      };

      const prediction = {
        failureProbability: failProb,
        riskLevel,
        shapValues,
        explanation: `Analysis indicates ${riskLevel} risk. Vibration contribution is ${(shapValues.vibration * 100).toFixed(1)}%. Logic gates mapped to local weights [v:${modelWeights.vibration.toFixed(2)}, t:${modelWeights.temperature.toFixed(2)}].`,
        recommendedAction: failProb > 0.6 
          ? "SCHEDULE IMMEDIATE HARDWARE AUDIT" 
          : failProb > 0.3 
            ? "INCREASE TELEMETRY FREQUENCY" 
            : "MAINTAIN CURRENT OPERATION PARAMETERS"
      };

      setCurrentPredict(prediction);

      // Update history with the prediction result for graphing
      setDataHistory(prev => {
        const next = [...prev];
        if (next.length > 0) {
          next[next.length - 1] = { ...next[next.length - 1], failureProbability: failProb };
        }
        return next;
      });

    } catch (err: any) {
      console.error("Prediction Error:", err);
      setError("Inference Error");
      addLog("[ERROR] Local kernel failed to compute state.", "error");
    }
  };

  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);

  const handleRetrain = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setTrainProgress(0);
    setIsAutoPilot(false);
    addLog("[SYSTEM] Initiating model re-training cycle...", "info");

    const epochs = 5;
    for (let i = 1; i <= epochs; i++) {
        await new Promise(r => setTimeout(r, 600));
        setTrainProgress((i / epochs) * 100);
        
        // Simulate "learning" by slightly adjusting weights
        setModelWeights(prev => ({
          vibration: prev.vibration * (0.95 + Math.random() * 0.1),
          temperature: prev.temperature * (0.95 + Math.random() * 0.1),
          pressure: prev.pressure * (0.95 + Math.random() * 0.1),
          bias: prev.bias + (Math.random() - 0.5) * 0.1
        }));

        addLog(`[TRAIN] Epoch ${i}/${epochs} - Loss: ${(0.42 / i).toFixed(4)} - Alpha: ${(0.01 / i).toFixed(5)}`, "info");
    }

    await new Promise(r => setTimeout(r, 400));
    setIsTraining(false);
    setTrainProgress(0);
    setIsAutoPilot(true);
    addLog("[SUCCESS] Model recalibrated. Weights synchronized to local baseline.", "info");
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.type === "application/json") {
      const reader = new FileReader();
      reader.onload = (event) => {
        const json = JSON.parse(event.target?.result as string);
        if (Array.isArray(json)) { setUploadedData(json); setUploadIndex(0); setIsAutoPilot(true); }
      };
      reader.readAsText(file);
    } else if (file.name.endsWith(".csv")) {
      Papa.parse(file, { header: true, dynamicTyping: true, complete: (res) => {
          const parsed = res.data.map((row: any) => ({
            time: row.time || new Date().toLocaleTimeString(),
            vibration: row.vibration || 0,
            temperature: row.temperature || 0,
            pressure: row.pressure || 0
          })).filter(r => r.vibration !== undefined);
          setUploadedData(parsed); setUploadIndex(0); setIsAutoPilot(true);
      }});
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 flex font-mono text-zinc-100 overflow-x-hidden">
      {/* Mobile Drawer Overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setMobileMenuOpen(false)}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[60] lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar Navigation */}
      <aside className={cn(
        "fixed inset-y-0 left-0 w-64 border-r border-zinc-800 bg-zinc-900 flex flex-col z-[70] transition-transform duration-300 lg:relative lg:translate-x-0",
        mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="p-6 border-b border-zinc-800">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-6 h-6 bg-hazard rounded-sm flex items-center justify-center text-zinc-950 shadow-[0_0_15px_rgba(250,204,21,0.3)]">
              <Activity size={14} strokeWidth={3} />
            </div>
            <h1 className="text-sm font-black tracking-tighter uppercase italic">
              FactoryGuard <span className="text-hazard">AI</span>
            </h1>
          </div>
          <p className="text-[10px] text-zinc-600 font-bold tracking-widest leading-none">ZAALIMA PRODUCTION NODE</p>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          <NavItem 
            active={viewMode === 'dashboard'} 
            onClick={() => { setViewMode('dashboard'); setMobileMenuOpen(false); }}
            icon={<Gauge size={16} />} 
            label="Live Telemetry" 
          />
          <NavItem 
            active={viewMode === 'model-lab'} 
            onClick={() => { setViewMode('model-lab'); setMobileMenuOpen(false); }}
            icon={<Database size={16} />} 
            label="Model Lab" 
          />
          <NavItem 
            active={viewMode === 'stack'} 
            onClick={() => { setViewMode('stack'); setMobileMenuOpen(false); }}
            icon={<FileJson size={16} />} 
            label="Production Stack" 
          />
        </nav>

        <div className="p-4 border-t border-zinc-800 bg-zinc-950/50">
          <div className="bg-zinc-900/50 p-3 rounded-sm border border-zinc-800 space-y-2">
            <div className="flex items-center justify-between text-[10px] uppercase font-bold text-zinc-500">
              <span>Environment</span>
              <span className="text-emerald-500">Production</span>
            </div>
            <div className="flex items-center justify-between text-[10px] uppercase font-bold text-zinc-500">
              <span>Latency</span>
              <span className="text-zinc-400">22ms</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col relative w-full lg:max-w-[calc(100vw-16rem)] overflow-y-auto">
        {/* Top Action Bar */}
        <header className="h-16 border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md sticky top-0 z-40 flex items-center justify-between px-4 lg:px-8">
          <div className="flex items-center gap-4 text-[10px] lg:text-xs font-bold uppercase text-zinc-500">
            <button 
              onClick={() => setMobileMenuOpen(true)}
              className="lg:hidden p-2 -ml-2 text-zinc-400 hover:text-white"
            >
              <Activity size={20} />
            </button>
            <span className="hidden sm:inline text-hazard">[{viewMode.replace('-', '_').toUpperCase()}]</span>
            <span className="hidden sm:inline text-zinc-700">/</span>
            <div className="flex items-center gap-2">
              <div className={cn("w-1.5 h-1.5 rounded-full", uploadedData.length > 0 ? "bg-blue-400 animate-pulse" : "bg-emerald-500")} />
              <span className="truncate max-w-[100px] lg:max-w-none">
                {uploadedData.length > 0 ? `BATCH_RUN: ${uploadIndex}/${uploadedData.length}` : "LIVE_STREAM"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 lg:gap-4">
            <input type="file" ref={fileInputRef} onChange={handleFileUpload} className="hidden" accept=".csv,.json" />
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="hidden md:block px-3 py-1.5 border border-zinc-800 hover:border-hazard rounded-sm text-[10px] font-bold uppercase transition-all"
            >
              Feed Dataset
            </button>
            <button 
              onClick={() => setIsAutoPilot(!isAutoPilot)}
              className={cn(
                "px-2 lg:px-4 py-1.5 rounded-sm text-[10px] font-bold border transition-all truncate",
                isAutoPilot 
                  ? "bg-hazard text-zinc-950 border-hazard" 
                  : "border-zinc-800 text-zinc-500"
              )}
            >
              <span className="sm:hidden">{isAutoPilot ? 'ACTIVE' : 'PAUSED'}</span>
              <span className="hidden sm:inline">{isAutoPilot ? 'SIMULATION_ACTIVE' : 'SIMULATION_PAUSED'}</span>
            </button>
          </div>
        </header>

        <main className="p-4 lg:p-8 flex-1 min-w-0 overflow-hidden">
          <AnimatePresence mode="wait">
            {viewMode === 'dashboard' && (
              <motion.div 
                key="dashboard"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-6"
              >
                {/* Dashboard Components */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <SensorCard title="VIBRATION" value={dataHistory[dataHistory.length - 1]?.vibration?.toFixed(3)} unit="mm/s²" icon={<Activity size={16} />} />
                  <SensorCard title="TEMPERATURE" value={dataHistory[dataHistory.length - 1]?.temperature?.toFixed(1)} unit="°C" icon={<Thermometer size={16} />} />
                  <SensorCard title="PRESSURE" value={dataHistory[dataHistory.length - 1]?.pressure?.toFixed(2)} unit="bar" icon={<Gauge size={16} />} />
                  <PredictionSummaryCard prediction={currentPredict} loading={loading} error={error} />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 space-y-6">
                    <div className="bg-zinc-900 border border-zinc-800 p-4 lg:p-6 rounded-sm">
                      <div className="flex justify-between items-end mb-6">
                        <div>
                          <h3 className="text-xs font-black uppercase text-zinc-500 italic">Risk Level Trend</h3>
                          <p className="text-[10px] text-zinc-700 uppercase">Probability mapped against alert zones</p>
                        </div>
                        <div className="flex gap-4">
                          <LegendItem color="#ef4444" label="Critical" />
                          <LegendItem color="#f97316" label="High" />
                          <LegendItem color="#3b82f6" label="Medium" />
                          <LegendItem color="#10b981" label="Low" />
                        </div>
                      </div>
                      <div className="h-48 lg:h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={dataHistory}>
                            <defs>
                              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.2}/>
                                <stop offset="15%" stopColor="#f97316" stopOpacity={0.15}/>
                                <stop offset="35%" stopColor="#3b82f6" stopOpacity={0.1}/>
                                <stop offset="65%" stopColor="#10b981" stopOpacity={0.05}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                            <XAxis dataKey="time" hide />
                            <YAxis domain={[0, 1]} hide />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#09090b', border: '1px solid #27272a', fontSize: '10px' }}
                              formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Failure Prob']}
                            />
                            
                            {/* Background Zones */}
                            <ReferenceLine y={0.85} stroke="#ef4444" strokeWidth={1} strokeDasharray="5 5" label={{ value: 'CRITICAL', position: 'insideRight', fill: '#ef4444', fontSize: 8 }} />
                            <ReferenceLine y={0.65} stroke="#f97316" strokeWidth={1} strokeDasharray="3 3" label={{ value: 'HIGH', position: 'insideRight', fill: '#f97316', fontSize: 8 }} />
                            <ReferenceLine y={0.35} stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" label={{ value: 'MED', position: 'insideRight', fill: '#3b82f6', fontSize: 8 }} />

                            <Area 
                              type="monotone" 
                              dataKey="failureProbability" 
                              stroke="#fbbf24" 
                              strokeWidth={3}
                              fill="url(#riskGradient)" 
                              isAnimationActive={false} 
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="bg-zinc-900 border border-zinc-800 p-4 lg:p-6 rounded-sm">
                      <div className="flex justify-between items-end mb-6">
                        <div>
                          <h3 className="text-xs font-black uppercase text-zinc-500 italic">Temporal Analysis</h3>
                          <p className="text-[10px] text-zinc-700 uppercase">Multi-variate sensor fusion display</p>
                        </div>
                      </div>
                      <div className="h-64 lg:h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={dataHistory}>
                            <defs>
                              <linearGradient id="colorVib" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                            <XAxis dataKey="time" stroke="#3f3f46" fontSize={10} axisLine={false} tickLine={false} />
                            <YAxis stroke="#3f3f46" fontSize={10} axisLine={false} tickLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: '#09090b', border: '1px solid #27272a', fontSize: '10px' }} />
                            <Area type="monotone" dataKey="vibration" stroke="#0ea5e9" fillOpacity={1} fill="url(#colorVib)" isAnimationActive={false} />
                            <Area type="monotone" dataKey="temperature" stroke="#f97316" fill="transparent" isAnimationActive={false} />
                            <Area type="monotone" dataKey="pressure" stroke="#10b981" fill="transparent" isAnimationActive={false} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-zinc-900 border border-zinc-800 p-6 rounded-sm">
                        <h3 className="text-xs font-black uppercase text-zinc-500 italic mb-4 flex items-center gap-2">
                          <Clock size={14} /> Maintenance Schedule
                        </h3>
                        <div className="space-y-4">
                          <ScheduleItem task="ROBOT_ARM_CALIBRATION" time="MAY 02, 14:00" priority="low" />
                          <ScheduleItem task="LUBRICATION_NODE_492" time="MAY 04, 09:30" priority="medium" />
                          {currentPredict?.riskLevel === 'high' || currentPredict?.riskLevel === 'critical' ? (
                            <ScheduleItem task="EMERGENCY_INSPECTION" time="IMMEDIATE" priority="high" pulse />
                          ) : (
                            <ScheduleItem task="SYSTEM_FLUSH_P3" time="MAY 08, 22:00" priority="low" />
                          )}
                        </div>
                      </div>
                      <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-sm flex flex-col">
                        <h3 className="text-xs font-black uppercase text-zinc-500 italic mb-4 flex items-center gap-2">
                          <Terminal size={14} /> Kernel Events
                        </h3>
                        <div className="flex-1 space-y-1.5 overflow-y-auto max-h-[160px] font-mono text-[9px]">
                          {logs.map((log, i) => (
                            <p key={i} className={cn(
                              "leading-tight",
                              log.type === 'warn' ? 'text-hazard' : log.type === 'error' ? 'text-red-500' : 'text-zinc-500'
                            )}>
                              <span className="opacity-30 mr-1">[{log.time}]</span> {log.msg}
                            </p>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="bg-zinc-900 border border-zinc-800 p-6 rounded-sm">
                    <h3 className="text-xs font-black uppercase text-zinc-500 italic mb-4">SHAP Explainability</h3>
                    <div className="space-y-4">
                      {currentPredict ? Object.entries(currentPredict.shapValues).map(([k, v]) => {
                        const val = v as number;
                        return (
                          <div key={k} className="space-y-1">
                            <div className="flex justify-between text-[10px] uppercase font-bold">
                              <span className="text-zinc-400">{k}</span>
                              <span className={val > 0 ? 'text-red-400' : 'text-emerald-400'}>{(val * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                              <div className={cn("h-full", val > 0 ? "bg-red-500" : "bg-emerald-500")} style={{ width: `${Math.abs(val) * 100}%` }} />
                            </div>
                          </div>
                        );
                      }) : <div className="text-[10px] text-zinc-700 italic">Calculating local feature impact...</div>}
                    </div>
                    <div className="mt-6 bg-zinc-950 p-4 border-l-2 border-hazard text-[11px] text-zinc-400 italic leading-relaxed">
                      {currentPredict?.explanation || "Awaiting sensor window threshold completion..."}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {viewMode === 'model-lab' && (
              <motion.div 
                key="model-lab"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {/* PR Curve */}
                  <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 p-6 rounded-sm">
                    <h3 className="text-xs font-black uppercase text-zinc-500 mb-6 flex items-center gap-2">
                       <Activity size={14} className="text-hazard" />
                       Precision-Recall Performance (Production Model)
                    </h3>
                    <div className="h-64 flex items-end justify-between gap-2 px-4 relative">
                      {[0.95, 0.92, 0.88, 0.82, 0.75, 0.65, 0.4, 0.2].map((v, i) => (
                        <motion.div 
                          key={i} 
                          initial={{ height: 0 }}
                          animate={{ height: `${v * 100}%` }}
                          className="flex-1 bg-gradient-to-t from-hazard/5 to-hazard/60 border-t-2 border-hazard" 
                        />
                      ))}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="px-4 py-2 bg-zinc-950 border border-hazard text-hazard text-[10px] font-black uppercase tracking-widest shadow-[0_0_15px_rgba(250,204,21,0.2)]">
                          PR_AUC: 0.942
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between text-[10px] text-zinc-600 mt-4 uppercase font-black italic">
                      <span>Recall (Sensitivity)</span>
                      <span>Precision (v2.1 Master)</span>
                    </div>
                  </div>

                  {/* Tuning Controls */}
                  <div className="bg-zinc-900 border border-zinc-800 p-6 rounded-sm space-y-6">
                    <div>
                      <h3 className="text-xs font-black uppercase text-zinc-500 mb-4 italic">Hyper-optimization</h3>
                      <div className="space-y-6">
                        <TuningSlider label="Learning Rate" value={0.025} min={0.001} max={0.1} />
                        <TuningSlider label="Max Depth" value={8} min={3} max={15} />
                        <TuningSlider label="Scale Pos Weight" value={120} min={1} max={500} />
                      </div>
                    </div>
                    <div className="pt-6 border-t border-zinc-800 space-y-4">
                      {isTraining && (
                        <div className="space-y-1">
                          <div className="flex justify-between text-[9px] font-black text-hazard uppercase">
                            <span>Training Progress</span>
                            <span>{Math.round(trainProgress)}%</span>
                          </div>
                          <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                            <motion.div 
                              className="h-full bg-hazard"
                              initial={{ width: 0 }}
                              animate={{ width: `${trainProgress}%` }}
                            />
                          </div>
                        </div>
                      )}
                      <button 
                        onClick={handleRetrain}
                        disabled={isTraining}
                        className={cn(
                          "w-full py-2 text-[10px] font-black uppercase tracking-widest transition-colors",
                          isTraining 
                            ? "bg-zinc-800 text-zinc-600 cursor-not-allowed" 
                            : "bg-hazard text-zinc-950 hover:bg-white"
                        )}
                      >
                        {isTraining ? 'TRAINING_IN_PROGRESS...' : 'Re-train Engine'}
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {viewMode === 'stack' && (
              <motion.div 
                key="stack"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                className="bg-zinc-900 border border-zinc-800 p-12 rounded-sm max-w-4xl mx-auto"
              >
                <div className="text-center mb-12">
                  <h2 className="text-2xl font-black italic tracking-tighter uppercase mb-2">Zaalima Production AI Architecture</h2>
                  <p className="text-[10px] text-zinc-500 tracking-widest uppercase">Proprietary Hybrid Intelligence Infrastructure</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
                   {/* Stack Items */}
                   <StackItem title="Data Foundation" tools={["Local Stream", "Ring Buffer"]} desc="High-velocity telemetry captured in a localized 10s window." />
                   <StackItem title="Modeling Engine" tools={["Heuristic ML", "Kernel Inference"]} desc="Proprietary weight-based inference running fully on-edge." />
                   <StackItem title="Deployment/Ops" tools={["Vite Runtime", "React Core"]} desc="Isolated production enviroment with zero external latency." />
                </div>

                <div className="mt-12 pt-12 border-t border-zinc-800">
                  <div className="flex items-center gap-6 justify-center">
                    <img src="https://img.icons8.com/color/48/docker.png" className="grayscale opacity-50 hover:grayscale-0 transition-all cursor-crosshair" alt="Docker" />
                    <img src="https://img.icons8.com/color/48/apache-spark.png" className="grayscale opacity-50 hover:grayscale-0 transition-all cursor-crosshair" alt="Spark" />
                    <img src="https://img.icons8.com/color/48/python.png" className="grayscale opacity-50 hover:grayscale-0 transition-all cursor-crosshair" alt="Python" />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        <footer className="mt-auto border-t border-zinc-900 py-6 px-8 flex justify-between items-center bg-zinc-950/20">
          <p className="text-[10px] text-zinc-700 tracking-widest font-black uppercase">
            ZAALIMA DEVELOPMENT PVT. LTD. // SERIAL_NO: XF-902-PROD
          </p>
          <div className="flex gap-4">
             <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
             <div className="w-2 h-2 rounded-full bg-emerald-500" />
             <div className="w-2 h-2 rounded-full bg-zinc-800" />
          </div>
        </footer>
      </div>
    </div>
  );
}

function ScheduleItem({ task, time, priority, pulse }: any) {
  const colors = {
    low: 'bg-emerald-500',
    medium: 'bg-blue-400',
    high: 'bg-red-500'
  };
  return (
    <div className={cn(
      "p-3 bg-zinc-950 border border-zinc-800 rounded-sm flex items-center justify-between",
      pulse && "animate-pulse border-red-900/50"
    )}>
      <div className="space-y-0.5">
        <div className="text-[10px] font-black uppercase text-zinc-400">{task}</div>
        <div className="text-[10px] text-zinc-600 font-bold">{time}</div>
      </div>
      <div className={cn("w-1.5 h-6 rounded-full opacity-50", colors[priority as keyof typeof colors])} />
    </div>
  );
}

function NavItem({ active, icon, label, onClick }: any) {
  return (
    <button 
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2.5 rounded-sm text-xs font-bold uppercase tracking-tighter transition-all group relative",
        active ? "bg-hazard text-zinc-950 shadow-lg" : "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800"
      )}
    >
      {icon}
      {label}
      {active && <div className="absolute right-2 w-1.5 h-1.5 bg-zinc-950 rounded-full" />}
    </button>
  );
}

function TuningSlider({ label, value, min, max }: any) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-[9px] uppercase font-bold text-zinc-500">
        <span>{label}</span>
        <span className="text-hazard">{value}</span>
      </div>
      <div className="h-1 bg-zinc-800 rounded-full overflow-hidden relative">
        <div 
          className="absolute inset-y-0 left-0 bg-hazard/40" 
          style={{ width: `${((value - min) / (max - min)) * 100}%` }} 
        />
      </div>
    </div>
  );
}

function StackItem({ title, tools, desc }: any) {
  return (
    <div className="p-6 bg-zinc-950 border border-zinc-800 rounded-sm hover:border-hazard transition-colors group">
      <h4 className="text-[10px] font-black text-hazard uppercase mb-4 italic tracking-widest">{title}</h4>
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {tools.map((t: string) => (
            <span key={t} className="px-1.5 py-0.5 bg-zinc-900 text-[9px] font-bold text-zinc-400 border border-zinc-800 rounded-sm">{t}</span>
          ))}
        </div>
        <p className="text-[10px] text-zinc-600 leading-relaxed font-bold uppercase">{desc}</p>
      </div>
    </div>
  );
}

function SensorCard({ title, value, unit, icon }: any) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-sm p-4 hover:border-zinc-700 transition-all group overflow-hidden relative">
      <div className="absolute -right-4 -top-4 opacity-5 group-hover:opacity-10 transition-opacity">
        {React.cloneElement(icon, { size: 64 })}
      </div>
      <span className="text-[10px] font-black text-zinc-600 uppercase italic tracking-tighter mb-4 block">{title}</span>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-black italic tracking-tighter">{value || '---'}</span>
        <span className="text-[10px] text-zinc-600 font-bold uppercase">{unit}</span>
      </div>
    </div>
  );
}

function PredictionSummaryCard({ prediction, loading, error }: { prediction: PredictionResult | null, loading: boolean, error: string | null }) {
  const riskColors: Record<string, string> = { low: 'text-emerald-500', medium: 'text-blue-400', high: 'text-orange-400', critical: 'text-red-500' };
  return (
    <div className={cn(
      "bg-zinc-900 border rounded-sm p-4 transition-all relative overflow-hidden flex flex-col justify-between",
      prediction?.riskLevel === 'critical' ? 'border-red-500' : error ? 'border-red-900/50' : 'border-zinc-800'
    )}>
      {/* Subtle Progress Bar for Loading */}
      {loading && (
        <motion.div 
          initial={{ x: '-100%' }}
          animate={{ x: '100%' }}
          transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
          className="absolute top-0 left-0 right-0 h-[2px] bg-hazard/50 z-20"
        />
      )}
      
      <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] font-black text-zinc-600 uppercase italic">P_FAILURE_24H</span>
        <div className={cn(
          "px-1.5 py-0.5 rounded-full text-[8px] font-black uppercase tracking-widest",
          loading ? "bg-hazard/10 text-hazard animate-pulse" : "bg-zinc-800 text-zinc-500"
        )}>
          {loading ? "Inference" : "Synced"}
        </div>
      </div>

      <div className="flex items-center justify-between mt-auto">
        <span className={cn(
          "text-2xl font-black italic tracking-tighter uppercase", 
          error ? "text-red-500 text-xs" : prediction ? riskColors[prediction.riskLevel] : 'text-zinc-600',
          prediction?.riskLevel === 'critical' && "animate-pulse"
        )}>
          {error || (prediction ? prediction.riskLevel : 'IDLE')}
        </span>
        {!error && (
          <span className="text-xl font-bold font-mono">
            {prediction ? Math.round(prediction.failureProbability * 100) : 0}%
          </span>
        )}
      </div>
    </div>
  );
}


function LegendItem({ color, label }: { color: string, label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-[10px] font-black text-zinc-500 uppercase">{label}</span>
    </div>
  );
}
