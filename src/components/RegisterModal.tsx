import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Mail, Lock, User, X, Shield } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from '../lib/utils';

interface RegisterProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToLogin: () => void;
}

export function RegisterModal({ isOpen, onClose, onSwitchToLogin }: RegisterProps) {
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    try {
      const res = await fetch('/api/register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (res.ok) {
        login(data.token, data.user);
        onClose();
      } else {
        setError(data.error || 'Registration failed');
      }
    } catch (err) {
      setError('Network error');
    }
    setIsLoading(false);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[100] flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div 
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            className="bg-zinc-900 border border-zinc-800 rounded-lg p-8 w-full max-w-md"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-8">
              <h2 className="text-2xl font-black tracking-tight flex items-center gap-2">
                <Shield size={24} className="text-hazard" />
                Create Account
              </h2>
              <button onClick={onClose} className="p-1 hover:bg-zinc-800 rounded-lg">
                <X size={20} />
              </button>
            </div>
            <form onSubmit={handleSubmit} className="space-y-6">
              {error && (
                <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-3 rounded-lg text-sm">
                  {error}
                </div>
              )}
              <div className="space-y-4">
                <div>
                  <label className="text-xs font-bold uppercase text-zinc-400 mb-2 block flex items-center gap-2">
                    <User size={14} /> Username
                  </label>
                  <input
                    type="text"
                    value={formData.username}
                    onChange={(e) => setFormData({...formData, username: e.target.value})}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-hazard focus:border-transparent transition-all"
                    required
                  />
                </div>
                <div>
                  <label className="text-xs font-bold uppercase text-zinc-400 mb-2 block flex items-center gap-2">
                    <Mail size={14} /> Email
                  </label>
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({...formData, email: e.target.value})}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-hazard focus:border-transparent transition-all"
                    required
                  />
                </div>
                <div>
                  <label className="text-xs font-bold uppercase text-zinc-400 mb-2 block flex items-center gap-2">
                    <Lock size={14} /> Password
                  </label>
                  <input
                    type="password"
                    value={formData.password}
                    onChange={(e) => setFormData({...formData, password: e.target.value})}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-hazard focus:border-transparent transition-all"
                    required
                    minLength={6}
                  />
                </div>
              </div>
              <button 
                type="submit" 
                disabled={isLoading}
                className={cn(
                  "w-full py-3 rounded-lg font-bold uppercase text-sm transition-all flex items-center justify-center gap-2",
                  isLoading ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed' : 'bg-hazard hover:bg-white text-zinc-950 shadow-lg hover:shadow-hazard/25'
                )}
              >
                {isLoading ? 'Creating Account...' : 'Create Account'}
              </button>
              <div className="text-center text-xs text-zinc-500">
                Have an account? <button onClick={onSwitchToLogin} className="text-hazard font-bold hover:underline">Sign in</button>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

