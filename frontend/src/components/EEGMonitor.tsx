import React, { useEffect, useState, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid } from 'recharts';
import { bciSocket, type BCIEvent } from '../services/api';

interface EEGMonitorProps {
  isFlashing: boolean;
}

interface DataPoint {
  time: number;
  af7: number;
  af8: number;
  tp9: number;
  tp10: number;
}

const MAX_POINTS = 200;

const EEGMonitor: React.FC<EEGMonitorProps> = ({ isFlashing }) => {
  const [data, setData] = useState<DataPoint[]>([]);
  const [blinkCount, setBlinkCount] = useState(0);
  const [clenchCount, setClenchCount] = useState(0);
  const timeRef = useRef(0);
  const wsSubscribedRef = useRef(false);
  const pendingRef = useRef<DataPoint[]>([]);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const sendSub = () => {
      bciSocket.subscribeEEG();
      wsSubscribedRef.current = true;
    };

    if (bciSocket.connected) {
      sendSub();
    }
    const unsubConnected = bciSocket.on('ws_connected', sendSub);

    const unsubEEG = bciSocket.on('eeg_sample', (e: BCIEvent) => {
      const sample = e.data;
      timeRef.current += 1;
      pendingRef.current.push({
        time: timeRef.current,
        af7: sample.af7 ?? 0,
        af8: sample.af8 ?? 0,
        tp9: sample.tp9 ?? 0,
        tp10: sample.tp10 ?? 0,
      });
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(() => {
          rafRef.current = null;
          const batch = pendingRef.current;
          pendingRef.current = [];
          if (batch.length > 0) {
            setData(prev => [...prev, ...batch].slice(-MAX_POINTS));
          }
        });
      }
    });

    const unsubBlink = bciSocket.on('blink_detected', () => {
      setBlinkCount(c => c + 1);
    });

    const unsubClench = bciSocket.on('clench_detected', () => {
      setClenchCount(c => c + 1);
    });

    return () => {
      unsubConnected();
      unsubEEG();
      unsubBlink();
      unsubClench();
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Fallback: simulated data when no backend data arrives
  useEffect(() => {
    if (data.length > 0) return;

    const interval = setInterval(() => {
      timeRef.current += 1;
      const t = timeRef.current * 0.1;
      const spike = isFlashing && Math.random() > 0.95 ? 50 : 0;
      setData(prev => {
        const next = [...prev, {
          time: timeRef.current,
          af7: (Math.random() - 0.5) * 12 + 8 * Math.sin(2 * Math.PI * 0.8 * t) + spike,
          af8: (Math.random() - 0.5) * 10 + 6 * Math.sin(2 * Math.PI * 1.2 * t + 1.0),
          tp9: (Math.random() - 0.5) * 14 + 7 * Math.sin(2 * Math.PI * 1.5 * t + 2.5),
          tp10: (Math.random() - 0.5) * 11 + 5 * Math.sin(2 * Math.PI * 0.6 * t + 4.0),
        }];
        if (next.length > MAX_POINTS) next.shift();
        return next;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isFlashing, data.length]);

  const channels: Array<{ key: 'af7' | 'af8' | 'tp9' | 'tp10'; label: string; color: string }> = [
    { key: 'af7', label: 'AF7', color: '#3b82f6' },
    { key: 'af8', label: 'AF8', color: '#8b5cf6' },
    { key: 'tp9', label: 'TP9', color: '#06b6d4' },
    { key: 'tp10', label: 'TP10', color: '#f59e0b' },
  ];

  return (
    <div className="w-full bg-white/90 rounded-2xl p-4 border-4 border-blue-200 shadow-xl backdrop-blur-sm">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-3">
          <h3 className="text-blue-500 font-bold text-xs tracking-widest uppercase flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isFlashing ? 'bg-red-500 animate-pulse' : 'bg-blue-400'}`}></div>
            EEG Signal
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-bold text-blue-400">
            👁 {blinkCount} | 💪 {clenchCount}
          </span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${isFlashing ? 'bg-red-100 text-red-500' : 'bg-blue-100 text-blue-500'}`}>
            {isFlashing ? 'ACTIVE' : 'IDLE'}
          </span>
        </div>
      </div>
      <div className="flex flex-col gap-0.5">
        {channels.map(ch => (
          <div key={ch.key} className="flex items-center gap-1">
            <div className="w-10 flex items-center justify-end pr-1 shrink-0">
              <span className="text-[9px] font-bold flex items-center gap-1" style={{ color: ch.color }}>
                <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ backgroundColor: ch.color }}></span>
                {ch.label}
              </span>
            </div>
            <div className="flex-1 h-[72px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 2, right: 4, bottom: 2, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#93c5fd" opacity={0.15} />
                  <YAxis domain={[-80, 80]} hide />
                  <Line
                    type="monotone"
                    dataKey={ch.key}
                    stroke={ch.color}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EEGMonitor;
