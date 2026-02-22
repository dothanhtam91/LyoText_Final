import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { speakText, speakWithWebSpeech } from '../services/gemini';
import {
  bciSocket,
  getPhrases,
  getStatus,
  startSelection,
  stopSelection,
  doneSend,
  clearSentence as clearSentenceAPI,
  type BCIEvent,
  type SystemStatus,
} from '../services/api';
import {
  Volume2,
  Play,
  StopCircle,
  Send,
  RefreshCw,
  Wifi,
  WifiOff,
  Eye,
  Zap,
  Check,
  MoreHorizontal,
  Square,
  SkipForward,
  ChevronRight,
  Trash2,
} from 'lucide-react';
import EEGMonitor from './EEGMonitor';
import BandPowerHistogram from './BandPowerHistogram';

const FALLBACK_PHRASES = ["I", "we", "patient", "nurse", "doctor", "Other"];
const OTHER_LABEL = "Other";
const SKIP_LABEL = "Skip";

const GRAMMAR_STEPS = [
  { key: 'subject', label: 'Subject', color: 'blue' },
  { key: 'adverb', label: 'Adverb', color: 'purple' },
  { key: 'adjective', label: 'Adjective', color: 'amber' },
  { key: 'action', label: 'Action', color: 'emerald' },
];

type SelectionPhase = 'idle' | 'warmup' | 'calibrating' | 'clench_calibrating' | 'highlighting' | 'confirming' | 'executing' | 'stopped';

const P300Grid: React.FC = () => {
  const [phrases, setPhrases] = useState<string[]>(FALLBACK_PHRASES);
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<string>('');

  const [blinkFlash, setBlinkFlash] = useState(false);
  const [clenchFlash, setClenchFlash] = useState(false);

  const [selPhase, setSelPhase] = useState<SelectionPhase>('idle');
  const [warmupProgress, setWarmupProgress] = useState(0);
  const [calBlinks, setCalBlinks] = useState(0);
  const [calNeeded, setCalNeeded] = useState(2);
  const [highlightIndex, setHighlightIndex] = useState<number | null>(null);
  const [confirmedIndex, setConfirmedIndex] = useState<number | null>(null);
  const [confirmedPhrase, setConfirmedPhrase] = useState<string>('');

  const [grammarStep, setGrammarStep] = useState<string>('subject');
  const [grammarStepIndex, setGrammarStepIndex] = useState(0);
  const [selectedSlots, setSelectedSlots] = useState<Record<string, string>>({});
  const [isSkippable, setIsSkippable] = useState(false);

  const [showDeleteToast, setShowDeleteToast] = useState(false);
  const [showClenchPending, setShowClenchPending] = useState(false);
  const [clenchCalCount, setClenchCalCount] = useState(0);
  const [clenchCalNeeded, setClenchCalNeeded] = useState(3);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const deleteAudioCtxRef = useRef<AudioContext | null>(null);

  const playDeleteSound = useCallback(() => {
    try {
      if (!deleteAudioCtxRef.current) {
        deleteAudioCtxRef.current = new AudioContext();
      }
      const ctx = deleteAudioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.2);
    } catch { /* audio not available */ }
  }, []);

  useEffect(() => {
    bciSocket.connect();

    const fetchInitial = async () => {
      try {
        const [st, ph] = await Promise.all([getStatus(), getPhrases()]);
        setStatus(st);
        if (Array.isArray(ph.phrases) && ph.phrases.length > 0) {
          setPhrases(ph.phrases);
        }
        if (ph.grammar_step) setGrammarStep(ph.grammar_step);
        if (ph.grammar_step_index !== undefined) setGrammarStepIndex(ph.grammar_step_index);
        if (ph.skippable !== undefined) setIsSkippable(ph.skippable);
        if (ph.selected_slots) setSelectedSlots(ph.selected_slots);
      } catch {
        console.warn('Backend not reachable, using fallback mode');
      }
    };
    fetchInitial();

    return () => bciSocket.disconnect();
  }, []);

  useEffect(() => {
    const unsubs: (() => void)[] = [];

    unsubs.push(
      bciSocket.on('ws_connected', () => setWsConnected(true)),
      bciSocket.on('ws_disconnected', () => setWsConnected(false)),

      bciSocket.on('phrases_updated', (e: BCIEvent) => {
        const newPhrases = e.data.phrases;
        if (Array.isArray(newPhrases) && newPhrases.length > 0) {
          setPhrases(newPhrases);
        }
      }),

      bciSocket.on('words_updated', (e: BCIEvent) => {
        const { phrases: fullPhrases, grammar_step, grammar_step_index, skippable, selected_slots } = e.data;
        if (Array.isArray(fullPhrases) && fullPhrases.length > 0) {
          setPhrases(fullPhrases);
        }
        if (grammar_step) setGrammarStep(grammar_step);
        if (grammar_step_index !== undefined) setGrammarStepIndex(grammar_step_index);
        if (skippable !== undefined) setIsSkippable(skippable);
        if (selected_slots) setSelectedSlots(selected_slots);
      }),

      bciSocket.on('grammar_step_changed', (e: BCIEvent) => {
        const { step, step_index, skippable: sk, selected_slots: slots } = e.data;
        if (step) setGrammarStep(step);
        if (step_index !== undefined) setGrammarStepIndex(step_index);
        if (sk !== undefined) setIsSkippable(sk);
        if (slots) setSelectedSlots(slots);
      }),

      bciSocket.on('word_selected', (e: BCIEvent) => {
        if (e.data.selected_slots) setSelectedSlots(e.data.selected_slots);
        if (e.data.grammar_step) setGrammarStep(e.data.grammar_step);
        if (e.data.word) setLastEvent(`Added: "${e.data.word}"`);
      }),

      bciSocket.on('sentence_cleared', (e: BCIEvent) => {
        setSelectedSlots({});
        setGrammarStep('subject');
        setGrammarStepIndex(0);
        setShowClenchPending(false);
        if (e.data.spoken) {
          setLastEvent(`Spoke: "${e.data.spoken}"`);
        } else {
          setShowDeleteToast(true);
          playDeleteSound();
          setTimeout(() => setShowDeleteToast(false), 2000);
          setLastEvent('Sentence deleted');
        }
      }),

      bciSocket.on('clench_pending', () => {
        setShowClenchPending(true);
        setTimeout(() => setShowClenchPending(false), 2000);
      }),

      bciSocket.on('sentence_auto_sent', (e: BCIEvent) => {
        const text = e.data.sentence_text || '';
        setSelectedSlots({});
        setGrammarStep('subject');
        setGrammarStepIndex(0);
        if (text) {
          setIsSpeaking(true);
          setLastEvent('Speaking...');
          speakText(text).then(async (audioDataUrl) => {
            if (audioDataUrl) {
              const audio = new Audio(audioDataUrl);
              audioRef.current = audio;
              audio.onended = () => {
                setIsSpeaking(false);
                audioRef.current = null;
                setLastEvent(`Spoke: "${text}"`);
              };
              try {
                await audio.play();
              } catch {
                audioRef.current = null;
                await speakWithWebSpeech(text);
                setIsSpeaking(false);
                setLastEvent(`Spoke: "${text}"`);
              }
            } else {
              setIsSpeaking(false);
              setLastEvent(`Spoke: "${text}"`);
            }
          }).catch(() => {
            setIsSpeaking(false);
            setLastEvent(`Auto-sent: "${text}"`);
          });
        } else {
          setLastEvent('Auto-sent');
        }
      }),

      bciSocket.on('session_stopped', () => {
        setSelPhase('stopped');
        setHighlightIndex(null);
        setConfirmedIndex(null);
        setLastEvent('Session stopped');
      }),

      bciSocket.on('blink_detected', () => {
        setBlinkFlash(true);
        setTimeout(() => setBlinkFlash(false), 500);
      }),

      bciSocket.on('clench_detected', () => {
        setClenchFlash(true);
        setTimeout(() => setClenchFlash(false), 500);
      }),

      bciSocket.on('phrase_confirmed', (e: BCIEvent) => {
        if (e.data.selected_slots) setSelectedSlots(e.data.selected_slots);
      }),

      bciSocket.on('system_status', (e: BCIEvent) => {
        setStatus(e.data as any);
      }),

      bciSocket.on('warmup_status', (e: BCIEvent) => {
        const { state, progress, message } = e.data;
        if (state === 'warmup') {
          setSelPhase('warmup');
          setWarmupProgress(progress ?? 0);
          setLastEvent(message || 'Stabilizing...');
        } else if (state === 'idle') {
          setSelPhase('idle');
          setHighlightIndex(null);
          setConfirmedIndex(null);
          setLastEvent(message || '');
        }
      }),

      bciSocket.on('calibration_status', (e: BCIEvent) => {
        const { state, blinks_detected, blinks_needed } = e.data;
        if (state === 'calibrating') {
          setSelPhase('calibrating');
          setCalBlinks(blinks_detected ?? 0);
          setCalNeeded(blinks_needed ?? 2);
          setLastEvent(`Blink calibration: ${blinks_detected}/${blinks_needed}`);
        } else if (state === 'complete') {
          setLastEvent('Blink calibration complete');
        }
      }),

      bciSocket.on('clench_calibration_status', (e: BCIEvent) => {
        const { state, clenches_detected, clenches_needed } = e.data;
        if (state === 'calibrating') {
          setSelPhase('clench_calibrating');
          setClenchCalCount(clenches_detected ?? 0);
          setClenchCalNeeded(clenches_needed ?? 3);
          setLastEvent(`Clench calibration: ${clenches_detected}/${clenches_needed}`);
        } else if (state === 'complete') {
          setLastEvent('Clench calibration complete');
        }
      }),

      bciSocket.on('highlight_changed', (e: BCIEvent) => {
        setSelPhase('highlighting');
        setHighlightIndex(e.data.index);
        setConfirmedIndex(null);
      }),

      bciSocket.on('selection_confirmed', (e: BCIEvent) => {
        setSelPhase('confirming');
        setConfirmedIndex(e.data.index);
        setConfirmedPhrase(e.data.phrase || '');
        setHighlightIndex(null);
        setLastEvent(`Selected: "${e.data.phrase}"`);
      }),

      bciSocket.on('selection_executed', (e: BCIEvent) => {
        if (e.data.action === 'done_send' || e.data.action === 'auto_send') {
          setLastEvent(`Spoke: "${e.data.phrase}"`);
        } else {
          setLastEvent(`Confirmed: "${e.data.phrase}"`);
        }
        setConfirmedIndex(null);
        setHighlightIndex(null);
      }),
    );

    return () => unsubs.forEach((u) => u());
  }, [playDeleteSound]);

  const handleStartStop = useCallback(async () => {
    if (selPhase === 'idle' || selPhase === 'stopped') {
      try {
        setSelPhase('idle');
        setSelectedSlots({});
        setGrammarStep('subject');
        setGrammarStepIndex(0);
        await startSelection();
      } catch (e: any) {
        setLastEvent(`Error: ${e.message}`);
      }
    } else {
      try {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current = null;
        }
        await stopSelection();
        setSelPhase('stopped');
        setHighlightIndex(null);
        setConfirmedIndex(null);
      } catch (e: any) {
        setLastEvent(`Error: ${e.message}`);
      }
    }
  }, [selPhase]);

  const handleDoneSend = async () => {
    if (Object.keys(selectedSlots).length === 0) return;
    setIsSpeaking(true);
    setLastEvent('Speaking...');

    try {
      const result = await doneSend();
      const text = result.sentence || '';
      if (text) {
        const audioDataUrl = await speakText(text);
        if (audioDataUrl) {
          const audio = new Audio(audioDataUrl);
          audioRef.current = audio;
          audio.onended = () => {
            setIsSpeaking(false);
            audioRef.current = null;
          };
          try {
            await audio.play();
          } catch {
            audioRef.current = null;
            await speakWithWebSpeech(text);
            setIsSpeaking(false);
          }
        } else {
          setIsSpeaking(false);
        }
      } else {
        setIsSpeaking(false);
      }
      setSelectedSlots({});
      setGrammarStep('subject');
      setGrammarStepIndex(0);
    } catch {
      setIsSpeaking(false);
    }
  };

  const handleClear = async () => {
    try {
      await clearSentenceAPI();
      setSelectedSlots({});
      setGrammarStep('subject');
      setGrammarStepIndex(0);
    } catch {
      setSelectedSlots({});
    }
  };

  const isActive = selPhase !== 'idle' && selPhase !== 'stopped';
  const isStopped = selPhase === 'stopped';
  const hasSelections = Object.keys(selectedSlots).length > 0;
  const currentStepConfig = GRAMMAR_STEPS[grammarStepIndex] || GRAMMAR_STEPS[0];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full max-w-7xl mx-auto p-4">
      {/* Left Column: EEG + Band Power */}
      <div className="flex flex-col gap-4 order-1 lg:order-none">
        <div className="bg-white/40 backdrop-blur-md p-2 rounded-2xl border border-white/50 shadow-sm">
          <div className="bg-white/60 rounded-xl p-1">
            <EEGMonitor isFlashing={selPhase === 'highlighting'} />
          </div>
        </div>
        <div className="bg-white/40 backdrop-blur-md p-2 rounded-2xl border border-white/50 shadow-sm">
          <div className="bg-white/60 rounded-xl p-1">
            <BandPowerHistogram />
          </div>
        </div>
      </div>

      {/* Right Column: Grammar-First Word Selection */}
      <div className="flex flex-col gap-3 order-0 lg:order-none">
        {/* Status Bar */}
        <div className="flex items-center justify-between bg-white/30 backdrop-blur-md rounded-full px-4 py-2 border border-white/40">
          <div className="flex items-center gap-3">
            <div className={`flex items-center gap-1.5 text-xs font-bold ${wsConnected ? 'text-emerald-600' : 'text-red-500'}`}>
              {wsConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
              {wsConnected ? 'Connected' : 'Offline'}
            </div>
            {status?.simulate_mode && (
              <span className="text-[10px] font-bold bg-amber-200 text-amber-800 px-2 py-0.5 rounded-full">SIM</span>
            )}
            {status?.eegnet_gesture && (
              <span className="text-[10px] font-bold bg-blue-200 text-blue-800 px-2 py-0.5 rounded-full">GESTURE-DL</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold transition-all duration-200 ${blinkFlash ? 'bg-blue-400 text-white scale-110' : 'bg-blue-100 text-blue-500'}`}>
              <Eye size={10} /> BLINK
            </div>
            <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold transition-all duration-200 ${clenchFlash ? 'bg-orange-400 text-white scale-110' : 'bg-orange-100 text-orange-500'}`}>
              <Zap size={10} /> CLENCH
            </div>
          </div>
        </div>

        {/* Grammar Step Progress */}
        <div className="flex items-center gap-1 bg-white/30 backdrop-blur-md rounded-xl px-3 py-2 border border-white/40">
          {GRAMMAR_STEPS.map((step, i) => {
            const isCompleted = !!selectedSlots[step.key];
            const isCurrent = i === grammarStepIndex;
            const isSkipped = !selectedSlots[step.key] && i < grammarStepIndex;

            return (
              <React.Fragment key={step.key}>
                {i > 0 && (
                  <ChevronRight size={14} className="text-white/40 flex-shrink-0" />
                )}
                <div
                  className={`
                    flex-1 text-center px-2 py-1.5 rounded-lg text-xs font-bold transition-all duration-300
                    ${isCurrent
                      ? 'bg-white/90 text-zinc-800 shadow-sm ring-2 ring-white/60'
                      : isCompleted
                      ? 'bg-emerald-500/80 text-white'
                      : isSkipped
                      ? 'bg-white/20 text-white/50 line-through'
                      : 'bg-white/10 text-white/50'
                    }
                  `}
                >
                  <div className="flex items-center justify-center gap-1">
                    {isCompleted && <Check size={10} />}
                    <span className="truncate">{step.label}</span>
                  </div>
                  {isCompleted && (
                    <div className="text-[9px] font-medium opacity-80 truncate mt-0.5">
                      {selectedSlots[step.key]}
                    </div>
                  )}
                </div>
              </React.Fragment>
            );
          })}
        </div>

        {/* Start / Stop Button */}
        <div className="flex justify-center py-1">
          <button
            onClick={handleStartStop}
            className={`
              relative group overflow-hidden px-10 py-3 rounded-full font-black text-lg text-white shadow-xl transition-all transform hover:scale-105 active:scale-95
              ${isActive
                ? 'bg-gradient-to-b from-red-400 to-red-600 shadow-red-500/40 border-2 border-red-300'
                : 'bg-gradient-to-b from-emerald-400 to-emerald-600 shadow-emerald-500/40 border-2 border-emerald-300'
              }
            `}
          >
            <div className="absolute inset-0 bg-white/20 group-hover:bg-white/30 transition-colors" />
            <div className="absolute top-0 left-0 w-full h-1/2 bg-white/20 rounded-t-full blur-[1px]" />
            <div className="flex items-center gap-3 relative z-10 drop-shadow-md">
              {isActive
                ? <><Square size={24} fill="currentColor" /><span className="tracking-wide">STOP</span></>
                : <><Play size={24} fill="currentColor" /><span className="tracking-wide">START</span></>
              }
            </div>
          </button>
        </div>

        {/* Stopped Overlay */}
        <AnimatePresence>
          {isStopped && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-zinc-100 border-2 border-zinc-400 rounded-2xl p-5 text-center shadow-lg"
            >
              <div className="flex items-center justify-center gap-2 text-zinc-600 font-bold text-lg">
                <StopCircle size={24} className="text-zinc-500" />
                Session Stopped
              </div>
              <p className="text-zinc-500 text-sm mt-1">Press START to begin a new session</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Warmup Overlay */}
        <AnimatePresence>
          {selPhase === 'warmup' && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-amber-50 border-2 border-amber-300 rounded-2xl p-5 text-center shadow-lg"
            >
              <p className="text-amber-800 font-bold text-lg mb-3">Stabilizing signal...</p>
              <p className="text-amber-600 text-sm mb-3">Stay still and relax</p>
              <div className="w-full bg-amber-200 rounded-full h-3 overflow-hidden">
                <motion.div
                  className="h-full bg-amber-500 rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: `${warmupProgress * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Blink Calibration Overlay */}
        <AnimatePresence>
          {selPhase === 'calibrating' && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-blue-50 border-2 border-blue-300 rounded-2xl p-5 text-center shadow-lg"
            >
              <p className="text-blue-800 font-bold text-lg mb-2">Blink to calibrate</p>
              <p className="text-blue-600 text-sm mb-4">Perform {calNeeded} intentional blinks</p>
              <div className="flex justify-center gap-3">
                {Array.from({ length: calNeeded }).map((_, i) => (
                  <motion.div
                    key={i}
                    animate={i < calBlinks ? { scale: [1, 1.3, 1] } : {}}
                    transition={{ duration: 0.3 }}
                    className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold border-2 transition-all ${
                      i < calBlinks
                        ? 'bg-blue-500 text-white border-blue-600 shadow-lg shadow-blue-500/40'
                        : 'bg-white text-blue-300 border-blue-200'
                    }`}
                  >
                    {i < calBlinks ? <Check size={24} /> : <Eye size={24} />}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Clench Calibration Overlay */}
        <AnimatePresence>
          {selPhase === 'clench_calibrating' && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-orange-50 border-2 border-orange-300 rounded-2xl p-5 text-center shadow-lg"
            >
              <p className="text-orange-800 font-bold text-lg mb-2">Clench your jaw to calibrate</p>
              <p className="text-orange-600 text-sm mb-4">Perform {clenchCalNeeded} firm jaw clenches</p>
              <div className="flex justify-center gap-3">
                {Array.from({ length: clenchCalNeeded }).map((_, i) => (
                  <motion.div
                    key={i}
                    animate={i < clenchCalCount ? { scale: [1, 1.3, 1] } : {}}
                    transition={{ duration: 0.3 }}
                    className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold border-2 transition-all ${
                      i < clenchCalCount
                        ? 'bg-orange-500 text-white border-orange-600 shadow-lg shadow-orange-500/40'
                        : 'bg-white text-orange-300 border-orange-200'
                    }`}
                  >
                    {i < clenchCalCount ? <Check size={24} /> : <Zap size={24} />}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Confirmation Banner */}
        <AnimatePresence>
          {selPhase === 'confirming' && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-emerald-50 border-2 border-emerald-400 rounded-2xl p-4 text-center shadow-lg"
            >
              <div className="flex items-center justify-center gap-2 text-emerald-700 font-bold text-lg">
                <Check size={24} className="text-emerald-500" />
                {confirmedPhrase === OTHER_LABEL
                  ? 'Loading more words...'
                  : confirmedPhrase === SKIP_LABEL
                  ? `Skipping ${currentStepConfig.label}...`
                  : `Selected: "${confirmedPhrase}"`}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Clench Pending Toast */}
        <AnimatePresence>
          {showClenchPending && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-yellow-500 text-black rounded-xl px-4 py-2 text-center font-bold shadow-lg"
            >
              <div className="flex items-center justify-center gap-2">
                <Trash2 size={16} />
                Clench again to clear sentence
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Delete Toast */}
        <AnimatePresence>
          {showDeleteToast && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-red-500 text-white rounded-xl px-4 py-2 text-center font-bold shadow-lg"
            >
              <div className="flex items-center justify-center gap-2">
                <Trash2 size={16} />
                Sentence Deleted (Double Clench)
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Current Step Label */}
        <div className="text-center">
          <span className="text-sm font-bold text-white/80 bg-white/20 backdrop-blur-sm rounded-full px-4 py-1">
            Choose a {currentStepConfig.label}
            {isSkippable && <span className="text-white/50 ml-1">(or skip)</span>}
          </span>
        </div>

        {/* Word Grid */}
        <div className="relative p-1 rounded-3xl bg-gradient-to-br from-blue-400 via-purple-400 to-pink-400 shadow-xl">
          <div className="absolute inset-0 bg-white/40 backdrop-blur-xl rounded-3xl m-[2px]"></div>
          <div className="relative grid grid-cols-2 md:grid-cols-3 gap-3 p-3 min-h-[200px]">
            {phrases.map((phrase, index) => {
              const isHighlighted = highlightIndex === index && selPhase === 'highlighting';
              const isConfirmed = confirmedIndex === index && selPhase === 'confirming';
              const isOther = phrase === OTHER_LABEL;
              const isSkip = phrase === SKIP_LABEL;

              return (
                <motion.button
                  key={`${phrase}-${index}`}
                  disabled={isActive && selPhase !== 'idle'}
                  animate={{
                    scale: isConfirmed ? 1.10 : isHighlighted ? 1.08 : 1,
                  }}
                  transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                  className={`
                    relative overflow-hidden rounded-xl font-bold
                    flex items-center justify-center p-3 h-20 md:h-24
                    group shadow-sm border transition-colors duration-200
                    ${isConfirmed
                      ? 'bg-emerald-500 text-white border-emerald-600 ring-4 ring-emerald-400 shadow-xl shadow-emerald-500/50 z-30'
                      : isHighlighted
                      ? 'bg-amber-400 text-white border-amber-500 ring-4 ring-amber-300 shadow-xl shadow-amber-400/50 z-20'
                      : isStopped
                      ? 'bg-zinc-200 text-zinc-400 border-zinc-300 cursor-not-allowed'
                      : isSkip
                      ? 'bg-zinc-100 text-zinc-500 border-2 border-dashed border-zinc-400 hover:bg-zinc-200'
                      : isOther
                      ? 'bg-slate-100 text-slate-600 border-2 border-dashed border-slate-400 hover:bg-slate-200'
                      : isActive
                      ? 'bg-white/60 text-zinc-400 border-white/40'
                      : 'bg-white/80 text-zinc-600 border-white/60 hover:bg-white hover:scale-[1.02]'
                    }
                  `}
                >
                  <span className={`relative z-10 text-center leading-tight ${(isOther || isSkip) ? 'text-sm' : 'text-lg'}`}>
                    {isOther ? (
                      <span className="flex items-center gap-1.5">
                        <MoreHorizontal size={18} />
                        More
                      </span>
                    ) : isSkip ? (
                      <span className="flex items-center gap-1.5">
                        <SkipForward size={18} />
                        Skip
                      </span>
                    ) : (
                      phrase
                    )}
                  </span>

                  {isConfirmed && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="absolute top-2 right-2 bg-white/30 rounded-full p-1"
                    >
                      <Check size={14} />
                    </motion.div>
                  )}

                  <span className={`absolute top-1.5 left-2.5 text-[10px] font-mono ${isHighlighted || isConfirmed ? 'opacity-60' : 'opacity-30'}`}>
                    {index + 1}
                  </span>

                  {isHighlighted && (
                    <motion.div
                      className="absolute bottom-0 left-0 right-0 h-1 bg-white/60"
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ duration: 2, ease: 'linear' }}
                      style={{ transformOrigin: 'left' }}
                    />
                  )}
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Last Event */}
        {lastEvent && (
          <div className="text-center text-xs font-medium text-white/80 bg-white/10 backdrop-blur-sm rounded-full px-4 py-1.5 mx-auto">
            {lastEvent}
          </div>
        )}

        {/* Sentence Slots */}
        <div className="bg-white/20 backdrop-blur-xl rounded-2xl p-3 shadow-lg border border-white/40 ring-1 ring-white/20 w-full">
          <div className="grid grid-cols-4 gap-2">
            {GRAMMAR_STEPS.map((step) => {
              const word = selectedSlots[step.key];
              const isCurrent = step.key === grammarStep;
              return (
                <div
                  key={step.key}
                  className={`
                    rounded-lg px-2 py-2 text-center transition-all min-h-[52px] flex flex-col items-center justify-center
                    ${word
                      ? 'bg-emerald-100 border border-emerald-300'
                      : isCurrent
                      ? 'bg-white/60 border-2 border-dashed border-blue-400 animate-pulse'
                      : 'bg-white/20 border border-white/30'
                    }
                  `}
                >
                  <span className="text-[9px] font-bold uppercase text-zinc-400 leading-none">{step.label}</span>
                  <span className={`text-sm font-bold leading-tight mt-0.5 truncate w-full ${word ? 'text-zinc-800' : 'text-zinc-300'}`}>
                    {word || '---'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white/40 backdrop-blur-md p-3 rounded-3xl border border-white/50 shadow-sm space-y-2">
          <button
            onClick={handleDoneSend}
            disabled={!hasSelections || isLoading || isSpeaking}
            className="w-full group relative overflow-hidden bg-gradient-to-r from-emerald-400 to-teal-500 hover:from-emerald-500 hover:to-teal-600 text-white py-3 rounded-xl font-bold shadow-lg shadow-emerald-500/20 active:translate-y-0.5 transition-all disabled:opacity-50"
          >
            <div className="flex items-center justify-center gap-2">
              {isSpeaking ? <Volume2 className="animate-pulse" size={20} /> : isLoading ? <RefreshCw className="animate-spin" size={20} /> : <Send size={20} />}
              <span className="text-base">{isSpeaking ? 'Speaking...' : 'Done / Send'}</span>
            </div>
          </button>
          <button
            onClick={handleClear}
            disabled={!hasSelections}
            className="w-full bg-white hover:bg-zinc-50 text-zinc-700 py-2.5 rounded-xl font-bold shadow-sm border border-zinc-200/50 flex items-center justify-center gap-2 active:translate-y-0.5 transition-all disabled:opacity-50"
          >
            <RefreshCw size={16} />
            <span>Clear All</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default P300Grid;
