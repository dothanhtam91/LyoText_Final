"""Hacklytic BCI — Main entry point.

Launches:
1. FastAPI server (with WebSocket) on the main thread via Uvicorn
2. Muse 2 EEG streaming (background thread)
3. Signal processing pipeline (background thread)
4. Pygame stimulus window (separate process)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
import time
from enum import Enum

import numpy as np
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from api.server import create_app
from database.store import redis_store
from eeg.artifacts import artifact_detector
from eeg.classifier import p300_classifier
from eeg.deep_trainer import deep_trainer
from eeg.processing import signal_processor
from eeg.stream import eeg_stream
from llm.phrase_engine import OTHER_LABEL, SKIP_LABEL, phrase_engine
from stimulus.flasher import StimulusFlasher
from utils.events import Event, EventType, event_bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("hacklytic")


class SelectionState(str, Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    CALIBRATING = "calibrating"
    CLENCH_CALIBRATING = "clench_calibrating"
    HIGHLIGHTING = "highlighting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"


HIGHLIGHT_DURATION = 2.0
WARMUP_DURATION = 2.0
CALIBRATION_BLINKS = 2
CALIBRATION_CLENCHES = 3
CLENCH_COOLDOWN_SEC = 1.0
CLENCH_IDLE_GUARD_SEC = 0.3
DOUBLE_CLENCH_WINDOW_SEC = 2.0
DEBOUNCE_PERIOD = 1.5
CONFIRM_DISPLAY_SEC = 1.5
DEFAULT_BLINK_THRESHOLD = 100.0


class BCIOrchestrator:

    def __init__(self) -> None:
        self._sample_queue: queue.Queue = queue.Queue(maxsize=1024)
        self._flasher = StimulusFlasher()
        self._running = False
        self._processing_thread: threading.Thread | None = None
        self._stimulus_thread: threading.Thread | None = None
        self._current_phrases: list[str] = []
        self._cycle_epochs: list[tuple] = []
        self._flash_active = False
        self._last_p300_selection: tuple[int, str] | None = None
        self._live_test_active = False
        self._gesture_vote_buffer: list[tuple[int, str, float]] = []

        # EEGNet-based gesture detection results
        self._eegnet_blink_time = 0.0
        self._eegnet_blink_conf = 0.0
        self._eegnet_clench_time = 0.0
        self._eegnet_clench_conf = 0.0
        self._last_voted_gesture = "idle"
        self._last_gesture_fire_time = 0.0

        # Selection state machine
        self._sel_state = SelectionState.IDLE
        self._sel_warmup_start = 0.0
        self._sel_last_progress_emit = 0.0
        self._sel_cal_blinks = 0
        self._sel_cal_amplitudes: list[float] = []
        self._sel_blink_threshold = DEFAULT_BLINK_THRESHOLD
        self._sel_highlight_index = 0
        self._sel_highlight_start = 0.0
        self._sel_last_blink_time = 0.0
        self._sel_confirmed_index = -1
        self._sel_confirm_start = 0.0

        # Clench gating & calibration
        self._last_clench_action_time = 0.0
        self._clench_pending_time = 0.0
        self._clench_cal_count = 0
        self._clench_cal_rms_values: list[float] = []
        self._clench_cal_last_time = 0.0
        self._dynamic_clench_threshold: float | None = None

    def start(self) -> None:
        self._running = True

        if p300_classifier.load():
            logger.info("P300 LDA model loaded from disk")
        else:
            logger.info("No saved LDA model — calibration will be required")

        dl_status = deep_trainer.load_models()
        if dl_status.get("p300"):
            logger.info("EEGNet P300 model loaded — will be used for classification")
        if dl_status.get("gesture"):
            logger.info("EEGNet gesture model loaded — ML gesture detection active")

        eeg_stream.start(sample_buffer=self._sample_queue)
        logger.info("EEG stream started (simulate=%s)", config.SIMULATE_EEG)

        self._flasher.start()
        logger.info("Pygame stimulus window launched")

        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()

        self._stimulus_thread = threading.Thread(
            target=self._stimulus_event_loop, daemon=True
        )
        self._stimulus_thread.start()

        event_bus.on(EventType.BLINK_DETECTED, self._on_blink)
        event_bus.on(EventType.CLENCH_DETECTED, self._on_clench)

        threading.Thread(target=self._async_init, daemon=True).start()

    def _async_init(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._load_initial_phrases())
        except Exception:
            logger.exception("Async init failed")
        finally:
            loop.close()

    async def _load_initial_phrases(self) -> None:
        try:
            words = phrase_engine.get_words_for_step()
            labels = [OTHER_LABEL]
            if phrase_engine.is_skippable:
                labels.insert(0, SKIP_LABEL)
            self._current_phrases = words + labels
            self._flasher.set_phrases(self._current_phrases)
            self._emit_grammar_update(words)
            logger.info("Initial words loaded: %s", self._current_phrases)

            if p300_classifier.is_trained:
                time.sleep(1.0)
                self._start_flash_cycle()
        except Exception:
            logger.exception("Failed to load initial words")

    def _emit_grammar_update(self, words: list[str]) -> None:
        """Emit grammar-aware WORDS_UPDATED and GRAMMAR_STEP_CHANGED events."""
        event_bus.emit(Event(
            type=EventType.WORDS_UPDATED,
            data={
                "words": words,
                "phrases": self._current_phrases,
                "sentence": phrase_engine.sentence,
                "grammar_step": phrase_engine.current_step_name,
                "grammar_step_index": phrase_engine.step_index,
                "skippable": phrase_engine.is_skippable,
                "selected_slots": phrase_engine.selected_slots,
            },
        ))
        event_bus.emit(Event(
            type=EventType.GRAMMAR_STEP_CHANGED,
            data={
                "step": phrase_engine.current_step_name,
                "step_index": phrase_engine.step_index,
                "skippable": phrase_engine.is_skippable,
                "selected_slots": phrase_engine.selected_slots,
            },
        ))

    def stop(self) -> None:
        self._running = False
        eeg_stream.stop()
        self._flasher.stop()
        if self._processing_thread:
            self._processing_thread.join(timeout=3)
        if self._stimulus_thread:
            self._stimulus_thread.join(timeout=3)

    @property
    def live_test_active(self) -> bool:
        return self._live_test_active

    @live_test_active.setter
    def live_test_active(self, value: bool) -> None:
        self._live_test_active = value
        self._gesture_vote_buffer.clear()
        if value:
            print("\n" + "=" * 58)
            print("  🧠  LIVE GESTURE TEST — 0.25s interval, 9-window majority vote")
            print("=" * 58)
            print("  ICON  CLASS    │    CONFIDENCE BAR   │  CONF%  │ VOTE")
            print("  " + "─" * 54)
        else:
            print("  " + "─" * 54)
            print("  Live test stopped.")
            print("=" * 50 + "\n")
        logger.info("Live test mode %s", "ENABLED" if value else "DISABLED")

    def _processing_loop(self) -> None:
        last_gesture_check = 0.0
        last_selection_tick = 0.0

        while self._running:
            try:
                samples, timestamps = self._sample_queue.get(timeout=0.05)
            except queue.Empty:
                now = time.time()
                if self._sel_state != SelectionState.IDLE and now - last_selection_tick >= 0.1:
                    last_selection_tick = now
                    self._run_selection_tick()
                continue

            filtered = signal_processor.process_chunk(samples, timestamps)
            artifact_detector.process_samples(filtered, timestamps)

            epochs = signal_processor.try_extract_epochs()
            for epoch, phrase_idx in epochs:
                if not p300_classifier.is_calibrating:
                    self._cycle_epochs.append((epoch, phrase_idx))

            now = time.time()

            if self._sel_state != SelectionState.IDLE and now - last_selection_tick >= 0.1:
                last_selection_tick = now
                self._run_selection_tick()

            interval = 0.25
            if (
                deep_trainer.gesture_model is not None
                and (self._live_test_active or not self._flash_active)
                and (self._live_test_active or self._sel_state != SelectionState.IDLE)
                and now - last_gesture_check >= interval
            ):
                last_gesture_check = now
                self._run_gesture_classification()

    def _stimulus_event_loop(self) -> None:
        while self._running:
            try:
                ev = self._flasher.event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            ev_type = ev.get("event")

            if ev_type == "stimulus_onset":
                event_bus.emit(Event(
                    type=EventType.STIMULUS_ONSET,
                    data={"phrase_index": ev["phrase_index"]},
                    timestamp=ev["timestamp"],
                ))
            elif ev_type == "flash_cycle_complete":
                self._flash_active = False
                artifact_detector.set_flash_active(False)
                self._on_flash_cycle_complete()

    def _start_flash_cycle(self) -> None:
        self._cycle_epochs.clear()
        self._flash_active = True
        self._last_p300_selection = None
        artifact_detector.set_flash_active(True)
        self._flasher.start_flash()
        logger.info("Flash cycle started")

    def _on_flash_cycle_complete(self) -> None:
        use_deep = deep_trainer.p300_model is not None
        if not use_deep and not p300_classifier.is_trained:
            logger.warning("Flash complete but no classifier available")
            return

        if not self._cycle_epochs:
            logger.warning("No epochs collected during flash cycle")
            self._schedule_next_cycle(delay=1.0)
            return

        try:
            if use_deep:
                winner_idx, confidence = deep_trainer.select_phrase_deep(
                    self._cycle_epochs, config.NUM_PHRASES
                )
                logger.info("Using EEGNet P300 for classification")
            else:
                winner_idx, confidence = p300_classifier.select_phrase(
                    self._cycle_epochs, config.NUM_PHRASES
                )
            winner_phrase = (
                self._current_phrases[winner_idx]
                if winner_idx < len(self._current_phrases)
                else "?"
            )

            self._last_p300_selection = (winner_idx, winner_phrase)
            logger.info(
                "P300 result: phrase[%d]='%s' (confidence=%.2f)",
                winner_idx, winner_phrase, confidence,
            )

            self._flasher.highlight(winner_idx, "green")

            event_bus.emit(Event(
                type=EventType.P300_RESULT,
                data={
                    "selected_index": winner_idx,
                    "confidence": confidence,
                    "phrase": winner_phrase,
                },
            ))

            try:
                redis_store.push_event("p300_result", {
                    "selected_index": winner_idx,
                    "confidence": confidence,
                    "phrase": winner_phrase,
                })
            except Exception:
                logger.debug("Redis push failed (non-critical)")

        except Exception:
            logger.exception("Classification failed")
            self._schedule_next_cycle(delay=1.0)

    def _on_blink(self, event: Event) -> None:
        if self._flash_active:
            return
        if self._last_p300_selection is None:
            return

        idx, phrase = self._last_p300_selection
        self._last_p300_selection = None

        phrase_engine.confirm_phrase(phrase)
        logger.info("Phrase confirmed by blink: '%s'", phrase)

        event_bus.emit(Event(
            type=EventType.PHRASE_CONFIRMED,
            data={"phrase": phrase, "history": phrase_engine.history},
        ))

        try:
            redis_store.push_event("phrase_confirmed", {"phrase": phrase})
        except Exception:
            pass

        self._flasher.reset_highlight()
        self._refresh_phrases_and_flash()

    def _on_clench(self, event: Event) -> None:
        if self._flash_active:
            return

        now = time.time()
        if now - self._last_clench_action_time < CLENCH_COOLDOWN_SEC:
            return
        if now - self._sel_last_blink_time < CLENCH_IDLE_GUARD_SEC:
            return

        if not phrase_engine.has_selections():
            self._last_clench_action_time = now
            self._last_p300_selection = None
            self._flasher.reset_highlight()
            return

        if self._clench_pending_time > 0 and (now - self._clench_pending_time) < DOUBLE_CLENCH_WINDOW_SEC:
            self._clench_pending_time = 0.0
            self._last_clench_action_time = now
            logger.info("Sentence cleared by double jaw clench")
            phrase_engine.clear_sentence()
            self._last_p300_selection = None
            event_bus.emit(Event(
                type=EventType.SENTENCE_CLEARED,
                data={"spoken": ""},
            ))
            try:
                redis_store.push_event("sentence_cleared", {"action": "double_clench"})
            except Exception:
                pass
            self._flasher.reset_highlight()
            self._refresh_grammar_words_and_resume()
        else:
            self._clench_pending_time = now
            self._last_clench_action_time = now
            logger.info("First clench detected — waiting for second clench to confirm delete")
            event_bus.emit(Event(
                type=EventType.CLENCH_PENDING,
                data={"message": "Clench again to clear"},
            ))

    def _refresh_phrases_and_flash(self) -> None:
        def _do():
            try:
                words = phrase_engine.get_words_for_step()
                labels = [OTHER_LABEL]
                if phrase_engine.is_skippable:
                    labels.insert(0, SKIP_LABEL)
                self._current_phrases = words + labels
                self._flasher.set_phrases(self._current_phrases)
                self._emit_grammar_update(words)
                time.sleep(0.5)
                self._start_flash_cycle()
            except Exception:
                logger.exception("Failed to refresh phrases")

        threading.Thread(target=_do, daemon=True).start()

    def _get_raw_window(self) -> "np.ndarray | None":
        """Grab a 1-second raw EEG window from Redis, return (4, 256) or None."""
        import numpy as _np
        raw_samples = redis_store.get_recent_raw(seconds=1.0)
        if len(raw_samples) < config.EEG_SAMPLE_RATE * 0.8:
            return None
        data = _np.array([
            [s["tp9"], s["af7"], s["af8"], s["tp10"]]
            for s in raw_samples
        ], dtype=_np.float32)
        if len(data) > config.EEG_SAMPLE_RATE:
            data = data[-config.EEG_SAMPLE_RATE:]
        elif len(data) < config.EEG_SAMPLE_RATE:
            import numpy as _np2
            pad = config.EEG_SAMPLE_RATE - len(data)
            data = _np2.pad(data, ((pad, 0), (0, 0)), mode="edge")
        return data.T

    _CONFIDENCE_THRESHOLD = 0.80
    _VOTE_WINDOW = 9
    _SUPERMAJORITY = 7          # need 7/9 votes for a gesture to win
    _GESTURE_REFRACTORY_SEC = 3.0  # cooldown before the same gesture can fire again

    def _majority_vote(self) -> tuple[str, float]:
        """Return the majority class and its average confidence, requiring a supermajority."""
        from collections import Counter
        if not self._gesture_vote_buffer:
            return "idle", 0.0
        names = [name for _, name, _ in self._gesture_vote_buffer]
        counts = Counter(names)
        winner, count = counts.most_common(1)[0]
        if count < self._SUPERMAJORITY:
            return "idle", 0.0
        winner_confs = [c for _, n, c in self._gesture_vote_buffer if n == winner]
        return winner, sum(winner_confs) / len(winner_confs)

    def _run_gesture_classification(self) -> None:
        """Use the trained gesture model to detect blinks/clenches from raw EEG.

        Guards against false positives with four layers:
        1. Per-frame confidence threshold — low-confidence frames become "idle"
        2. Full vote window required — must accumulate 9 consecutive predictions
        3. Supermajority — 7/9 windows must agree on the gesture class
        4. Transition-based firing — only fires on idle→gesture state change,
           NOT on sustained predictions (prevents auto-repeat from model bias)
        """
        try:
            window = self._get_raw_window()
            if window is None:
                return

            cls_idx, cls_name, confidence = deep_trainer.predict_gesture(window)

            effective_name = cls_name if confidence >= self._CONFIDENCE_THRESHOLD else "idle"
            effective_conf = confidence if confidence >= self._CONFIDENCE_THRESHOLD else 0.0

            self._gesture_vote_buffer.append((cls_idx, effective_name, effective_conf))
            if len(self._gesture_vote_buffer) > self._VOTE_WINDOW:
                self._gesture_vote_buffer.pop(0)

            voted_name, voted_conf = self._majority_vote()

            if self._live_test_active:
                conf_pct = confidence * 100
                bar = "█" * int(conf_pct / 5) + "░" * (20 - int(conf_pct / 5))
                icon = {"idle": "😐", "blink": "👁️", "clench": "😬"}.get(cls_name, "❓")
                voted_icon = {"idle": "😐", "blink": "👁️", "clench": "😬"}.get(voted_name, "❓")
                print(
                    f"  {icon}  {cls_name:<8} │{bar}│ {conf_pct:5.1f}%  │ {voted_icon} {voted_name}",
                    flush=True,
                )
                event_bus.emit(Event(
                    type=EventType.GESTURE_PREDICTION,
                    data={
                        "class_index": cls_idx,
                        "class_name": voted_name,
                        "confidence": round(voted_conf, 4),
                        "raw_class": cls_name,
                        "raw_confidence": round(confidence, 4),
                    },
                ))

            prev_gesture = self._last_voted_gesture
            self._last_voted_gesture = voted_name

            if voted_name == "idle" or voted_conf < self._CONFIDENCE_THRESHOLD:
                return

            if len(self._gesture_vote_buffer) < self._VOTE_WINDOW:
                return

            # Only fire on a STATE TRANSITION (idle/other → this gesture).
            # If the model keeps predicting "clench" every cycle, it fires
            # exactly once on the first transition, then stops until the
            # user releases (idle) and clenches again.
            if prev_gesture == voted_name:
                return

            now = time.time()
            if now - self._last_gesture_fire_time < self._GESTURE_REFRACTORY_SEC:
                return

            self._last_gesture_fire_time = now
            logger.info("Gesture detected: %s (voted_conf=%.2f)", voted_name, voted_conf)

            if voted_name == "blink":
                self._eegnet_blink_time = now
                self._eegnet_blink_conf = voted_conf
                event_bus.emit(Event(
                    type=EventType.BLINK_DETECTED,
                    data={"peak_uv": 0, "channel": "ML", "confidence": voted_conf},
                ))
            elif voted_name == "clench":
                self._eegnet_clench_time = now
                self._eegnet_clench_conf = voted_conf
                event_bus.emit(Event(
                    type=EventType.CLENCH_DETECTED,
                    data={"rms": 0, "confidence": voted_conf},
                ))
        except Exception:
            logger.debug("Gesture classification failed", exc_info=True)

    # ── Selection state machine ────────────────────────────────

    def start_selection(self) -> dict:
        """Begin warmup → blink cal → clench cal → highlighting cycle."""
        if self._sel_state != SelectionState.IDLE:
            return {"error": "Selection already active"}
        if not self._current_phrases:
            words = phrase_engine.get_words_for_step()
            labels = [OTHER_LABEL]
            if phrase_engine.is_skippable:
                labels.insert(0, SKIP_LABEL)
            self._current_phrases = words + labels
            self._emit_grammar_update(words)

        phrase_engine.clear_sentence()
        self._sel_state = SelectionState.WARMUP
        self._sel_warmup_start = time.time()
        self._sel_last_progress_emit = 0.0
        self._sel_cal_blinks = 0
        self._sel_cal_amplitudes.clear()
        self._sel_last_blink_time = 0.0
        self._sel_blink_threshold = DEFAULT_BLINK_THRESHOLD
        self._gesture_vote_buffer.clear()
        self._clench_cal_count = 0
        self._clench_cal_rms_values.clear()
        self._last_clench_action_time = 0.0
        self._clench_pending_time = 0.0

        logger.info("Selection started — entering warmup phase")
        event_bus.emit(Event(
            type=EventType.WARMUP_STATUS,
            data={"state": "warmup", "message": "Stabilizing signal...", "progress": 0.0},
        ))
        return {"status": "started", "state": "warmup"}

    def stop_selection(self) -> dict:
        """Full session stop: halt everything, reset state."""
        prev = self._sel_state
        self._sel_state = SelectionState.IDLE
        self._sel_highlight_index = 0
        logger.info("Session stopped (was %s)", prev.value)
        event_bus.emit(Event(
            type=EventType.SESSION_STOPPED,
            data={"previous_state": prev.value, "message": "Session stopped"},
        ))
        event_bus.emit(Event(
            type=EventType.WARMUP_STATUS,
            data={"state": "idle", "message": "Session stopped", "progress": 0.0},
        ))
        return {"status": "stopped"}

    @property
    def selection_status(self) -> dict:
        return {
            "state": self._sel_state.value,
            "highlight_index": self._sel_highlight_index,
            "blink_threshold": self._sel_blink_threshold,
            "calibration_blinks": self._sel_cal_blinks,
            "phrases": self._current_phrases,
            "grammar_step": phrase_engine.current_step_name,
            "grammar_step_index": phrase_engine.step_index,
            "skippable": phrase_engine.is_skippable,
            "selected_slots": phrase_engine.selected_slots,
        }

    def _detect_raw_blink(self) -> tuple[bool, float]:
        """Detect blink from raw frontal EEG peak-to-peak amplitude (last 500ms)."""
        try:
            raw_samples = redis_store.get_recent_raw(seconds=0.5)
            if len(raw_samples) < 64:
                return False, 0.0
            data = np.array(
                [[s["af7"], s["af8"]] for s in raw_samples], dtype=np.float32,
            )
            pp_af7 = float(data[:, 0].max() - data[:, 0].min())
            pp_af8 = float(data[:, 1].max() - data[:, 1].min())
            max_pp = max(pp_af7, pp_af8)
            return max_pp > self._sel_blink_threshold, max_pp
        except Exception:
            return False, 0.0

    def _detect_blink(self) -> tuple[bool, float]:
        """Detect blink using EEGNet if available, then fall back to raw detection."""
        if deep_trainer.gesture_model is not None:
            now = time.time()
            age = now - self._eegnet_blink_time
            if age < 0.5 and self._eegnet_blink_conf >= self._CONFIDENCE_THRESHOLD:
                self._eegnet_blink_time = 0.0
                return True, self._eegnet_blink_conf * 200
        return self._detect_raw_blink()

    def _run_selection_tick(self) -> None:
        """Advance the selection state machine (called ~every 100ms)."""
        now = time.time()

        if self._sel_state == SelectionState.WARMUP:
            elapsed = now - self._sel_warmup_start
            progress = min(elapsed / WARMUP_DURATION, 1.0)

            if now - self._sel_last_progress_emit >= 0.3:
                self._sel_last_progress_emit = now
                event_bus.emit(Event(
                    type=EventType.WARMUP_STATUS,
                    data={
                        "state": "warmup",
                        "message": "Stabilizing signal...",
                        "progress": round(progress, 2),
                    },
                ))

            if elapsed >= WARMUP_DURATION:
                self._sel_state = SelectionState.CALIBRATING
                self._sel_cal_blinks = 0
                self._sel_cal_amplitudes.clear()
                self._sel_last_blink_time = 0.0
                logger.info("Warmup complete — entering calibration (blink %d times)", CALIBRATION_BLINKS)
                event_bus.emit(Event(
                    type=EventType.CALIBRATION_STATUS,
                    data={
                        "state": "calibrating",
                        "blinks_detected": 0,
                        "blinks_needed": CALIBRATION_BLINKS,
                        "threshold": self._sel_blink_threshold,
                    },
                ))

        elif self._sel_state == SelectionState.CALIBRATING:
            is_blink, amplitude = self._detect_blink()
            if is_blink and now - self._sel_last_blink_time > DEBOUNCE_PERIOD:
                self._sel_last_blink_time = now
                self._sel_cal_blinks += 1
                self._sel_cal_amplitudes.append(amplitude)
                logger.info(
                    "Calibration blink %d/%d (amplitude=%.1f uV)",
                    self._sel_cal_blinks, CALIBRATION_BLINKS, amplitude,
                )
                event_bus.emit(Event(
                    type=EventType.CALIBRATION_STATUS,
                    data={
                        "state": "calibrating",
                        "blinks_detected": self._sel_cal_blinks,
                        "blinks_needed": CALIBRATION_BLINKS,
                        "threshold": round(amplitude, 1),
                    },
                ))

                if self._sel_cal_blinks >= CALIBRATION_BLINKS:
                    self._sel_blink_threshold = min(self._sel_cal_amplitudes) * 0.7
                    logger.info(
                        "Blink calibration complete — threshold set to %.1f uV",
                        self._sel_blink_threshold,
                    )
                    event_bus.emit(Event(
                        type=EventType.CALIBRATION_STATUS,
                        data={
                            "state": "complete",
                            "blinks_detected": CALIBRATION_BLINKS,
                            "blinks_needed": CALIBRATION_BLINKS,
                            "threshold": round(self._sel_blink_threshold, 1),
                        },
                    ))
                    self._sel_state = SelectionState.CLENCH_CALIBRATING
                    self._clench_cal_count = 0
                    self._clench_cal_rms_values.clear()
                    self._clench_cal_last_time = now
                    logger.info("Entering clench calibration (clench %d times)", CALIBRATION_CLENCHES)
                    event_bus.emit(Event(
                        type=EventType.CLENCH_CALIBRATION_STATUS,
                        data={
                            "state": "calibrating",
                            "clenches_detected": 0,
                            "clenches_needed": CALIBRATION_CLENCHES,
                        },
                    ))

        elif self._sel_state == SelectionState.CLENCH_CALIBRATING:
            self._run_clench_calibration_tick(now)

        elif self._sel_state == SelectionState.HIGHLIGHTING:
            is_blink, amplitude = self._detect_blink()
            if is_blink and now - self._sel_last_blink_time > DEBOUNCE_PERIOD:
                self._sel_last_blink_time = now
                self._sel_confirmed_index = self._sel_highlight_index
                self._sel_confirm_start = now
                self._sel_state = SelectionState.CONFIRMING
                phrase = (
                    self._current_phrases[self._sel_confirmed_index]
                    if self._sel_confirmed_index < len(self._current_phrases)
                    else "?"
                )
                logger.info(
                    "Selection confirmed: [%d] '%s' (amplitude=%.1f uV)",
                    self._sel_confirmed_index, phrase, amplitude,
                )
                event_bus.emit(Event(
                    type=EventType.SELECTION_CONFIRMED,
                    data={
                        "index": self._sel_confirmed_index,
                        "phrase": phrase,
                        "confidence": round(amplitude / self._sel_blink_threshold, 2),
                    },
                ))
            elif now - self._sel_highlight_start >= HIGHLIGHT_DURATION:
                n = len(self._current_phrases) or 1
                self._sel_highlight_index = (self._sel_highlight_index + 1) % n
                self._sel_highlight_start = now
                phrase = (
                    self._current_phrases[self._sel_highlight_index]
                    if self._sel_highlight_index < len(self._current_phrases)
                    else ""
                )
                event_bus.emit(Event(
                    type=EventType.HIGHLIGHT_CHANGED,
                    data={
                        "index": self._sel_highlight_index,
                        "phrase": phrase,
                        "total": len(self._current_phrases),
                    },
                ))

        elif self._sel_state == SelectionState.CONFIRMING:
            if now - self._sel_confirm_start >= CONFIRM_DISPLAY_SEC:
                idx = self._sel_confirmed_index
                phrase = (
                    self._current_phrases[idx]
                    if idx < len(self._current_phrases)
                    else "?"
                )

                if phrase == SKIP_LABEL:
                    logger.info("Step '%s' skipped", phrase_engine.current_step_name)
                    phrase_engine.skip_step()
                    self._sel_state = SelectionState.EXECUTING
                    self._handle_post_selection()
                elif phrase == OTHER_LABEL:
                    logger.info("'Other' selected — loading more words for '%s'", phrase_engine.current_step_name)
                    self._sel_state = SelectionState.EXECUTING
                    self._refresh_grammar_words_and_resume(is_other=True)
                else:
                    phrase_engine.select_word_for_step(phrase)
                    logger.info("Word selected: '%s' for step '%s'", phrase, phrase_engine.current_step_name)
                    event_bus.emit(Event(
                        type=EventType.WORD_SELECTED,
                        data={
                            "word": phrase,
                            "sentence": phrase_engine.sentence,
                            "grammar_step": phrase_engine.current_step_name,
                            "selected_slots": phrase_engine.selected_slots,
                        },
                    ))
                    self._sel_state = SelectionState.EXECUTING
                    self._handle_post_selection()

    def _handle_post_selection(self) -> None:
        """After a word is selected or step skipped, decide next action."""
        if phrase_engine.is_sentence_complete():
            self._auto_send_sentence()
        else:
            self._refresh_grammar_words_and_resume()

    def _auto_send_sentence(self) -> None:
        """Auto-trigger done/send when all required grammar slots are filled."""
        sentence_text = phrase_engine.sentence_text
        sentence_list = phrase_engine.sentence
        logger.info("Sentence auto-sent: '%s'", sentence_text)

        phrase_engine.done_send()

        event_bus.emit(Event(
            type=EventType.SENTENCE_AUTO_SENT,
            data={"sentence_text": sentence_text, "sentence": sentence_list},
        ))
        event_bus.emit(Event(
            type=EventType.SELECTION_EXECUTED,
            data={"phrase": sentence_text, "sentence": sentence_list, "action": "auto_send"},
        ))
        event_bus.emit(Event(
            type=EventType.SENTENCE_CLEARED,
            data={"spoken": sentence_text},
        ))

        self._refresh_grammar_words_and_resume()

    def _refresh_grammar_words_and_resume(self, is_other: bool = False) -> None:
        """Load words for the current grammar step and resume highlighting."""
        def _do():
            try:
                if is_other:
                    words = phrase_engine.get_other_words()
                else:
                    words = phrase_engine.get_words_for_step()
                labels = [OTHER_LABEL]
                if phrase_engine.is_skippable:
                    labels.insert(0, SKIP_LABEL)
                self._current_phrases = words + labels
                self._flasher.set_phrases(self._current_phrases)
                self._emit_grammar_update(words)
                self._sel_highlight_index = 0
                self._sel_highlight_start = time.time()
                self._sel_last_blink_time = time.time()
                self._sel_state = SelectionState.HIGHLIGHTING
                event_bus.emit(Event(
                    type=EventType.HIGHLIGHT_CHANGED,
                    data={
                        "index": 0,
                        "phrase": self._current_phrases[0] if self._current_phrases else "",
                        "total": len(self._current_phrases),
                    },
                ))
            except Exception:
                logger.exception("Failed to refresh grammar words")
                self._sel_state = SelectionState.HIGHLIGHTING
                self._sel_highlight_start = time.time()
        threading.Thread(target=_do, daemon=True).start()

    def _refresh_words_and_resume(self, is_other: bool = False) -> None:
        """Backward compat wrapper."""
        self._refresh_grammar_words_and_resume(is_other=is_other)

    def _refresh_phrases_async(self) -> None:
        """Compatibility wrapper."""
        self._refresh_grammar_words_and_resume(is_other=False)

    def _run_clench_calibration_tick(self, now: float) -> None:
        """Handle clench calibration: user clenches 3 times to set personal threshold."""
        try:
            raw_samples = redis_store.get_recent_raw(seconds=0.3)
            if len(raw_samples) < 30:
                return
            data = np.array(
                [[s["tp9"], s["tp10"]] for s in raw_samples], dtype=np.float32,
            )
            rms_tp9 = float(np.sqrt(np.mean(np.square(data[:, 0]))))
            rms_tp10 = float(np.sqrt(np.mean(np.square(data[:, 1]))))
            rms = max(rms_tp9, rms_tp10)

            if rms >= config.CLENCH_RMS_THRESHOLD and now - self._clench_cal_last_time > 1.5:
                self._clench_cal_last_time = now
                self._clench_cal_count += 1
                self._clench_cal_rms_values.append(rms)
                logger.info(
                    "Clench calibration %d/%d (rms=%.1f)",
                    self._clench_cal_count, CALIBRATION_CLENCHES, rms,
                )
                event_bus.emit(Event(
                    type=EventType.CLENCH_CALIBRATION_STATUS,
                    data={
                        "state": "calibrating",
                        "clenches_detected": self._clench_cal_count,
                        "clenches_needed": CALIBRATION_CLENCHES,
                    },
                ))

                if self._clench_cal_count >= CALIBRATION_CLENCHES:
                    vals = np.array(self._clench_cal_rms_values)
                    baseline = float(vals.mean())
                    std = float(vals.std()) if len(vals) > 1 else 5.0
                    threshold = max(baseline * 0.5, np.percentile(vals, 40))
                    self._dynamic_clench_threshold = threshold
                    config.CLENCH_RMS_THRESHOLD = threshold
                    logger.info(
                        "Clench calibration complete — dynamic threshold=%.1f (mean=%.1f, std=%.1f)",
                        threshold, baseline, std,
                    )
                    event_bus.emit(Event(
                        type=EventType.CLENCH_CALIBRATION_STATUS,
                        data={
                            "state": "complete",
                            "clenches_detected": CALIBRATION_CLENCHES,
                            "clenches_needed": CALIBRATION_CLENCHES,
                            "threshold": round(threshold, 1),
                        },
                    ))
                    self._enter_highlighting(now)
        except Exception:
            logger.debug("Clench calibration tick error", exc_info=True)

    def _enter_highlighting(self, now: float) -> None:
        """Transition to the highlighting state."""
        self._sel_state = SelectionState.HIGHLIGHTING
        self._sel_highlight_index = 0
        self._sel_highlight_start = now
        self._sel_last_blink_time = now
        phrase = self._current_phrases[0] if self._current_phrases else ""
        event_bus.emit(Event(
            type=EventType.HIGHLIGHT_CHANGED,
            data={
                "index": 0,
                "phrase": phrase,
                "total": len(self._current_phrases),
            },
        ))

    def done_send(self) -> dict:
        """Done/Send: assemble sentence, clear it, generate fresh words, keep looping."""
        sentence_text = phrase_engine.sentence_text
        sentence_list = phrase_engine.sentence
        if not phrase_engine.has_selections():
            return {"error": "No sentence to send"}

        phrase_engine.done_send()
        logger.info("Done/Send: '%s'", sentence_text)

        event_bus.emit(Event(
            type=EventType.SELECTION_EXECUTED,
            data={"phrase": sentence_text, "sentence": sentence_list, "action": "done_send"},
        ))
        event_bus.emit(Event(
            type=EventType.SENTENCE_CLEARED,
            data={"spoken": sentence_text},
        ))

        if self._sel_state in (SelectionState.HIGHLIGHTING, SelectionState.EXECUTING):
            self._sel_state = SelectionState.EXECUTING
            self._refresh_grammar_words_and_resume(is_other=False)

        return {"status": "sent", "sentence": sentence_text}

    def _schedule_next_cycle(self, delay: float = 0.5) -> None:
        def _delayed():
            time.sleep(delay)
            if self._running:
                self._start_flash_cycle()
        threading.Thread(target=_delayed, daemon=True).start()


def main() -> None:
    app = create_app()
    app.state.orchestrator = BCIOrchestrator()

    logger.info(
        "Starting Hacklytic BCI on %s:%d (simulate=%s)",
        config.FASTAPI_HOST,
        config.FASTAPI_PORT,
        config.SIMULATE_EEG,
    )

    uvicorn.run(
        app,
        host=config.FASTAPI_HOST,
        port=config.FASTAPI_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
