# Hacklytic — Hybrid BCI Assistive Communication

A brain-computer interface that uses **Muse 2** consumer EEG hardware with **blink-to-select** and **jaw clench** gesture detection, combined with **Gemini-powered phrase suggestions**, to enable rapid assistive communication.

## How It Works

1. **Muse 2** streams 4-channel EEG (TP9, AF7, AF8, TP10) via Lab Streaming Layer
2. A real-time pipeline applies bandpass filtering and **EEGNet** ML-based gesture classification
3. **Blink** to confirm a highlighted word/phrase; words follow a grammar flow (Subject → Adverb → Adjective → Action)
4. **Jaw clench** (double clench) to clear the sentence / undo
5. **Gemini** generates contextual word suggestions for each grammar step
6. **Text-to-speech** speaks the completed sentence aloud (Gemini TTS or Web Speech API fallback)
7. Optional **P300** mode with Pygame stimulus window for attention-based selection

## Architecture

```
Muse 2 → muselsl → LSL → FastAPI Backend → React Frontend
                              ↕ WebSocket
                         EEG pipeline, gesture detection,
                         phrase engine, TTS triggers
```

- **FastAPI backend** — EEG pipeline, Redis, EEGNet gesture model, phrase engine, Gemini API
- **React frontend** — grammar UI, sentence building, TTS playback, controls
- **Pygame window** — optional P300 stimulus flashing

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis server
- Muse 2 headband (or use simulated mode)
- Gemini API key (optional for TTS; Web Speech API fallback available)

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your GEMINI_API_KEY (optional)
```

### Frontend Setup

```bash
cd frontend
npm install

cp .env.example .env
# Edit .env with your GEMINI_API_KEY (optional, for Gemini TTS)
```

### Run

**Option A — Simulated EEG (no Muse 2 needed):**

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start backend (simulated EEG)
cd backend
source .venv/bin/activate
SIMULATE_EEG=true python main.py

# Terminal 3: Start frontend
cd frontend
npm run dev
```

**Option B — With Muse 2 (real EEG):**

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Connect Muse 2 and stream (power on headset, hold button to pair)
cd backend
source .venv/bin/activate
muselsl stream

# Terminal 3: Start backend (connect to existing LSL stream)
cd backend
source .venv/bin/activate
python main.py

# Terminal 4: Start frontend
cd frontend
npm run dev
```

- **Backend API:** http://localhost:8000 (Swagger at `/docs`)
- **Frontend:** http://localhost:3000
- **Pygame stimulus window** opens automatically when P300 mode is used

### First-Time Setup (Blink-to-Select)

1. Click **Start** to begin the selection session
2. **Warmup** (~2 s) — let the signal stabilize
3. **Blink calibration** — blink 2 times when prompted
4. **Clench calibration** — clench 3 times when prompted
5. Words cycle in highlight; **blink** to select, **double clench** to clear

## Tech Stack

**Backend:** FastAPI, muselsl, pylsl, Redis, SciPy, scikit-learn, PyTorch (EEGNet), Pygame, Gemini API  
**Frontend:** React 19, Vite, TypeScript, Tailwind CSS, Recharts, Framer Motion, Gemini TTS, Web Speech API
