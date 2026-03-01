# Backend Folder — Analysis & Run Guide

This document gives a structured overview of the backend and step-by-step instructions to run the project.

---

## 1. Backend Folder Structure

```
backend/
├── config.py                 # Central config (env vars, paths, model names)
├── db_connection.py          # DB connection helpers
├── pyproject.toml            # Package definition (uv/pip), Python ≥3.10
├── requirements.txt          # Pip dependencies (alternative to pyproject.toml)
├── setup.py                  # Setup script: pip install -e ., create dirs, test demo
├── run_web_app.py            # ★ Start FastAPI web app (uvicorn)
├── run_full_analysis.py      # CLI: full pipeline on one audio file
├── run_diarization_only.py   # CLI: diarization on existing transcription
├── run_demo.py              # CLI: run demo (DemoSystem) without web
│
├── src/
│   └── call_analysis/       # Main package
│       ├── __init__.py      # Exports + main() → DemoSystem().run_demo()
│       ├── preprocessing.py # AudioProcessor, TextProcessor (Whisper, PyAnnote, etc.)
│       ├── feature_extraction.py
│       ├── models.py        # SentimentAnalyzer, EmotionDetector, SalePredictor, ConversationAnalyzer
│       ├── dashboard.py    # Plotly dashboards / HTML export
│       ├── demo.py          # DemoSystem (demo conversations + pipeline)
│       └── web_app_fastapi.py  # FastAPI app (routes, upload, analyze, export)
│
├── scripts/                 # Training & validation
│   ├── train_emotion_model.py
│   ├── train_sale_predictor.py
│   ├── generate_sale_training_dataset.py
│   ├── validate_trained_models.py
│   ├── validate_production_readiness.py
│   ├── test_training_pipeline.py
│   ├── transcribe_audio.py
│   ├── analyze_pyannote_output.py
│   └── README_TRAINING.md
│
├── tests/
│   ├── test_models.py
│   ├── test_integration.py
│   └── test_feature_extraction.py
│
├── docs/
│   ├── COMPLETE_SYSTEM_FLOW.md   # End-to-end flow (training → inference)
│   ├── DESIGN_DECISIONS.md       # Architecture and design choices
│   ├── FYP_DOCUMENT_ALIGNMENT.md # FYP requirements vs implementation
│   └── BACKEND_ANALYSIS_AND_RUN_GUIDE.md  # This file
│
├── models/                  # (Created by training) emotion_model.pth, sale_model.pkl, etc.
├── uploads/                 # (Created at runtime) uploaded audio
├── output/                  # (Created at runtime) analysis outputs
└── .env                     # (You create) env vars — see CONFIGURATION.md
```

---

## 2. Main Components

| Component | Role |
|----------|------|
| **config.py** | Reads `.env` via `python-dotenv`; exposes `Config` (HF_TOKEN, DB, FastAPI host/port, model names, demo flags). |
| **preprocessing** | Audio transcription (Whisper), speaker diarization (PyAnnote/WhisperX/Resemblyzer), text processing. |
| **feature_extraction** | Audio + text features (MFCC, mel, BERT, etc.) for models. |
| **models** | Sentiment (DistilBERT/FinBERT), emotion (CNN+LSTM), sale predictor (XGBoost); `ConversationAnalyzer` orchestrates. |
| **dashboard** | Builds Plotly dashboards and HTML exports. |
| **demo** | `DemoSystem`: demo conversations, simulated audio, full pipeline for POC. |
| **web_app_fastapi** | REST API: health, upload, analyze, results, status, history, export (JSON/PDF/CSV). |

---

## 3. How to Run the Project

All commands below are run from the **backend** directory unless stated otherwise.

### 3.1 Prerequisites

- **Python 3.10+**
- **Virtual environment** (recommended):  
  `python -m venv .venv` then activate (e.g. `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux).

### 3.2 One-time setup

1. **Install the package and dependencies**

   Using pip (from `backend/`):

   ```bash
   pip install -e .
   ```

   Or install from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Or using uv (if you use it):

   ```bash
   uv sync
   ```

2. **Environment variables**

   - Copy the sample from `CONFIGURATION.md` into a `.env` file in `backend/`.
   - Minimum for basic/demo use:
     - `HF_TOKEN` — Hugging Face token (needed for real Whisper/PyAnnote and some models).
   - Optional: MongoDB, PostgreSQL, `DEMO_MODE`, `USE_SIMULATED_AUDIO`, etc. (see `config.py` and `CONFIGURATION.md`).

3. **spaCy model (for PII masking)**

   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Optional: ffmpeg**  
   Needed for non-WAV uploads (e.g. MP3/M4A). Install and add to PATH (see `WARNINGS_FIX.md` for OS-specific notes).

### 3.3 Run the web application (main way to “run the project”)

From `backend/`:

```bash
python run_web_app.py
```

- Server runs at **http://0.0.0.0:8000** (or the host/port in `Config`: `HOST`, `PORT`).
- With `DEBUG=True`, uvicorn runs with reload.
- Frontend (e.g. Next.js) typically runs on port 3000; CORS is set for `localhost:3000`.

**Useful URLs:**

- API root: http://localhost:8000/
- Health: http://localhost:8000/health
- OpenAPI: http://localhost:8000/docs

### 3.4 Run the demo (no web server)

From `backend/`:

```bash
python run_demo.py
```

This runs `DemoSystem().run_demo()`: demo conversations, simulated audio, full pipeline, and prints/shows demo results.

### 3.5 Run full analysis on one audio file (CLI)

From `backend/` (or ensure `backend` and `src` are on `PYTHONPATH`):

```bash
python run_full_analysis.py <path_to_audio_file> [--call-id CALL_ID] [--output-dir OUTPUT_DIR]
```

Example:

```bash
python run_full_analysis.py ../data/sample.wav --output-dir output
```

Runs: audio → transcription → diarization → features → models → dashboard/output.

### 3.6 Run diarization only (existing transcription)

When you already have a transcription and only want diarization:

```bash
python run_diarization_only.py <audio.wav> <transcription_json_or_txt>
```

See script help and `run_diarization_only.py` for optional arguments.

### 3.7 Run tests

From `backend/`:

```bash
python -m pytest tests/ -v
```

Or run a single test file:

```bash
python -m pytest tests/test_models.py -v
```

---

## 4. Training (optional)

If you want to train or retrain models:

- **Emotion model (CNN+LSTM):**  
  `scripts/train_emotion_model.py` — expects RAVDESS-style data; outputs e.g. `backend/models/emotion_model.pth`.
- **Sale predictor (XGBoost):**  
  `scripts/train_sale_predictor.py` — expects a CSV with fused features and `sale_outcome`; outputs `sale_model.pkl`, scaler, etc.
- **Dataset generation for sale model:**  
  `scripts/generate_sale_training_dataset.py`.

See `scripts/README_TRAINING.md` for detailed usage and paths.

---

## 5. Configuration summary

| Variable | Purpose | Default (if any) |
|----------|---------|------------------|
| `HF_TOKEN` | Hugging Face (Whisper, PyAnnote, etc.) | — |
| `DEMO_MODE` | Use demo/simulated data | `True` |
| `USE_SIMULATED_AUDIO` | Simulated vs real audio processing | `True` |
| `HOST` / `PORT` | FastAPI bind address | `0.0.0.0` / `8000` |
| `DEBUG` | Uvicorn reload | `True` |
| `MONGODB_URI` | MongoDB connection | `mongodb://localhost:27017/` |

Full list and defaults are in `config.py` and `CONFIGURATION.md`.

---

## 6. Quick reference: “I want to…”

| Goal | Command / action |
|------|-------------------|
| Run the app in the browser | `python run_web_app.py` → open http://localhost:8000 |
| Run demo in terminal | `python run_demo.py` |
| Analyze one audio file | `python run_full_analysis.py <audio_file> [--output-dir output]` |
| Only diarize from existing transcription | `python run_diarization_only.py <audio> <transcription>` |
| Install deps | `pip install -e .` or `pip install -r requirements.txt` |
| Set config | Create `backend/.env` from `CONFIGURATION.md` |
| Run tests | `python -m pytest tests/ -v` |
| Train emotion model | `python scripts/train_emotion_model.py ...` (see scripts/README_TRAINING.md) |
| Train sale model | `python scripts/train_sale_predictor.py ...` (see scripts/README_TRAINING.md) |

---

## 7. Notes

- **`run_demo.py`** is a thin wrapper that sets `sys.path` and calls `DemoSystem().run_demo()` so the README’s “run the demo” instruction works.
- **Models:** If `backend/models/` is missing or models are not trained, the app may still run in demo/simulated mode; real inference may require training first (see `scripts/README_TRAINING.md` and `docs/COMPLETE_SYSTEM_FLOW.md`).
- **Warnings:** Non-critical warnings (e.g. ffmpeg, optional libs) are explained in `WARNINGS_FIX.md`.
