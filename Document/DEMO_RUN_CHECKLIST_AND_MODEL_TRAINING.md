# Demo Run Checklist and Model Training Guide

Generated on: 2026-03-02

Detailed model-training manuals:
- `Document/EMOTION_MODEL_TRAINING_GUIDE_MARCH_2026.md`
- `Document/SALE_PREDICTOR_MODEL_TRAINING_GUIDE_MARCH_2026.md`

## 1. Current Project State (What Matters for Demo)

### Backend (`backend/`)
- FastAPI app entrypoint is `backend/run_web_app.py` and runtime app is `src/call_analysis/web_app_fastapi.py`.
- `ConversationAnalyzer` requires trained model files at startup:
  - `backend/models/emotion_model.pth` **or** `backend/models/best_emotion_wav2vec2_v2/best_checkpoint`
  - `backend/models/sale_model.pkl`
- `backend/models/` must contain a valid emotion model artifact and `sale_model.pkl`, otherwise analysis fails.
- MongoDB is effectively required for frontend upload/analyze/status/history flow (`/api/analyze`, `/api/status`, `/api/results`, `/api/history` all depend on DB records).
- Backend and frontend are aligned to port `8000`:
  - Backend default: `Config.PORT=8000`
  - Frontend proxy: `/api/* -> http://localhost:8000/api/*`

### Frontend (`frontend/`)
- Next.js app with proxy rewrite in `frontend/next.config.js`:
  - `/api/*` -> `http://localhost:8000/api/*`
- API client uses relative `/api` by default (`frontend/src/lib/api.ts`).
- With backend on `8000`, frontend calls work through proxy without extra changes.

### Data (`data/`)
- Contains generated sale datasets (for example `sale_training_data.csv`, `synthetic_sales_full.csv`) and dataset statistics images/json.
- No raw SER dataset (for emotion model) exists in repo.
- No real labeled sale training dataset exists in repo.

### Documentation (`Document/`)
- Useful but inconsistent in places:
  - Some docs still referenced old variable names; standardized naming is now `HOST`, `PORT`, `DEBUG`, `SECRET_KEY`.
  - Emotion checkpoint path has been standardized to `best_emotion_wav2vec2_v2`.


## 2. Critical Blockers Before Demo
- [x] `backend/models/` exists and contains sale + emotion artifacts.

## 3. End-to-End Demo Checklist

Use PowerShell from repo root `d:\RealtimeCallSemanticAnal`.

### A. System prerequisites
- [ ] Install FFmpeg (needed for mp3/m4a audio handling).
  - **When it’s needed:** Required if you upload **.mp3** or **.m4a** (or convert other formats like .flac). **.wav** files work without FFmpeg.
  - **Windows:** Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) (e.g. “Windows builds” from gyan.dev or BtbN), extract the zip, and add the `bin` folder to your system PATH. Or with **winget:** `winget install FFmpeg` (if available). Or with **Chocolatey:** `choco install ffmpeg`.
  - **macOS:** `brew install ffmpeg`
  - **Linux:** `sudo apt-get install ffmpeg` (Debian/Ubuntu) or `sudo yum install ffmpeg` (RHEL/Fedora).
  - Verify: `ffmpeg -version`

### B. Backend environment + dependencies
- [ ] Enter backend:
  ```powershell
  cd backend
  ```
- [ ] Install dependencies (recommended: uv):
  ```powershell
  uv sync
  ```
- [ ] Install spaCy model:
  ```powershell
  python -m spacy download en_core_web_sm
  ```
- [ ] (If using Wav2Vec2 checkpoint) set model path in `backend/.env.local`:
  ```env
  EMOTION_MODEL_PATH=backend/models/best_emotion_wav2vec2_v2/best_checkpoint
  ```
### D. Train models (mandatory before web demo)
- [ ] (Optional) Retrain emotion model (Section 4.1 below) if you want fresh metrics.
- [ ] (Optional) Retrain sale predictor model (Section 4.2 below) if dataset/schema changed.
- [ ] Validate model files:
  ```powershell
  python backend\scripts\validate_trained_models.py
  ```
Expected outputs in `backend/models/`:
- `emotion_model.pth` **or** `best_emotion_wav2vec2_v2/best_checkpoint/`
- `sale_model.pkl`
- `sale_training_results.json`
- optional: `sale_model_scaler.pkl`, `sale_model_imputer.pkl`

### E. Start backend
- [ ] Run backend:
  ```powershell
  cd backend
  python run_web_app.py
  ```
- [ ] Verify health:
  ```powershell
  curl http://localhost:8000/health
  ```

### F. Start frontend
- [ ] In another terminal:
  ```powershell
  cd frontend
  npm install
  npm run dev
  ```
- [ ] Open: `http://localhost:3000`

### G. Demo flow
- [ ] Upload a small `.wav` file first (faster for live demo).
- [ ] Click Analyze.
- [ ] Wait for progress to reach completed.
- [ ] Show:
  - Sentiment chart
  - Emotion chart
  - Sale probability gauge
  - Key phrases
- [ ] Show History page and Export buttons (JSON/PDF/CSV).

## 4. Model Training Guide (Detailed)

## 4.1 Emotion Model (Recommended: `best_train_emotion_model.py`)

### Dataset options (for SER)
Preferred (already aligned with project script):
- RAVDESS (official): https://zenodo.org/records/1188976

Optional additional datasets (for future improvement):
- IEMOCAP info page: https://sail.usc.edu/iemocap/
- CREMA-D repository: https://github.com/CheyneyComputerScience/CREMA-D

### Data placement
- [ ] Download/extract RAVDESS into a folder, for example:
  - `data/raw/ravdess/`
- [ ] Confirm `.wav` files exist recursively under that folder.

### Training command (recommended for demo)
Run from repo root:
```powershell
python backend\scripts\best_train_emotion_model.py `
  --mode train `
  --data_dir data/raw/ravdess `
  --output_dir backend/models/best_emotion_wav2vec2_v2 `
  --base_model facebook/wav2vec2-base `
  --emotion_mapping default `
  --split_strategy actor_holdout `
  --train_actor_max 18 `
  --epochs 30 `
  --batch_size 8 `
  --learning_rate 2e-5
```

### Output artifacts
- `backend/models/best_emotion_wav2vec2_v2/best_checkpoint/`
- `backend/models/best_emotion_wav2vec2_v2/data_validation_report.json`
- `backend/models/best_emotion_wav2vec2_v2/best_emotion_training_history.json`
- `backend/models/best_emotion_wav2vec2_v2/best_emotion_training_summary.json`
- `backend/models/best_emotion_wav2vec2_v2/best_emotion_confusion_matrix.png`

### Practical notes
- Training on CPU is slow; use GPU if available.
- If memory issues occur, reduce `--batch_size` and/or `--max_seconds`.
- Keep `--emotion_mapping default` for sale-prediction demo compatibility.

## 4.2 Sale Predictor (`train_sale_predictor.py`)

You need a tabular dataset with at least:
- `sentiment_mean`
- `sentiment_variance`
- `emotion_neutral`
- `emotion_happiness`
- `emotion_anger`
- `emotion_sadness`
- `emotion_frustration`
- `silence_ratio`
- `interruption_frequency`
- `talk_listen_ratio`
- `turn_taking_frequency`
- `sale_outcome` (0/1 label)

### Option A: Generate synthetic training data (quickest for demo)
```powershell
python backend\scripts\generate_sale_training_dataset.py `
  --n_samples 10000 `
  --sale_ratio 0.30 `
  --output data/sale_training_data.csv `
  --stats_dir data/dataset_stats `
  --overwrite
```

Then train:
```powershell
python backend\scripts\train_sale_predictor.py `
  --csv_path data/sale_training_data.csv `
  --output_dir backend/models `
  --early_stopping_rounds 20 `
  --optimize_threshold
```

### Option B: Build real dataset from call center + CRM data (recommended for final quality)

1. Collect call recordings with matching final outcome from CRM.
2. Keep one row per call with `sale_outcome` (0/1).
3. Run your preprocessing/analysis pipeline to extract per-call features listed above.
4. Join extracted features with CRM label.
5. Export final CSV and train with `train_sale_predictor.py`.

### Real data sourcing strategy (practical)
- Use your own call center recordings plus CRM conversion status as ground truth.
- If internal data is limited, start with synthetic data for demo, then progressively replace with real labeled calls.
- Keep strict privacy controls:
  - remove direct PII identifiers
  - store consent/compliance metadata
  - avoid exporting raw personal fields into training CSV

## 4.3 Validate trained models
```powershell
python backend\scripts\validate_trained_models.py
```
If this fails, do not proceed to demo until both emotion and sale models pass.

## 5. Known Issues to Address Before Final Presentation

1. Port alignment:
   - Backend default is `8000`.
   - Frontend proxy is set to `http://localhost:8000/api/:path*`.

2. Documentation alignment:
   - Use `HOST`, `PORT`, `DEBUG`, `SECRET_KEY` consistently in `.env`.

3. Training pipeline script references missing files:
   - `backend/scripts/test_training_pipeline.py` references `generate_mock_emotion_data.py` and `generate_mock_sale_data.py`, which are not present.

4. Feature-schema consistency warning:
   - Runtime fused features include `filler_word_frequency`, while sale training script expected-feature list may differ.
   - Keep training feature columns and inference feature generation strictly aligned before final benchmark runs.

## 6. Minimum "Day of Demo" Command Set

Terminal 1:
```powershell
cd backend
python run_web_app.py
```

Terminal 2:
```powershell
cd frontend
npm run dev
```

Health check:
```powershell
curl http://localhost:8000/health
```

If health is up, open:
- `http://localhost:3000`

