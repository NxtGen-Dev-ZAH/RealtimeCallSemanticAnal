# Backend and Frontend Setup Guide

This guide reflects the current implementation: **FastAPI backend on port 8000** + **Next.js frontend on port 3000**.

## 1. Backend Setup

### Prerequisites
- Python 3.10+
- `uv`
- MongoDB Atlas URI (recommended)
- Optional: Hugging Face token for diarization/transcription models

### Install

```bash
cd backend
uv sync
python -m spacy download en_core_web_sm
```

### Environment

Create `backend/.env` from `backend/env_template.txt`:

```bash
cd backend
cp env_template.txt .env
```

Use these key values:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
MONGODB_DATABASE=call_center_db

HOST=0.0.0.0
PORT=8000
DEBUG=True

HF_TOKEN=your_hf_token
```

### Run backend

```bash
cd backend
python run_web_app.py
```

- API root: `http://localhost:8000/`
- Health: `http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`

## 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:3000`

## 3. How They Connect

`frontend/next.config.js` proxies:

- `/api/:path*` -> `http://localhost:8000/api/:path*`

So no CORS issue in local development.

## 4. MongoDB Atlas Setup (Recommended)

1. Create account: https://www.mongodb.com/cloud/atlas
2. Create free cluster (M0).
3. Create DB user (Database Access).
4. Whitelist IP (Network Access):
   - For dev: add `0.0.0.0/0`
5. Copy connection string from Atlas “Connect your application”.
6. Put it in `backend/.env` as `MONGODB_URI`.

Example format:

```env
MONGODB_URI=mongodb+srv://username:password@cluster-name.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
```

## 5. Model Status and Training

Current repository has model artifacts in `backend/models/`:
- `backend/models/sale_model.pkl`
- `backend/models/best_emotion_wav2vec2_v2/best_checkpoint/`

Set emotion model path for this checkpoint:

```env
EMOTION_MODEL_PATH=backend/models/best_emotion_wav2vec2_v2/best_checkpoint
```

Validate current artifacts:

```bash
python backend/scripts/validate_trained_models.py
```

Retraining remains optional and supported:

```bash
python backend/scripts/train_emotion_model.py --data_dir data/raw/ravdess --output_dir backend/models
python backend/scripts/train_sale_predictor.py --csv_path data/sale_training_data.csv --output_dir backend/models --early_stopping_rounds 20 --optimize_threshold
```
## 6. Troubleshooting

### Frontend cannot connect
- Ensure backend runs at `http://localhost:8000`
- Verify `frontend/next.config.js` destination uses port `8000`

### Upload works but analyze fails
- Usually MongoDB not connected. Confirm Atlas URI and network whitelist.

### Model load errors
- Verify `backend/models/sale_model.pkl` exists and `EMOTION_MODEL_PATH` points to `backend/models/best_emotion_wav2vec2_v2/best_checkpoint`.



