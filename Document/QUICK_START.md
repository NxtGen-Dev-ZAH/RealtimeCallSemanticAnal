# Quick Start Guide

## Run the Application

### Prerequisites
- Python 3.10+
- `uv` package manager
- Node.js 18+
- MongoDB Atlas URI (recommended) or local MongoDB
- FFmpeg (for mp3/m4a handling)

## Backend Setup

```bash
cd backend
uv sync
python -m spacy download en_core_web_sm
```

Create `backend/.env` (or copy from `backend/env_template.txt`):

```env
HF_TOKEN=your_hf_token
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
MONGODB_DATABASE=call_center_db

HOST=0.0.0.0
PORT=8000
DEBUG=True

SENTIMENT_MODEL=distilbert
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=104857600
ALLOWED_EXTENSIONS=wav,mp3,m4a,flac
```

Run backend:

```bash
cd backend
python run_web_app.py
```

Backend URL: `http://localhost:8000`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:3000`

## Connection

- Frontend `/api/*` is proxied to backend `http://localhost:8000/api/*`.
- Health check:

```bash
curl http://localhost:8000/health
```

## Important Note

Current repo now contains model artifacts under `backend/models/`:
- Sale model: `backend/models/sale_model.pkl`
- Emotion checkpoint: `backend/models/best_emotion_wav2vec2_v2/best_checkpoint`

Set this in `backend/.env` (or `.env.local`) for current emotion artifact:

```env
EMOTION_MODEL_PATH=backend/models/best_emotion_wav2vec2_v2/best_checkpoint
```

Validate before demo:

```bash
python backend/scripts/validate_trained_models.py
```
