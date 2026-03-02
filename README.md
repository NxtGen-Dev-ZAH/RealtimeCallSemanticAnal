# Call Analysis System

An AI-powered sentiment analysis system that predicts sale probability using multimodal data from call center recordings.

## 🎯 Project Status

**Overall Completion:** ~80%  
**Models Trained:** Sale predictor available (`backend/models/sale_model.pkl`); emotion model checkpoint available (`backend/models/best_emotion_wav2vec2_v2/best_checkpoint`)
**Integration:** ✅ Frontend-Backend API Complete  
**Production Ready:** ⚠️ Testing & Validation Ongoing

---

## ✨ Features

### Core Functionality
- **Multimodal Analysis**: Combines audio emotion detection and text sentiment analysis
- **Sale Prediction**: Predicts probability of sale (0-100%) based on conversation analysis
- **Speaker Diarization**: Identifies and separates different speakers in calls using Pyannote.audio
- **Real-time Dashboard**: Visual insights with sentiment curves and conversion metrics
- **Key Phrase Extraction**: Automatically identifies important phrases contributing to sentiment
- **Filler Word Detection**: Detects and counts filler words (um, uh, like, etc.)
- **Confidence Intervals**: Provides prediction uncertainty estimation

### Advanced Features
- **Dual Sentiment Models**: Supports both DistilBERT (general) and FinBERT (financial domain)
- **Model Training Pipeline**: Emotion and sale prediction training scripts are included (retraining supported)
- **Async Processing**: Non-blocking audio upload and analysis workflow
- **Export Capabilities**: PDF, CSV, and JSON export formats
- **MongoDB Integration**: Persistent storage with MongoDB Atlas support
- **PII Masking**: Automatic masking of personally identifiable information

---

## 🏗️ System Architecture

### Modules
1. **Data Preprocessing**: Audio transcription (Whisper ASR) and speaker separation (Pyannote.audio)
2. **Feature Extraction**: Audio (MFCC, spectral, chroma) and text (BERT embeddings) feature engineering
3. **Sentiment & Emotion Models**: 
   - **Sentiment**: DistilBERT/FinBERT for text analysis
   - **Emotion**: CNN+LSTM hybrid network for acoustic emotion recognition
4. **Sale Prediction Model**: XGBoost classifier with probability calibration
5. **Visualization Dashboard**: React-based web interface with interactive charts

### Technologies
- **Backend**: Python 3.10+, PyTorch 2.2+, FastAPI
- **ML Models**: Hugging Face Transformers, OpenAI Whisper, Pyannote.audio
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Database**: MongoDB (Atlas supported)
- **Visualization**: Plotly.js, Recharts

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm/yarn
- MongoDB (local or Atlas)
- Hugging Face account (for model access)

### Installation

#### 1. Backend Setup

**Using uv (Recommended):**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to backend directory
cd backend

# Install all dependencies (creates venv and installs from pyproject.toml)
uv sync

# OR install in editable mode
uv pip install -e .
```

**Alternative using pip:**
```bash
cd backend
pip install -e .
```

#### 2. Environment Configuration

Create a `.env` file in `backend/`:

```bash
# Copy the environment template
cp backend/env_template.txt backend/.env
```

**Required Environment Variables:**
```env
# Hugging Face Token (Required for Whisper and Pyannote)
HF_TOKEN=your_huggingface_token_here

# MongoDB Connection (Required for data persistence)
MONGODB_URI=mongodb://localhost:27017/  # or mongodb+srv://... for Atlas
MONGODB_DATABASE=call_center_db
MONGODB_ENABLED=true

# Sentiment Model Selection
SENTIMENT_MODEL=distilbert  # Options: 'distilbert' or 'finbert'
EMOTION_MODEL_PATH=backend/models/best_emotion_wav2vec2_v2/best_checkpoint

# Optional: File Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=104857600  # 100MB
```

#### 3. Install spaCy Model (for PII masking)
```bash
python -m spacy download en_core_web_sm
```

#### 4. Frontend Setup
```bash
cd frontend
npm install
# or
yarn install
```

#### 5. Start the Application

**Backend (FastAPI):**
```bash
cd backend
python -m uvicorn src.call_analysis.web_app_fastapi:app --reload --port 8000
```

**Frontend (Next.js):**
```bash
cd frontend
npm run dev
# or
yarn dev
```

**Access the Application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📚 Usage

### Upload and Analyze Audio

1. **Upload Audio File**
   - Navigate to http://localhost:3000
   - Click "Upload Audio" and select a WAV, MP3, or M4A file
   - File is uploaded and saved (status: pending)

2. **Start Analysis**
   - Click "Analyze" button
   - Analysis runs asynchronously with progress updates
   - Status updates: pending → processing → completed

3. **View Results**
   - Dashboard displays:
     - Sentiment timeline chart
     - Emotion distribution
     - Sale probability gauge
     - Key phrases (positive/negative)
     - Conversational dynamics metrics

4. **Export Results**
   - Click export buttons for PDF, CSV, or JSON formats
   - Files download automatically

### API Usage

**Upload Audio:**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "audio=@your_audio.wav"
```

**Start Analysis:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"call_id": "upload_20250101_120000"}'
```

**Get Results:**
```bash
curl http://localhost:8000/api/results/upload_20250101_120000
```

**Get Status:**
```bash
curl http://localhost:8000/api/status/upload_20250101_120000
```

---

## 🧪 Model Training

### Emotion Model (Wav2Vec2 Recommended)

Train on RAVDESS dataset:
```bash
python backend/scripts/best_train_emotion_model.py \
  --mode train \
  --data_dir data/raw/ravdess/ \
  --epochs 30 \
  --batch_size 8 \
  --output_dir backend/models/best_emotion_wav2vec2_v2
```

**Output:**
- `best_emotion_wav2vec2_v2/best_checkpoint/` - Trained Wav2Vec2 checkpoint
- `best_emotion_training_summary.json` - Training summary
- `best_emotion_training_history.json` - Training metrics

### Sale Predictor (XGBoost)

Train on labeled feature data:
```bash
python backend/scripts/train_sale_predictor.py \
  --csv_path data/sale_training_data.csv \
  --output_dir backend/models/ \
  --early_stopping_rounds 20 \
  --optimize_threshold
```

**Output:**
- `sale_model.pkl` - Trained XGBoost model
- `sale_model_imputer.pkl` - Feature imputation model
- `sale_model_scaler.pkl` - Feature scaler (if used)
- `sale_training_results.json` - Training metrics and optimal threshold
- `sale_model_feature_importance.png` - Feature importance visualization
- `sale_model_roc_curve.png` - ROC curve visualization

---

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload audio file (async) |
| POST | `/api/analyze` | Start analysis for uploaded call |
| GET | `/api/results/{call_id}` | Get analysis results |
| GET | `/api/status/{call_id}` | Get analysis status |
| GET | `/api/history` | Get call history |
| GET | `/api/export/{call_id}` | Export JSON |
| GET | `/api/export/{call_id}/pdf` | Export PDF report |
| GET | `/api/export/{call_id}/csv` | Export CSV data |

### Demo Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List demo conversations |
| GET | `/api/analyze/{id}` | Analyze demo conversation |
| GET | `/api/dashboard/{id}` | Generate dashboard HTML |
| GET | `/api/insights` | Get agent insights |

---

## 🔧 Configuration

### Sentiment Model Selection

Choose between DistilBERT (general) or FinBERT (financial):

**Via Environment Variable:**
```env
SENTIMENT_MODEL=finbert  # or 'distilbert'
```

**Via Code:**
```python
from src.call_analysis.models import SentimentAnalyzer

# Use FinBERT for financial domain
analyzer = SentimentAnalyzer(model_name='finbert')

# Use DistilBERT for general sentiment
analyzer = SentimentAnalyzer(model_name='distilbert')
```

### MongoDB Atlas Setup

1. Create MongoDB Atlas account: https://www.mongodb.com/cloud/atlas
2. Create cluster (free tier available)
3. Configure network access (add IP whitelist)
4. Create database user
5. Get connection string
6. Update `.env`:
   ```env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/call_center_db?retryWrites=true&w=majority
   ```

---

## 📁 Project Structure

```
Call_Analysis/
├── backend/
│   ├── src/call_analysis/
│   │   ├── models.py              # ML models (Sentiment, Emotion, Sale Predictor)
│   │   ├── preprocessing.py       # Audio processing, transcription, diarization
│   │   ├── feature_extraction.py  # Feature engineering
│   │   ├── dashboard.py           # Dashboard generation
│   │   ├── web_app_fastapi.py     # FastAPI application
│   │   └── demo.py                # Demo system
│   ├── scripts/
│   │   ├── train_emotion_model.py
│   │   ├── train_sale_predictor.py
│   │   └── validate_trained_models.py
│   ├── models/                    # Trained model files
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── app/                   # Next.js pages
│   │   ├── components/            # React components
│   │   └── lib/                   # API service, types
│   └── package.json
├── data/
│   └── raw/                       # Training datasets
├── output/                        # Generated reports and dashboards
└── uploads/                       # Uploaded audio files
```

---

## 🧪 Testing

### Validate Trained Models
```bash
python backend/scripts/validate_trained_models.py
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Test upload
curl -X POST http://localhost:8000/api/upload -F "audio=@test.wav"
```

### Run Integration Tests
```bash
cd backend
pytest tests/
```

---

## 📖 Documentation

- **FYP Completion Guide**: `FYP_COMPLETION_README.md`
- **Document Alignment**: `backend/docs/FYP_DOCUMENT_ALIGNMENT.md`
- **Architecture**: `Document/ARCHITECTURE_DIAGRAMS.md`
- **API Documentation**: http://localhost:8000/docs (when server is running)

---

## 🐛 Troubleshooting

### Models Not Loading
- Ensure model files exist in `backend/models/`
- Check file permissions
- Verify model paths in code

### MongoDB Connection Failed
- Check `MONGODB_URI` in `.env`
- Verify network access (for Atlas)
- Check IP whitelist (for Atlas)
- Ensure MongoDB is running (for local)

### Hugging Face Authentication Error
- Verify `HF_TOKEN` is set correctly
- Check token hasn't expired
- Accept model licenses on Hugging Face website

### Frontend Can't Connect to Backend
- Ensure backend is running on port 8000
- Check CORS configuration
- Verify `NEXT_PUBLIC_API_URL` in frontend

---

## 👥 Intended Users

- Call center managers
- Sales agents and analysts
- Customer service teams
- Researchers studying conversation analysis

---

## 📝 License

This project is part of a Final Year Project (FYP) submission.

---

## 🙏 Acknowledgments

- OpenAI Whisper for ASR
- Hugging Face for transformer models
- Pyannote.audio for speaker diarization
- RAVDESS dataset for emotion training

---

## 📞 Support

For issues or questions:
1. Check `FYP_COMPLETION_README.md` for common issues
2. Review API documentation at `/docs`
3. Check logs in `backend/logs/`



