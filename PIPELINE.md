# Complete Project Pipeline Documentation

## Overview
This document describes the complete end-to-end pipeline for the Real-time Call Semantic Analysis system, from audio file upload to final results visualization.

---

## 🎯 System Architecture

```
┌─────────────┐
│   Frontend  │  Next.js (React) - User Interface
│  (Next.js)  │
└──────┬──────┘
       │ HTTP/REST API
       ▼
┌─────────────┐
│   Backend   │  FastAPI - REST API Server
│  (FastAPI)  │
└──────┬──────┘
       │
       ├──► Audio Processing Pipeline
       ├──► Feature Extraction
       ├──► ML Models (Sentiment, Emotion, Sale Prediction)
       └──► Database (MongoDB/PostgreSQL)
```

---

## 📊 Complete Pipeline Flow

### **Phase 1: Audio Upload & Initialization**

#### 1.1 Frontend Upload (`frontend/src/components/UploadForm.tsx`)
- **User Action**: User selects audio file (.wav, .mp3, .m4a)
- **Validation**: 
  - File type validation (allowed: wav, mp3, m4a)
  - File size validation (max 100MB)
- **Upload**: POST request to `/api/upload` endpoint
- **Response**: Returns `call_id`, `filename`, `size`

#### 1.2 Backend Receives Upload (`backend/src/call_analysis/web_app_fastapi.py`)
- **Endpoint**: `POST /api/upload`
- **Actions**:
  - Validates file format
  - Saves file to `uploads/` directory
  - Generates unique `call_id` (format: `upload_YYYYMMDD_HHMMSS`)
  - **Immediately starts full analysis pipeline** (synchronous)

---

### **Phase 2: Audio Preprocessing**

#### 2.1 Audio Format Validation (`backend/src/call_analysis/preprocessing.py`)
- **Module**: `AudioProcessor.validate_audio_format()`
- **Actions**:
  - Checks file extension (.wav, .mp3, .m4a)
  - Converts non-WAV formats to WAV if needed
  - Returns validated audio path

#### 2.2 Audio Transcription (Whisper ASR)
- **Module**: `AudioProcessor.transcribe_audio()`
- **Model**: OpenAI Whisper (configurable size: tiny, base, small, medium, large)
- **Process**:
  1. Load Whisper model (lazy loading)
  2. Transcribe audio to text
  3. Extract segments with timestamps
  4. Detect language
  5. Calculate duration
- **Output**: 
  ```json
  {
    "text": "Full transcription text...",
    "segments": [
      {"start": 0.0, "end": 5.2, "text": "Hello..."},
      ...
    ],
    "language": "en",
    "duration": 120.5
  }
  ```
- **Storage**: 
  - Saved to `output/{call_id}_transcription.json`
  - Optionally saved to MongoDB (`transcriptions` collection)

#### 2.3 Speaker Diarization (REQUIRED)
- **Module**: `AudioProcessor.perform_speaker_diarization()`
- **Methods Available** (in order of preference):
  
  **Method 1: WhisperX 3.x Built-in (Pyannote.audio)** ⭐ Most Accurate
  - Uses WhisperX's built-in diarization pipeline
  - Requires `HF_TOKEN` in .env
  - More accurate but slower (20-40 min for long files)
  - Process:
    1. Transcribe with WhisperX (word-level timestamps)
    2. Align timestamps
    3. Run Pyannote.audio diarization
    4. Assign speakers to words using `assign_word_speakers`
  
  **Method 2: WhisperX + Resemblyzer** ⚡ Fastest (10x faster)
  - CPU-friendly, recommended for development
  - Process:
    1. Transcribe with WhisperX
    2. Align word-level timestamps
    3. Extract speaker embeddings using Resemblyzer
    4. Cluster speakers using hierarchical clustering
    5. Merge similar speakers (configurable threshold)
  
  **Method 3: Pyannote.audio (Fallback)**
  - Traditional Pyannote pipeline
  - Slower but reliable fallback

- **Features**:
  - **Chunking Support**: For long audio files (>5 min), processes in chunks
  - **Cross-chunk Speaker Matching**: Matches speakers across chunks using embedding similarity
  - **Role Identification**: Identifies AGENT vs CUSTOMER using:
    - First speaker heuristic
    - Speaking time analysis
    - Keyword analysis (agent/customer keywords)
    - Question pattern analysis
  - **Configurable Parameters**:
    - `max_speakers`: Maximum speakers (safety limit)
    - `clustering_threshold`: Distance threshold (lower = more speakers)
    - `min_segment_duration`: Minimum segment duration
    - `speaker_merge_threshold`: Similarity for merging speakers

- **Output**:
  ```json
  [
    {
      "speaker": "AGENT",  // or "CUSTOMER" or "SPEAKER_00"
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, how can I help you?",
      "original_speaker_id": "SPEAKER_00"  // Original ID before role identification
    },
    ...
  ]
  ```
- **Storage**: 
  - Saved to `output/{call_id}_diarization.json`
  - Optionally saved to MongoDB (`diarization` collection)

#### 2.4 Text Processing & Segmentation
- **Module**: `TextProcessor.segment_conversation()`
- **Process**:
  1. **PII Masking**: Masks personally identifiable information (phones, emails, names)
  2. **Text-to-Segment Mapping**: Maps transcription text to diarization segments using timestamps
  3. **Feature Extraction**: Extracts BERT-compatible text features for each segment
  4. **Segment Creation**: Creates processed segments with:
     - Start/end times
     - Speaker labels
     - Text content
     - Text features (BERT tokens, word count, etc.)
- **Output**: List of processed segments with features
- **Storage**: Optionally saved to MongoDB (`segments` collection)

---

### **Phase 3: Feature Extraction**

#### 3.1 Audio Feature Extraction
- **Module**: `AudioProcessor.extract_audio_features()`
- **Libraries**: librosa
- **Features Extracted**:
  - **MFCC** (Mel-frequency cepstral coefficients): 13 coefficients
  - **Spectral Centroid**: Brightness of sound
  - **Spectral Rolloff**: Frequency below which 85% of energy is contained
  - **Zero Crossing Rate**: Rate of sign changes
  - **Chroma**: Pitch class profile (12 dimensions)
  - **Mel Spectrogram**: Frequency representation
  - **Duration**: Audio length in seconds
  - **Sample Rate**: Audio sample rate (typically 16kHz)

#### 3.2 Text Feature Extraction
- **Module**: `FeatureExtractor.extract_text_features()`
- **Model**: BERT (bert-base-uncased)
- **Process**:
  1. Mask PII in text
  2. Tokenize using BERT tokenizer
  3. Generate BERT embeddings (768-dimensional)
  4. Use [CLS] token embedding as text representation
- **Output**: 768-dimensional BERT embedding vector

#### 3.3 Temporal Feature Extraction
- **Module**: `FeatureExtractor.extract_temporal_features()`
- **Features**:
  - Total conversation duration
  - Number of speaker changes
  - Average segment duration
  - Speaker balance (ratio of speaking times)
  - Sentiment trend (linear fit of sentiment over time)
  - Sentiment volatility (standard deviation)
  - Final sentiment score

#### 3.4 Feature Fusion
- **Module**: `FeatureExtractor.combine_features()`
- **Process**:
  1. Flatten audio features (MFCC means/stds, spectral features, etc.)
  2. Combine with BERT text embeddings (768-dim)
  3. Append temporal features
  4. Normalize using StandardScaler
- **Output**: Single fused feature vector (~850+ dimensions)

---

### **Phase 4: ML Model Analysis**

#### 4.1 Sentiment Analysis
- **Module**: `SentimentAnalyzer` (`backend/src/call_analysis/models.py`)
- **Model**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- **Process**:
  1. Uses Hugging Face sentiment analysis pipeline
  2. Analyzes each conversation segment
  3. Returns sentiment label (positive/negative/neutral) and confidence score
- **Output per segment**:
  ```json
  {
    "sentiment": "positive",
    "score": 0.85,
    "confidence": 0.92,
    "positive_words": 3,
    "negative_words": 0
  }
  ```

#### 4.2 Emotion Detection
- **Module**: `EmotionDetector` (`backend/src/call_analysis/models.py`)
- **Model**: Rule-based + Audio features (CNN+LSTM architecture available)
- **Emotions**: neutral, happy, sad, angry, fearful, disgusted, surprised
- **Process**:
  1. Analyzes audio features (spectral centroid, MFCC, zero crossing rate)
  2. Uses heuristics to determine emotion probabilities
  3. Assigns dominant emotion per segment
- **Output per segment**:
  ```json
  {
    "emotion": "happy",
    "confidence": 0.75,
    "probabilities": {
      "neutral": 0.1,
      "happy": 0.75,
      "sad": 0.05,
      ...
    }
  }
  ```

#### 4.3 Sale Probability Prediction
- **Module**: `SalePredictor` (`backend/src/call_analysis/models.py`)
- **Model**: XGBoost (or LSTM for demo)
- **Process**:
  1. Takes fused feature vector as input
  2. Uses trained XGBoost classifier (or demo heuristics)
  3. Predicts probability of sale (0-1)
  4. Returns prediction with confidence
- **Output**:
  ```json
  {
    "sale_probability": 0.72,
    "prediction": "sale",
    "confidence": 0.44,
    "feature_importance": [...]
  }
  ```

#### 4.4 Conversation-Level Analysis
- **Module**: `ConversationAnalyzer.analyze_conversation()`
- **Process**:
  1. Runs sentiment analysis on all segments
  2. Runs emotion detection on all segments
  3. Calculates conversation-level metrics:
     - Average sentiment
     - Sentiment trend (increasing/decreasing)
     - Dominant emotion
     - Emotion distribution
     - Customer vs Agent sentiment
     - Sentiment volatility
     - Conversation flow score
  4. Generates natural language summary
- **Output**: Complete analysis results dictionary

---

### **Phase 5: Results Storage & Retrieval**

#### 5.1 Database Storage
- **Database**: MongoDB (primary), PostgreSQL (optional)
- **Collections**:
  - `calls`: Main call records with analysis results
  - `transcriptions`: Transcription data
  - `diarization`: Speaker diarization segments
  - `segments`: Processed conversation segments
  - `features`: Extracted features
  - `analyses`: Analysis results (backward compatibility)
  - `insights`: Agent insights and recommendations

#### 5.2 File Storage
- **Directory**: `output/`
- **Files Generated**:
  - `{call_id}_transcription.json`: Transcription results
  - `{call_id}_diarization.json`: Diarization segments
  - `{call_id}_results.json`: Complete analysis results
  - `{call_id}_dashboard.html`: Interactive HTML dashboard (optional)

#### 5.3 API Endpoints for Results
- **GET `/api/results/{call_id}`**: Get analysis results
- **GET `/api/status/{call_id}`**: Get analysis status
- **GET `/api/history`**: Get call history
- **GET `/api/export/{call_id}`**: Export JSON
- **GET `/api/export/{call_id}/pdf`**: Export PDF report
- **GET `/api/export/{call_id}/csv`**: Export CSV data

---

### **Phase 6: Frontend Visualization**

#### 6.1 Status Polling (`frontend/src/app/page.tsx`)
- **Process**:
  1. After upload, frontend polls `/api/status/{call_id}` every 2 seconds
  2. Displays progress bar and status
  3. When status = "completed", fetches results

#### 6.2 Results Display (`frontend/src/components/AnalysisDashboard.tsx`)
- **Components**:
  - **SaleGauge**: Circular gauge showing sale probability (0-100%)
  - **SentimentChart**: Line chart showing sentiment over time
  - **EmotionChart**: Bar/pie chart showing emotion distribution
  - **KeyPhrases**: Lists of positive/negative key phrases
  - **Summary Cards**: Average sentiment, duration, participants

#### 6.3 History View (`frontend/src/components/HistorySection.tsx`)
- **Features**:
  - Lists all previous calls
  - Shows call metadata (date, duration, sale probability)
  - Links to detailed results pages

---

## 🔄 Alternative Execution Paths

### **Path 1: Full Analysis Script** (`backend/run_full_analysis.py`)
- **Usage**: Command-line script for batch processing
- **Command**: `python run_full_analysis.py audio_file.wav --call-id custom_id --output-dir output/`
- **Process**: Runs complete pipeline (Steps 2-5) without web interface
- **Output**: JSON files and HTML dashboard in output directory

### **Path 2: Diarization-Only Script** (`backend/run_diarization_only.py`)
- **Usage**: Test diarization with different parameters
- **Command**: `python run_diarization_only.py audio.wav transcription.json --max-speakers 2`
- **Process**: Only runs diarization step (useful for parameter tuning)
- **Output**: Diarization JSON with summary

### **Path 3: Demo Mode** (`backend/src/call_analysis/demo.py`)
- **Usage**: Pre-configured demo conversations
- **Endpoints**: 
  - `GET /api/conversations`: List demo conversations
  - `GET /api/analyze/{id}`: Analyze specific demo
  - `GET /api/analyze-all`: Analyze all demos
- **Process**: Uses pre-generated conversation data (no audio processing)

---

## 📋 Pipeline Steps Summary

| Step | Module | Input | Output | Time Estimate |
|------|--------|-------|--------|---------------|
| 1. Upload | `UploadForm.tsx` | Audio file | `call_id` | <1s |
| 2.1 Format Validation | `AudioProcessor` | Audio file | Validated path | <1s |
| 2.2 Transcription | `AudioProcessor` | Audio file | Transcription JSON | 1-5 min |
| 2.3 Diarization | `AudioProcessor` | Audio file | Diarization segments | 5-40 min* |
| 2.4 Text Processing | `TextProcessor` | Transcription + Diarization | Processed segments | <1s |
| 3.1 Audio Features | `AudioProcessor` | Audio file | Audio features dict | <1s |
| 3.2 Text Features | `FeatureExtractor` | Text | BERT embeddings | 1-2s |
| 3.3 Temporal Features | `FeatureExtractor` | Segments | Temporal features | <1s |
| 3.4 Feature Fusion | `FeatureExtractor` | All features | Fused vector | <1s |
| 4.1 Sentiment | `SentimentAnalyzer` | Segments | Sentiment scores | 2-5s |
| 4.2 Emotion | `EmotionDetector` | Audio features | Emotion labels | <1s |
| 4.3 Sale Prediction | `SalePredictor` | Fused features | Sale probability | <1s |
| 4.4 Aggregation | `ConversationAnalyzer` | All results | Complete analysis | <1s |
| 5. Storage | MongoDB/File | Analysis results | Stored data | <1s |
| 6. Visualization | `AnalysisDashboard` | Results JSON | UI components | <1s |

*Diarization time varies significantly:
- WhisperX + Resemblyzer: 1-5 minutes (fast)
- WhisperX 3.x built-in: 20-40 minutes (accurate)
- Pyannote.audio: 15-30 minutes (fallback)

---

## 🔧 Configuration

### Environment Variables (`.env`)
```bash
# Required
HF_TOKEN=your_huggingface_token_here

# Database
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=call_center_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=call_analysis

# FastAPI
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Models
WHISPER_MODEL_SIZE=base
BERT_MODEL_NAME=distilbert-base-uncased
PYANNOTE_AUDIO_MODEL=pyannote/speaker-diarization

# File Upload
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=100MB
ALLOWED_EXTENSIONS=wav,mp3,m4a,flac

# Security
PII_MASKING_ENABLED=True
DATA_RETENTION_DAYS=30

# Performance
BATCH_SIZE=10
MAX_WORKERS=4
```

---

## 🚀 Quick Start Commands

### Start Backend
```bash
cd backend
python run_web_app.py
# Server runs on http://localhost:8000
```

### Start Frontend
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:3000
```

### Run Full Analysis (CLI)
```bash
cd backend
python run_full_analysis.py path/to/audio.wav
```

### Run Diarization Only (Testing)
```bash
cd backend
python run_diarization_only.py audio.wav transcription.json --max-speakers 2
```

---

## 📊 Data Flow Diagram

```
Audio File (.wav/.mp3/.m4a)
    │
    ▼
[Format Validation] ──► Validated Audio
    │
    ├──► [Whisper Transcription] ──► Transcription JSON
    │                                    │
    │                                    ▼
    │                            [Text Processing] ──► Processed Segments
    │                                    │
    │                                    ▼
    └──► [Speaker Diarization] ──► Diarization Segments ──► [Text-to-Segment Mapping]
    │                                    │
    │                                    ▼
    └──► [Audio Feature Extraction] ──► Audio Features ──► [Feature Fusion] ──► Fused Features
    │                                    │
    │                                    ▼
    └──► [BERT Text Features] ──► Text Embeddings ──► [ML Models]
    │                                    │
    │                                    ├──► [Sentiment Analysis] ──► Sentiment Scores
    │                                    ├──► [Emotion Detection] ──► Emotion Labels
    │                                    └──► [Sale Prediction] ──► Sale Probability
    │
    ▼
[Conversation Analyzer] ──► Complete Analysis Results
    │
    ├──► MongoDB Storage
    ├──► File Storage (JSON)
    └──► HTML Dashboard (optional)
    │
    ▼
[Frontend API] ──► Results JSON
    │
    ▼
[React Components] ──► Interactive Dashboard
```

---

## 🎯 Key Features

1. **Multimodal Analysis**: Combines audio and text features
2. **Real-time Processing**: Web-based interface with status updates
3. **Speaker Identification**: Automatic AGENT vs CUSTOMER detection
4. **PII Masking**: Security feature to mask sensitive information
5. **Scalable Architecture**: Supports batch processing and API integration
6. **Multiple Export Formats**: JSON, PDF, CSV
7. **Interactive Visualization**: Charts, gauges, and timelines

---

## 📝 Notes

- **Diarization is REQUIRED**: The system cannot proceed without speaker diarization
- **Chunking Support**: Long audio files (>5 min) are automatically chunked for faster processing
- **Fallback Mechanisms**: Multiple diarization methods with automatic fallback
- **Demo Mode**: Can run with simulated data for testing without audio files
- **PII Masking**: Automatically masks phone numbers, emails, names for security compliance

---

## 🔍 Troubleshooting

### Common Issues:
1. **Diarization fails**: Check `HF_TOKEN` is set and valid
2. **Transcription empty**: Verify audio file has speech content
3. **Slow processing**: Use WhisperX + Resemblyzer for faster processing
4. **Memory errors**: Reduce `WHISPER_MODEL_SIZE` or enable chunking
5. **Database connection**: Check MongoDB/PostgreSQL is running

---

**Last Updated**: December 2025
**Version**: 1.0.0
