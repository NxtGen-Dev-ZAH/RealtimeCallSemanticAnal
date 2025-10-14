# Configuration Guide

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```bash
# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here

# Database Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
MONGODB_DATABASE=call_center_db

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Model Configuration
WHISPER_MODEL_SIZE=base
BERT_MODEL_NAME=distilbert-base-uncased
PYANNOTE_AUDIO_MODEL=pyannote/speaker-diarization

# File Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=104857600
ALLOWED_EXTENSIONS=wav,mp3,m4a,flac

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/call_analysis.log

# Security Configuration
PII_MASKING_ENABLED=True
DATA_RETENTION_DAYS=30

# Performance Configuration
BATCH_SIZE=10
MAX_WORKERS=4
CACHE_TTL=3600

# Demo Configuration
DEMO_MODE=True
USE_SIMULATED_AUDIO=True
```

## Setup Instructions

1. Copy this configuration to a `.env` file in the backend directory
2. Replace placeholder values with your actual credentials
3. For Hugging Face token, visit https://hf.co/settings/tokens
4. For MongoDB, use your Atlas connection string

## Optional Features

- **spaCy Model**: Install with `python -m spacy download en_core_web_sm` for better PII masking
- **PyAnnote Model**: Requires Hugging Face token for speaker diarization
