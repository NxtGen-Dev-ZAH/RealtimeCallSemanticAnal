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

# LLM Configuration for Diarization Enhancement
USE_LLM_DIARIZATION=True
LLM_ROLE_IDENTIFICATION_MODEL=facebook/bart-large-mnli
LLM_REFINEMENT_MODEL=google/flan-t5-base
LLM_DEVICE=cpu

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
- **LLM Diarization Enhancement**: Uses Hugging Face models for improved role identification and error correction
  - `USE_LLM_DIARIZATION`: Enable/disable LLM-enhanced diarization (default: True)
  - `LLM_ROLE_IDENTIFICATION_MODEL`: Model for identifying agent vs customer roles (default: facebook/bart-large-mnli)
  - `LLM_REFINEMENT_MODEL`: Model for post-processing refinement (default: distilgpt2)
  - `LLM_DEVICE`: Device to run LLM models on ('cpu' or 'cuda', default: 'cpu')
  
  **Important**: Both LLM models are PUBLIC models and do NOT require a Hugging Face token.
  They will download automatically on first use.
  
  However, if you want to use gated models or avoid rate limits, you can set `HF_TOKEN`
  (same token used for Pyannote models). The token is optional for LLM diarization.
  
  **Hybrid Approach**:
  - **BART-MNLI** (`facebook/bart-large-mnli`): Zero-shot classification for role identification
    - Detects if speaker is AGENT or CUSTOMER
    - More accurate than heuristics for role identification
  - **FLAN-T5** (`google/flan-t5-base`): Text-to-text model for correction/refinement
    - Fixes wrong segments
    - Merges incorrect splits
    - Recovers short segments that were mis-assigned
  
  **Model Options**:
  - BART-MNLI: `facebook/bart-large-mnli` (recommended) or `facebook/bart-large-mnli-128`
  - FLAN-T5: `google/flan-t5-small` (fastest), `google/flan-t5-base` (recommended), `google/flan-t5-large` (more accurate)

### LLM Diarization Features (Hybrid Approach)

The system uses a hybrid approach combining multiple models:

1. **Resemblyzer**: Fast speaker embedding extraction (CPU-friendly, 10x faster than Pyannote)
2. **WhisperX**: Accurate word-level timestamps for transcription
3. **BART-MNLI**: Zero-shot classification to detect if speaker is AGENT or CUSTOMER
   - More accurate than keyword-based heuristics
   - Uses semantic understanding of conversation context
4. **FLAN-T5**: Text-to-text correction to fix diarization errors
   - Fixes wrong segments
   - Merges incorrect splits
   - Recovers short segments that were mis-assigned

**Pipeline Flow**:
```
Audio → Resemblyzer (speaker separation) → WhisperX (timestamps) 
→ BART-MNLI (role identification) → FLAN-T5 (correction) → Final Results
```

These features are enabled by default and will automatically fall back to heuristic methods if LLM models fail to load.
