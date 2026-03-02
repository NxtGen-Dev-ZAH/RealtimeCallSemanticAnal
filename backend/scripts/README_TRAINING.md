# Training Scripts Documentation

This directory contains training scripts for the Call Analysis System models.

## Scripts Overview

### 1. `best_train_emotion_model.py` (Recommended)
Trains the Wav2Vec2-based emotion model for emotion recognition using RAVDESS dataset format.

**Usage:**
```bash
python backend/scripts/best_train_emotion_model.py --mode train --data_dir data/raw/ravdess --output_dir backend/models/best_emotion_wav2vec2_v2 --epochs 30 --batch_size 8
```

**Arguments:**
- `--data_dir`: Path to RAVDESS dataset directory (default: `data/raw/`)
- `--output_dir`: Output directory for model (default: `backend/models/`)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--validation_split`: Validation split ratio (default: 0.2)
- `--max_time_frames`: Maximum time frames for padding/truncation (default: 500)

**Output:**
- `backend/models/best_emotion_wav2vec2_v2/best_checkpoint/` - Trained Wav2Vec2 checkpoint
- `backend/models/best_emotion_wav2vec2_v2/best_emotion_training_history.json` - Training history

### 1.1 `train_emotion_model.py` (Legacy)
Legacy CNN+LSTM training pipeline that outputs `backend/models/emotion_model.pth`.

**Dataset Format:**
- RAVDESS format: Filenames like `03-01-01-01-01-01-01.wav`
- Emotion mapping: `01=neutral, 02=calm→neutral, 03=happiness, 04=sadness, 05=anger, 06/07=frustration`
- Supports recursive directory scanning for .wav files

### 2. `train_sale_predictor.py`
Trains the SalePredictor (XGBoost) for sale probability prediction using CSV with fused feature vectors.

**Usage:**
```bash
python backend/scripts/train_sale_predictor.py --csv_path data/training_data.csv --output_dir backend/models/
```

**Arguments:**
- `--csv_path`: Path to CSV file with fused feature vectors (required)
- `--output_dir`: Output directory for model (default: `backend/models/`)
- `--test_split`: Test split ratio (default: 0.15)
- `--validation_split`: Validation split ratio (default: 0.15)
- `--n_estimators`: XGBoost n_estimators (default: 100)
- `--max_depth`: XGBoost max_depth (default: 6)
- `--learning_rate`: XGBoost learning_rate (default: 0.1)

**CSV Format:**
Required columns:
- `sentiment_mean`, `sentiment_variance` - Textual sentiment features
- `emotion_neutral`, `emotion_happiness`, `emotion_anger`, `emotion_sadness`, `emotion_frustration` - Emotion probabilities
- `silence_ratio`, `interruption_frequency`, `talk_listen_ratio`, `turn_taking_frequency` - Conversational dynamics
- `sale_outcome` - Binary label (0=no_sale, 1=sale)

**Output:**
- `backend/models/sale_model.pkl` - Trained XGBoost model
- `backend/models/sale_model_scaler.pkl` - Feature scaler (needed for inference)
- `backend/models/sale_model_feature_importance.png` - Feature importance visualization
- `backend/models/sale_model_roc_curve.png` - ROC curve plot
- `backend/models/sale_training_results.json` - Training metrics and results

### 3. `validate_trained_models.py`
Validates that trained models can be loaded and used correctly.

**Usage:**
```bash
python backend/scripts/validate_trained_models.py
```

**What it does:**
- Tests loading emotion artifact (`EMOTION_MODEL_PATH`, `best_emotion_wav2vec2_v2/best_checkpoint`, or legacy `emotion_model.pth`) with `EmotionDetector`
- Tests loading `sale_model.pkl` with `SalePredictor`
- Validates model inference with dummy data
- Reports validation status for each model

## Model Integration

After training, models are automatically loaded by:
- `EmotionDetector(model_path='backend/models/best_emotion_wav2vec2_v2/best_checkpoint')` - Loads recommended emotion checkpoint
- `SalePredictor(model_path='backend/models/sale_model.pkl')` - Loads sale prediction model

The models are used in `ConversationAnalyzer` for real-time inference.

## Notes

- **SentimentAnalyzer**: Uses pre-trained Hugging Face model (`distilbert-base-uncased-finetuned-sst-2-english` or `finbert`), no training needed. If the HF model is unavailable (deleted, renamed, or network down), the app falls back to **keyword-based sentiment** automatically so it keeps running.
- **RAVDESS Dataset**: Download from https://zenodo.org/record/1188976 and organize in `data/raw/`
- **Sale Training Data**: Prepare CSV with fused feature vectors extracted from real call data using `SalePredictor.create_fused_feature_vector()`
- **GPU Support**: Training scripts automatically use GPU if available (CUDA)

### Sentiment model: Hugging Face links

- **Default (DistilBERT SST-2)**:  
  https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english  
- **Alternative (FinBERT)**:  
  https://huggingface.co/ProsusAI/finbert  

### How to download the sentiment model locally

1. **Choose a folder** where the model will live (e.g. `backend/models/sentiment_distilbert` or `backend/models/sentiment_finbert`).

2. **Download and save the model** with Python (run from project root, with the backend venv activated):

   **DistilBERT (default):**
   ```bash
   cd backend
   mkdir -p models/sentiment_distilbert
   uv run python -c "
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   name = 'distilbert-base-uncased-finetuned-sst-2-english'
   tok = AutoTokenizer.from_pretrained(name)
   model = AutoModelForSequenceClassification.from_pretrained(name)
   tok.save_pretrained('models/sentiment_distilbert')
   model.save_pretrained('models/sentiment_distilbert')
   print('Saved to models/sentiment_distilbert')
   "
   ```

   **FinBERT:**
   ```bash
   uv run python -c "
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   name = 'ProsusAI/finbert'
   tok = AutoTokenizer.from_pretrained(name)
   model = AutoModelForSequenceClassification.from_pretrained(name)
   tok.save_pretrained('models/sentiment_finbert')
   model.save_pretrained('models/sentiment_finbert')
   print('Saved to models/sentiment_finbert')
   "
   ```

   Create the target directory first if it doesn’t exist (e.g. `mkdir backend\models\sentiment_distilbert` on Windows, or `mkdir -p backend/models/sentiment_distilbert` on macOS/Linux).

3. **Point the app at the local copy** using an environment variable (or `.env`):

   - For DistilBERT:  
     `SENTIMENT_MODEL_PATH=backend/models/sentiment_distilbert`  
   - For FinBERT:  
     `SENTIMENT_MODEL_PATH=backend/models/sentiment_finbert`  
     and keep `SENTIMENT_MODEL=finbert` if you use the FinBERT path.

   Use an absolute path if you run the app from a different working directory (e.g. `D:\RealtimeCallSemanticAnal\backend\models\sentiment_distilbert` on Windows).

4. **Optional:** Set `SENTIMENT_LOCAL_FILES_ONLY=true` so the app never tries to reach Hugging Face at runtime.

After this, the app loads the sentiment model from disk and no longer needs the Hugging Face model to be available online.

### Sentiment model robustness (if Hugging Face model is deleted or changed)

- **Fallback**: On load or inference failure, sentiment uses a built-in keyword-based scorer so the system does not crash.
- **Local copy (recommended for production)**: Use the steps in “How to download the sentiment model locally” above, then set `SENTIMENT_MODEL_PATH` to that directory.
- **Pin version**: Set `SENTIMENT_MODEL_REVISION` to a specific commit hash from the model repo to avoid surprises when the repo is updated.
- **Offline-only**: After the model is cached or saved locally, set `SENTIMENT_LOCAL_FILES_ONLY=true` so that only the cache/path is used and no network call is made.

## Example Workflow

1. **Prepare Emotion Dataset:**
   ```bash
   # Download RAVDESS dataset to data/raw/
   # Organize files in subdirectories (optional)
   ```

2. **Train Emotion Model:**
   ```bash
   python backend/scripts/best_train_emotion_model.py --mode train --data_dir data/raw/ravdess --output_dir backend/models/best_emotion_wav2vec2_v2 --epochs 30 --batch_size 8
   ```

3. **Prepare Sale Training Data:**
   ```python
   # Extract features from call recordings
   # Create CSV with columns: sentiment_mean, sentiment_variance, emotion_*, dynamics_*, sale_outcome
   ```

4. **Train Sale Predictor:**
   ```bash
   python backend/scripts/train_sale_predictor.py --csv_path data/sale_training_data.csv
   ```

5. **Validate Models:**
   ```bash
   python backend/scripts/validate_trained_models.py
   ```

## Troubleshooting

- **Import Errors**: Ensure you're running from the project root or have PYTHONPATH set correctly
- **CUDA Out of Memory**: Reduce `--batch_size` in emotion training
- **Missing Columns**: Verify CSV format matches expected feature names
- **Model Loading Errors**: Check that model files exist in `backend/models/` directory

