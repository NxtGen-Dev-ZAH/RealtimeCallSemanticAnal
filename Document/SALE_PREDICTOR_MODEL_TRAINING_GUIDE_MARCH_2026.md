# Sale Predictor Model Training Guide (March 2026)

This guide is for training the sale prediction model used by this project:
- Model: XGBoost-based `SalePredictor`
- Training script: `backend/scripts/train_sale_predictor.py`
- Output model file: `backend/models/sale_model.pkl`

Date context:
- Written for repository state on March 2, 2026.

## 1. What this model does

The model predicts probability of sale (`sale_outcome` = 0 or 1) from fused conversation features.

Required training features are exactly these 11 columns:
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

Required label column:
- `sale_outcome` (binary: `0` no sale, `1` sale)

## 2. Prerequisites

From repository root `D:\RealtimeCallSemanticAnal`:

1. Verify Python dependencies:
```powershell
cd D:\RealtimeCallSemanticAnal
python -c "import xgboost, sklearn, pandas, numpy, joblib, matplotlib; print('deps ok')"
```

2. Ensure output folder:
```powershell
New-Item -ItemType Directory -Force backend\models | Out-Null
```

## 3. How to find and prepare training data

This is the most important part. You have 3 practical options.

## Option A (recommended): Build dataset from your own calls + CRM outcomes

Use real call analysis outputs plus actual business result labels.

### A1. Required input assets
- Analyzed calls in MongoDB (`calls` collection, `status=completed`).
- External label file from CRM, e.g. `data/sale_labels.csv` with:
  - `call_id`
  - `sale_outcome` (0/1)

### A2. Build training CSV from MongoDB + labels

Create `data/sale_labels.csv` first, then run:

```powershell
cd D:\RealtimeCallSemanticAnal
python - <<'PY'
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient

mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongo_db = os.getenv("MONGODB_DATABASE", "call_center_db")

labels = pd.read_csv("data/sale_labels.csv")  # columns: call_id, sale_outcome
labels["call_id"] = labels["call_id"].astype(str)

client = MongoClient(mongo_uri)
db = client[mongo_db]

rows = []
cursor = db.calls.find(
    {"status": "completed"},
    {
        "call_id": 1,
        "avg_sentiment": 1,
        "sentiment_scores": 1,
        "emotions": 1,
        "result.conversational_dynamics": 1,
    },
)

for doc in cursor:
    call_id = str(doc.get("call_id", ""))
    if not call_id:
        continue

    sentiment_scores = [float(x.get("score", 0.0)) for x in doc.get("sentiment_scores", []) if isinstance(x, dict)]
    sentiment_mean = float(doc.get("avg_sentiment", np.mean(sentiment_scores) if sentiment_scores else 0.0))
    sentiment_variance = float(np.var(sentiment_scores)) if sentiment_scores else 0.0

    emotions = doc.get("emotions", {}) or {}
    dynamics = ((doc.get("result") or {}).get("conversational_dynamics") or {})

    rows.append({
        "call_id": call_id,
        "sentiment_mean": sentiment_mean,
        "sentiment_variance": sentiment_variance,
        "emotion_neutral": float(emotions.get("neutral", 0.0)),
        "emotion_happiness": float(emotions.get("happiness", 0.0)),
        "emotion_anger": float(emotions.get("anger", 0.0)),
        "emotion_sadness": float(emotions.get("sadness", 0.0)),
        "emotion_frustration": float(emotions.get("frustration", 0.0)),
        "silence_ratio": float(dynamics.get("silence_ratio", 0.0)),
        "interruption_frequency": float(dynamics.get("interruption_frequency", 0.0)),
        "talk_listen_ratio": float(dynamics.get("talk_listen_ratio", 1.0)),
        "turn_taking_frequency": float(dynamics.get("turn_taking_frequency", 0.0)),
    })

feature_df = pd.DataFrame(rows)
train_df = feature_df.merge(labels, on="call_id", how="inner")

# Keep only exact columns required by training script
required_cols = [
    "sentiment_mean", "sentiment_variance",
    "emotion_neutral", "emotion_happiness", "emotion_anger", "emotion_sadness", "emotion_frustration",
    "silence_ratio", "interruption_frequency", "talk_listen_ratio", "turn_taking_frequency","sale_outcome"
]
train_df = train_df[required_cols].dropna()
train_df["sale_outcome"] = train_df["sale_outcome"].astype(int)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/sale_training_data.csv", index=False)
print("saved rows:", len(train_df))
print("label distribution:", train_df["sale_outcome"].value_counts().to_dict())
PY
```

### A3. Data quality checks

```powershell
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/sale_training_data.csv")
print("rows:", len(df))
print("columns:", list(df.columns))
print("null counts:", df.isnull().sum().to_dict())
print("label distribution:", df['sale_outcome'].value_counts(normalize=True).to_dict())
PY
```

Recommended minimum for stable training:
- At least 500 labeled calls.
- Positive class ideally >= 15% (script also handles imbalance with `scale_pos_weight`).

## Option B: Use repository mock data for smoke test

Fast pipeline test only (not production quality):
- Existing generated files in this repo state: `data/sale_training_data.csv` (recommended) and `data/synthetic_sales_full.csv`

This lets you verify training end-to-end before collecting real labeled data.

## Option C: Use public proxy business datasets (for experimentation only)

If you need a public baseline:
- UCI Bank Marketing dataset can be used as a proxy classification dataset.
- But it is not in this project's fused-feature schema, so you must engineer/match features first.

For project demo, Option A is the correct path.

## 4. Train the sale predictor model

Basic run:
```powershell
cd D:\RealtimeCallSemanticAnal
python backend\scripts\train_sale_predictor.py `
  --csv_path data\sale_training_data.csv `
  --output_dir backend\models `
  --eval_metric auc `
  --early_stopping_rounds 20 `
  --optimize_threshold
```

Suggested stronger configuration:
```powershell
python backend\scripts\train_sale_predictor.py `
  --csv_path data\sale_training_data.csv `
  --output_dir backend\models `
  --n_estimators 300 `
  --max_depth 5 `
  --learning_rate 0.05 `
  --eval_metric auc `
  --subsample 0.9 `
  --colsample_bytree 0.8 `
  --min_child_weight 3 `
  --early_stopping_rounds 30 `
  --optimize_threshold
```

If your CSV has many missing values and you want explicit scaling too:
```powershell
python backend\scripts\train_sale_predictor.py --csv_path data\sale_training_data.csv --output_dir backend\models --use_scaling --eval_metric auc --optimize_threshold
```

If you already generated data with `data_split` or `call_timestamp`, you can use:
```powershell
# Use data_split column
python backend\scripts\train_sale_predictor.py --csv_path data\sale_training_data.csv --output_dir backend\models --use_data_split --eval_metric auc --optimize_threshold

# Use chronological split (call_timestamp)
python backend\scripts\train_sale_predictor.py --csv_path data\sale_training_data.csv --output_dir backend\models --time_split --eval_metric auc --optimize_threshold
```

If your generated dataset is already available at `data/sale_training_data.csv`, you can directly run the training commands above without rebuilding the CSV.

## 5. Outputs you should see

In `backend/models/` after successful training:
- `sale_model.pkl` (required)
- `sale_training_results.json` (required)
- `sale_model_feature_importance.png`
- `sale_model_roc_curve.png`
- `sale_model_imputer.pkl` (only if missing values were detected)
- `sale_model_scaler.pkl` (only if `--use_scaling` was used)

## 6. Validate trained model

```powershell
cd D:\RealtimeCallSemanticAnal
python backend\scripts\validate_trained_models.py
```

Expected:
- Sale model validation passes.
- Emotion model may fail if not yet trained (train emotion model first if needed).

Optional direct check:
```powershell
python - <<'PY'
from backend.src.call_analysis.models import SalePredictor
p = SalePredictor(model_path="backend/models/sale_model.pkl")
print("is_trained=", p.is_trained, "threshold=", p.threshold)
PY
```

## 7. How this script handles data and imbalance

Your script already includes:
- Train/validation/test split with stratification.
- Leakage prevention: split first, then impute/scale.
- Class imbalance handling via `scale_pos_weight`.
- Early stopping.
- Platt calibration (`CalibratedClassifierCV`).
- Optional threshold optimization (`--optimize_threshold`).
 - Optional split controls: `--use_data_split` or `--time_split`.
 - Additional XGBoost controls: `--eval_metric`, `--subsample`, `--colsample_bytree`, `--min_child_weight`.

So your main job is data quality and correct labels.

## 8. Troubleshooting

1. `CSV missing required columns`
- Ensure exact column names listed in Section 1.

2. `Only one label class present` or very poor metrics
- Ensure both labels `0` and `1` exist.
- Improve label quality from CRM.

3. `xgboost` import/version issues
- Reinstall environment dependencies and rerun.

4. Inference mismatch after training
- Keep model sidecar files (`sale_training_results.json`, optional scaler/imputer) in same folder as `sale_model.pkl`.

5. Good validation but bad real-world behavior
- Check label leakage.
- Verify feature extraction consistency between training CSV and runtime pipeline.

## 9. Minimum acceptance checklist

- [ ] `backend/models/sale_model.pkl` exists
- [ ] `backend/models/sale_training_results.json` exists
- [ ] `python backend/scripts/validate_trained_models.py` passes for sale model
- [ ] Backend startup no longer errors on missing sale model

## 10. Training order recommendation

Use this order for clean project setup:
1. Train emotion model first (`EMOTION_MODEL_TRAINING_GUIDE_MARCH_2026.md`)
2. Run/analyze calls to generate reliable emotion-related features
3. Build labeled sale CSV
4. Train sale predictor
5. Validate both models
