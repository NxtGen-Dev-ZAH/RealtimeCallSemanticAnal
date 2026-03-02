# Emotion Model Training Guide (March 2026)

This guide updates emotion-model training to the best-performing architecture choice for RAVDESS in this repo:
- New training script: `backend/scripts/best_train_emotion_model.py`
- Architecture: `Wav2Vec2ForSequenceClassification` (self-supervised speech transformer backbone + classification head)
- Main output: `backend/models/best_emotion_wav2vec2_v2/best_checkpoint/`

Date context:
- Updated on March 2, 2026 for repository `D:\RealtimeCallSemanticAnal`.

## 1. Why this is the best model direction

Short answer: transformer SSL speech backbones (Wav2Vec2/WavLM family) consistently beat older handcrafted-feature pipelines for SER.

Research notes that informed this update:
- `Emotion Recognition from Speech Using wav2vec 2.0 Embeddings` (Interspeech 2021) reports strong RAVDESS performance and shows wav2vec2 embeddings outperform earlier feature sets.
- More recent RAVDESS papers report very high accuracy for wav2vec2-large and related SSL models, but many use random 80/20 splits.
- Some papers show even higher numbers (for example Xception-based pipelines), also usually with random splits and augmentation-heavy protocols.

Important comparability rule:
- Random file-level split can inflate RAVDESS results.
- This script defaults to actor-holdout (`train actors 01-18`, `validation actors 19-24`) to reduce speaker leakage.

## 1.1 Project-scope alignment (call-center sale prediction demo)

Your platform goal is not just emotion accuracy, it is reliable sale-likelihood prediction from full call analysis.
For that reason, emotion training and deployment must satisfy all of these:
- segment-level emotion inference works on diarized call segments
- emotion output labels remain compatible with sale-feature fusion (`neutral, happiness, anger, sadness, frustration`)
- model artifact can be loaded by backend during live demo without code changes at runtime

This repo now supports both emotion model formats:
- Legacy: `backend/models/emotion_model.pth` (CNN+LSTM)
- Best model: `backend/models/best_emotion_wav2vec2_v2/best_checkpoint` (Wav2Vec2 checkpoint)

Model selection order at runtime:
1. `EMOTION_MODEL_PATH` env var (if set)
2. `backend/models/emotion_model.pth` (if exists)
3. `backend/models/best_emotion_wav2vec2_v2/best_checkpoint` (recommended checkpoint path)

## 2. Prerequisites (Windows, exact repo commands)

Run in PowerShell.

1. Go to repo root:
```powershell
cd D:\RealtimeCallSemanticAnal
```

2. Create and activate virtual environment (recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install backend dependencies from this repo:
```powershell
cd D:\RealtimeCallSemanticAnal\backend
python -m pip install --upgrade pip
python -m pip install -e .
```

4. Verify required packages:
```powershell
cd D:\RealtimeCallSemanticAnal
python -c "import torch, transformers, librosa, soundfile, sklearn, numpy; print('deps ok')"
```

5. Optional GPU check:
```powershell
python -c "import torch; print('cuda_available=', torch.cuda.is_available())"
```

6. Create model output folder:
```powershell
New-Item -ItemType Directory -Force backend\models\best_emotion_wav2vec2_v2 | Out-Null
```

## 3. Where to get data (RAVDESS)

Primary source:
- Zenodo record: `https://zenodo.org/records/1188976`
- Recommended file for this pipeline: `Audio_Speech_Actors_01-24.zip`

Expected speech file count:
- `1440` `.wav` files.

Download + extract:
```powershell
cd D:\RealtimeCallSemanticAnal
New-Item -ItemType Directory -Force data\raw\ravdess | Out-Null

Invoke-WebRequest `
  -Uri "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1" `
  -OutFile "data\raw\Audio_Speech_Actors_01-24.zip"

Expand-Archive -Path "data\raw\Audio_Speech_Actors_01-24.zip" -DestinationPath "data\raw\ravdess" -Force
```

Quick file-count check:
```powershell
(Get-ChildItem data\raw\ravdess -Recurse -Filter *.wav | Measure-Object).Count
```

## 4. Data validation checks

Run script-level validation first:
```powershell
cd D:\RealtimeCallSemanticAnal
python backend\scripts\best_train_emotion_model.py `
  --mode validate_data `
  --data_dir data\raw\ravdess `
  --output_dir backend\models\best_emotion_wav2vec2_v2 `
  --emotion_mapping default
```

This generates:
- `backend/models/best_emotion_wav2vec2_v2/data_validation_report.json`

What to verify in that JSON:
- `dataset_scan.valid_samples` should match expected subset size (usually `1440` speech files)
- `integrity_report.corrupted_files_count` should be `0`
- `integrity_report.actor_count` should be `24`
- `integrity_report.label_counts` should show all target classes

Optional manual checks:
```powershell
Get-ChildItem data\raw\ravdess -Recurse -Filter *.wav | Select-Object -First 5 -ExpandProperty Name
python -c "import glob,librosa;f=glob.glob(r'data/raw/ravdess/**/*.wav',recursive=True);y,sr=librosa.load(f[0],sr=None);print('sample=',f[0],'sr=',sr,'dur=',len(y)/sr)"
```

## 5. Full training commands

### Recommended baseline (best default for this repo)

5-class mapping, actor-holdout split, Wav2Vec2-base:
```powershell
cd D:\RealtimeCallSemanticAnal
python backend\scripts\best_train_emotion_model.py `
  --mode train `
  --data_dir data\raw\ravdess `
  --output_dir backend\models\best_emotion_wav2vec2_v2 `
  --base_model facebook/wav2vec2-base `
  --emotion_mapping default `
  --split_strategy actor_holdout `
  --train_actor_max 18 `
  --epochs 30 `
  --batch_size 8 `
  --learning_rate 2e-5 `
  --weight_decay 0.01 `
  --warmup_ratio 0.1 `
  --max_seconds 6.0
```

### Variant A: stronger model (more VRAM)

```powershell
python backend\scripts\best_train_emotion_model.py `
  --mode train `
  --data_dir data\raw\ravdess `
  --output_dir backend\models\best_emotion_wav2vec2_v2_large `
  --base_model facebook/wav2vec2-large-960h-lv60-self `
  --emotion_mapping default `
  --split_strategy actor_holdout `
  --train_actor_max 18 `
  --epochs 25 `
  --batch_size 2 `
  --learning_rate 1e-5 `
  --weight_decay 0.01
```

### Variant B: CPU-safe run

```powershell
python backend\scripts\best_train_emotion_model.py `
  --mode train `
  --data_dir data\raw\ravdess `
  --output_dir backend\models\best_emotion_wav2vec2_v2_cpu `
  --base_model facebook/wav2vec2-base `
  --emotion_mapping default `
  --split_strategy actor_holdout `
  --train_actor_max 18 `
  --epochs 15 `
  --batch_size 2 `
  --learning_rate 2e-5 `
  --num_workers 0 `
  --no_amp
```

### Variant C: strict 8-class RAVDESS labels

```powershell
python backend\scripts\best_train_emotion_model.py `
  --mode train `
  --data_dir data\raw\ravdess `
  --output_dir backend\models\best_emotion_wav2vec2_v2_8class `
  --base_model facebook/wav2vec2-base `
  --emotion_mapping strict `
  --split_strategy actor_holdout `
  --train_actor_max 18 `
  --epochs 30 `
  --batch_size 8 `
  --learning_rate 2e-5
```

Use `strict` only for research experiments.
For your call-center sale-prediction demo, keep `--emotion_mapping default` so emotion outputs stay aligned with sale-model feature engineering.

## 5.1 Google Colab training and deployment to this repo

Recommended if your local GPU is limited.

1. In Colab, clone/upload project and run training:
```bash
python backend/scripts/best_train_emotion_model.py \
  --mode train \
  --data_dir /content/data/ravdess \
  --output_dir /content/best_emotion_wav2vec2_v2 \
  --base_model facebook/wav2vec2-base \
  --emotion_mapping default \
  --split_strategy actor_holdout \
  --train_actor_max 18 \
  --epochs 30 \
  --batch_size 8 \
  --learning_rate 2e-5
```

2. Download the folder `/content/best_emotion_wav2vec2_v2/best_checkpoint`.

3. Place it on your demo machine at:
`D:\RealtimeCallSemanticAnal\backend\models\best_emotion_wav2vec2_v2\best_checkpoint`

4. Set backend env for explicit model selection (`backend/.env.local`):
```env
EMOTION_MODEL_PATH=backend/models/best_emotion_wav2vec2_v2/best_checkpoint
```

5. Run model validation before demo:
```powershell
cd D:\RealtimeCallSemanticAnal
python backend\scripts\validate_trained_models.py
python backend\scripts\validate_production_readiness.py
```

## 6. Expected output artifacts

For default run at `backend/models/best_emotion_wav2vec2_v2/`:
- `best_checkpoint/` (Hugging Face checkpoint folder)
  - `config.json`
  - `model.safetensors` or `pytorch_model.bin`
  - feature extractor files
- `data_validation_report.json`
- `best_emotion_training_history.json`
- `best_emotion_training_summary.json`
- `best_emotion_run_config.json`
- `best_emotion_eval_metrics.json` (when evaluate mode is run)
- `best_emotion_classification_report.json`
- `best_emotion_confusion_matrix.csv`
- `best_emotion_confusion_matrix.png`

## 7. Model validation commands

Run a clean evaluation using the saved best checkpoint:
```powershell
cd D:\RealtimeCallSemanticAnal
python backend\scripts\best_train_emotion_model.py `
  --mode evaluate `
  --data_dir data\raw\ravdess `
  --output_dir backend\models\best_emotion_wav2vec2_v2 `
  --checkpoint_dir backend\models\best_emotion_wav2vec2_v2\best_checkpoint `
  --emotion_mapping default `
  --split_strategy actor_holdout `
  --train_actor_max 18
```

Quick load-check of checkpoint:
```powershell
python -c "from transformers import AutoModelForAudioClassification; m=AutoModelForAudioClassification.from_pretrained('backend/models/best_emotion_wav2vec2_v2/best_checkpoint'); print('num_labels=',m.config.num_labels)"
```

Note:
- `backend/scripts/validate_trained_models.py` validates the legacy `emotion_model.pth` flow.
- This new best-model pipeline stores Hugging Face checkpoint artifacts instead.
- Validation scripts accept `EMOTION_MODEL_PATH`; set it to `backend/models/best_emotion_wav2vec2_v2/best_checkpoint` for this repo state.

## 8. Troubleshooting

1. `ModuleNotFoundError: transformers` (or librosa/soundfile)
- Re-run dependency install from `backend`:
```powershell
cd D:\RealtimeCallSemanticAnal\backend
python -m pip install -e .
```

2. `No .wav files found`
- Fix `--data_dir`
- Confirm extraction path and file count with `Get-ChildItem ... -Filter *.wav`

3. CUDA out-of-memory
- Reduce `--batch_size`
- Use `--max_seconds 4.0`
- Use base model instead of large model

4. Validation accuracy unexpectedly very high with weak generalization
- Ensure `--split_strategy actor_holdout` is used
- Avoid comparing actor-holdout scores to random split papers directly

5. Training is too slow on CPU
- Use GPU if available
- Reduce epochs for smoke testing (`--epochs 5`)
- Keep `--num_workers 0` on Windows if worker issues occur

6. Checkpoint folder missing after run
- Confirm training did not stop before first best epoch
- Check console for data corruption or OOM exceptions

## 9. Acceptance checklist

- [ ] `data_validation_report.json` exists and `corrupted_files_count == 0`
- [ ] `best_checkpoint/` exists with model/config files
- [ ] `best_emotion_training_history.json` exists
- [ ] `best_emotion_training_summary.json` exists
- [ ] Evaluate mode runs successfully and writes `best_emotion_eval_metrics.json`
- [ ] Confusion matrix artifacts (`.csv` and `.png`) are generated
- [ ] Actor-holdout macro-F1 is stable across reruns (target: at least 0.75 for baseline settings)

## 10. Sources used for model decision

- RAVDESS dataset (official Zenodo): https://zenodo.org/records/1188976
- Pepino et al., Interspeech 2021 (wav2vec2 embeddings for SER): https://www.isca-archive.org/interspeech_2021/pepino21_interspeech.html
- 2025 RAVDESS comparative paper (reports wav2vec2-large and other models, random 80/20 setup): https://arxiv.org/html/2411.08759v1
- Xception-based high-accuracy RAVDESS study (also random 80/20 split protocol): https://www.mdpi.com/2079-9292/12/22/4581
- SERAB benchmark discussion on robust speaker-independent evaluation concerns: https://arxiv.org/abs/2110.03414
- Hugging Face Transformers audio classification docs: https://huggingface.co/docs/transformers/tasks/audio_classification

