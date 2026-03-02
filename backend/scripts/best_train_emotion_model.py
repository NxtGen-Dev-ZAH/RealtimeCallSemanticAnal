"""
Best-practice RAVDESS emotion model training script (Wav2Vec2-based).

Why this script exists:
- `train_emotion_model.py` trains a CNN+LSTM on hand-crafted features.
- Recent SER literature shows strong gains from self-supervised speech backbones.
- This script fine-tunes a Wav2Vec2 audio-classification head on raw waveform audio.

Outputs:
- Hugging Face checkpoint folder with best model weights
- Structured validation/training/evaluation JSON artifacts
- Confusion matrix image and CSV for quick inspection
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_cosine_schedule_with_warmup,
)


logger = logging.getLogger("best_train_emotion_model")


@dataclass
class RavdessSample:
    path: Path
    label_id: int
    label_name: str
    actor_id: int
    emotion_code: int
    vocal_channel: int


def setup_logging() -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_emotion_mapping(mapping_type: str = "default") -> Dict[int, Tuple[int, str]]:
    """
    Map RAVDESS 8 emotions -> target classes.

    RAVDESS emotion codes:
    01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
    """
    if mapping_type == "default":
        # 5 classes, compatible with existing backend labels.
        return {
            1: (0, "neutral"),
            2: (0, "neutral"),
            3: (1, "happiness"),
            4: (3, "sadness"),
            5: (2, "anger"),
            6: (4, "frustration"),
            7: (4, "frustration"),
            8: (1, "happiness"),
        }
    if mapping_type == "strict":
        # Keep all 8 classes.
        return {
            1: (0, "neutral"),
            2: (1, "calm"),
            3: (2, "happiness"),
            4: (3, "sadness"),
            5: (4, "anger"),
            6: (5, "fear"),
            7: (6, "disgust"),
            8: (7, "surprise"),
        }
    if mapping_type == "expanded":
        # 5 classes with sadness separated from frustration.
        return {
            1: (0, "neutral"),
            2: (0, "neutral"),
            3: (1, "happiness"),
            4: (2, "sadness"),
            5: (3, "anger"),
            6: (4, "frustration"),
            7: (4, "frustration"),
            8: (1, "happiness"),
        }
    raise ValueError(f"Unsupported mapping type: {mapping_type}")


def build_label_maps(mapping: Dict[int, Tuple[int, str]]) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label: Dict[int, str] = {}
    for _, (label_id, label_name) in mapping.items():
        id2label[label_id] = label_name
    id2label = dict(sorted(id2label.items(), key=lambda x: x[0]))
    label2id = {name: idx for idx, name in id2label.items()}
    return id2label, label2id


def parse_ravdess_filename(filename: str) -> Dict[str, int]:
    """
    Parse RAVDESS 7-part file name.
    Example: 03-01-05-01-02-02-12.wav
    """
    name = filename.replace(".wav", "").replace(".WAV", "")
    parts = name.split("-")
    if len(parts) != 7:
        raise ValueError(f"Filename does not have 7 fields: {filename}")

    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Filename has non-numeric fields: {filename}") from exc

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor_id = values
    return {
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion_code,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor_id": actor_id,
    }


def discover_ravdess_samples(
    data_dir: Path,
    mapping_type: str,
    speech_only: bool,
) -> Tuple[List[RavdessSample], Dict[str, Any]]:
    mapping = get_emotion_mapping(mapping_type)
    wav_files = sorted(data_dir.rglob("*.wav"))
    if not wav_files:
        raise ValueError(f"No .wav files found in {data_dir}")

    invalid_filenames: List[str] = []
    skipped_non_speech = 0
    skipped_unknown_emotion = 0
    samples: List[RavdessSample] = []

    for wav_path in wav_files:
        try:
            info = parse_ravdess_filename(wav_path.name)
        except ValueError:
            invalid_filenames.append(str(wav_path))
            continue

        if speech_only and info["vocal_channel"] != 1:
            skipped_non_speech += 1
            continue

        emotion_code = info["emotion_code"]
        if emotion_code not in mapping:
            skipped_unknown_emotion += 1
            continue

        label_id, label_name = mapping[emotion_code]
        samples.append(
            RavdessSample(
                path=wav_path,
                label_id=label_id,
                label_name=label_name,
                actor_id=info["actor_id"],
                emotion_code=emotion_code,
                vocal_channel=info["vocal_channel"],
            )
        )

    if not samples:
        raise ValueError(
            "No valid RAVDESS samples discovered after filename parsing/filtering."
        )

    validation = {
        "data_dir": str(data_dir),
        "total_wav_found": len(wav_files),
        "valid_samples": len(samples),
        "invalid_filename_count": len(invalid_filenames),
        "invalid_filename_examples": invalid_filenames[:10],
        "skipped_non_speech_count": skipped_non_speech,
        "skipped_unknown_emotion_count": skipped_unknown_emotion,
        "mapping_type": mapping_type,
        "speech_only": speech_only,
    }
    return samples, validation


def validate_audio_integrity(
    samples: List[RavdessSample],
    max_validation_files: int,
) -> Dict[str, Any]:
    """
    Validate audio readability and collect signal statistics.
    max_validation_files=0 means validate all samples.
    """
    to_validate = samples if max_validation_files <= 0 else samples[:max_validation_files]

    corrupted: List[str] = []
    sample_rates: List[int] = []
    durations: List[float] = []
    channels: List[int] = []

    for sample in to_validate:
        try:
            info = sf.info(str(sample.path))
            sample_rates.append(int(info.samplerate))
            durations.append(float(info.duration))
            channels.append(int(info.channels))
            if info.frames <= 0:
                corrupted.append(str(sample.path))
        except Exception:
            corrupted.append(str(sample.path))

    actor_ids = sorted({s.actor_id for s in samples})
    label_counts: Dict[str, int] = {}
    for s in samples:
        label_counts[s.label_name] = label_counts.get(s.label_name, 0) + 1

    expected_speech_count = 1440
    speech_count_warning = None
    if len(samples) != expected_speech_count:
        speech_count_warning = (
            f"Expected {expected_speech_count} speech files for Audio_Speech_Actors_01-24.zip, "
            f"found {len(samples)}. This can be valid if you filtered, merged, or use a custom subset."
        )

    report = {
        "validated_files": len(to_validate),
        "corrupted_files_count": len(corrupted),
        "corrupted_file_examples": corrupted[:10],
        "sample_rate_unique": sorted(set(sample_rates)),
        "channels_unique": sorted(set(channels)),
        "duration_seconds": {
            "min": float(np.min(durations)) if durations else None,
            "max": float(np.max(durations)) if durations else None,
            "mean": float(np.mean(durations)) if durations else None,
            "median": float(np.median(durations)) if durations else None,
        },
        "actor_ids_present": actor_ids,
        "actor_count": len(actor_ids),
        "label_counts": label_counts,
        "warnings": [speech_count_warning] if speech_count_warning else [],
        "passes_integrity": len(corrupted) == 0,
    }
    return report


def split_samples(
    samples: List[RavdessSample],
    split_strategy: str,
    train_actor_max: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[RavdessSample], List[RavdessSample], Dict[str, Any]]:
    if split_strategy == "actor_holdout":
        train_samples = [s for s in samples if s.actor_id <= train_actor_max]
        val_samples = [s for s in samples if s.actor_id > train_actor_max]
        split_info = {
            "split_strategy": split_strategy,
            "train_actor_range": f"1-{train_actor_max}",
            "val_actor_range": f"{train_actor_max + 1}-24",
        }
    elif split_strategy == "stratified_random":
        labels = [s.label_id for s in samples]
        train_samples, val_samples = train_test_split(
            samples,
            test_size=val_ratio,
            random_state=seed,
            stratify=labels,
        )
        split_info = {
            "split_strategy": split_strategy,
            "val_ratio": val_ratio,
            "seed": seed,
        }
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    if not train_samples or not val_samples:
        raise ValueError(
            f"Split failed: train={len(train_samples)} val={len(val_samples)}. "
            "Adjust split strategy or data."
        )

    split_info["train_size"] = len(train_samples)
    split_info["val_size"] = len(val_samples)
    return train_samples, val_samples, split_info


def count_labels(samples: List[RavdessSample], id2label: Dict[int, str]) -> Dict[str, int]:
    counts = {label: 0 for label in id2label.values()}
    for sample in samples:
        counts[id2label[sample.label_id]] += 1
    return counts


class RavdessWaveDataset(Dataset):
    def __init__(
        self,
        samples: List[RavdessSample],
        sample_rate: int,
        max_seconds: float,
        augment: bool,
        seed: int,
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, wav: np.ndarray) -> np.ndarray:
        if len(wav) > self.max_samples:
            if self.augment:
                start = int(self.rng.integers(0, len(wav) - self.max_samples + 1))
            else:
                start = (len(wav) - self.max_samples) // 2
            wav = wav[start : start + self.max_samples]
        elif len(wav) < self.max_samples:
            wav = np.pad(wav, (0, self.max_samples - len(wav)))
        return wav.astype(np.float32)

    def _augment_wave(self, wav: np.ndarray) -> np.ndarray:
        # Gain jitter
        gain = float(self.rng.uniform(0.8, 1.2))
        wav = wav * gain

        # Add low-level Gaussian noise with random SNR.
        if float(self.rng.random()) < 0.5:
            noise = self.rng.normal(0.0, 1.0, size=wav.shape).astype(np.float32)
            signal_power = float(np.mean(wav**2) + 1e-8)
            noise_power = float(np.mean(noise**2) + 1e-8)
            snr_db = float(self.rng.uniform(15.0, 30.0))
            noise_scale = math.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10.0))))
            wav = wav + noise_scale * noise

        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        wav, _ = librosa.load(str(sample.path), sr=self.sample_rate, mono=True)
        wav = self._crop_or_pad(wav)
        if self.augment:
            wav = self._augment_wave(wav)
        return {"waveform": wav, "label": sample.label_id}


class AudioCollator:
    def __init__(
        self,
        feature_extractor: AutoFeatureExtractor,
        sample_rate: int,
        max_seconds: float,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_seconds)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        waveforms = [item["waveform"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        features = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features["labels"] = labels
        return features


def build_class_weights(samples: List[RavdessSample], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in samples:
        counts[s.label_id] += 1.0

    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_model(
    model: AutoModelForAudioClassification,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            losses.append(float(loss.item()))

            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_confusion_outputs(
    output_dir: Path,
    y_true: List[int],
    y_pred: List[int],
    id2label: Dict[int, str],
) -> None:
    labels = list(sorted(id2label.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_csv = output_dir / "best_emotion_confusion_matrix.csv"
    np.savetxt(cm_csv, cm, delimiter=",", fmt="%d")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        names = [id2label[i] for i in labels]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
        plt.title("Best Emotion Model - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(output_dir / "best_emotion_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning(f"Could not generate confusion matrix plot: {exc}")


def freeze_feature_encoder(model: AutoModelForAudioClassification) -> None:
    # API naming changed across transformers versions.
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    elif hasattr(model, "freeze_feature_extractor"):
        model.freeze_feature_extractor()


def unfreeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a Wav2Vec2-based SER model on RAVDESS."
    )

    parser.add_argument("--mode", choices=["train", "validate_data", "evaluate"], default="train")
    parser.add_argument("--data_dir", type=str, default="data/raw/ravdess")
    parser.add_argument("--output_dir", type=str, default="backend/models/best_emotion_wav2vec2")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--emotion_mapping", choices=["default", "strict", "expanded"], default="default")
    parser.add_argument(
        "--speech_only",
        dest="speech_only",
        action="store_true",
        default=True,
        help="Use speech-only files (vocal channel=01). Enabled by default.",
    )
    parser.add_argument(
        "--include_song",
        dest="speech_only",
        action="store_false",
        help="Include song files too (disables speech-only filtering).",
    )

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--freeze_feature_encoder_epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--split_strategy", choices=["actor_holdout", "stratified_random"], default="actor_holdout")
    parser.add_argument("--train_actor_max", type=int, default=18)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_validation_files", type=int, default=0)
    parser.add_argument("--disable_augment", action="store_true")
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    speech_only = args.speech_only
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    mapping = get_emotion_mapping(args.emotion_mapping)
    id2label, label2id = build_label_maps(mapping)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Emotion mapping: {args.emotion_mapping}")
    logger.info(f"Speech-only filter: {speech_only}")

    samples, dataset_scan = discover_ravdess_samples(
        data_dir=data_dir,
        mapping_type=args.emotion_mapping,
        speech_only=speech_only,
    )
    integrity_report = validate_audio_integrity(samples, max_validation_files=args.max_validation_files)

    validation_report = {
        "dataset_scan": dataset_scan,
        "integrity_report": integrity_report,
        "label_map": id2label,
    }
    save_json(output_dir / "data_validation_report.json", validation_report)

    logger.info(f"Discovered valid samples: {len(samples)}")
    logger.info(f"Audio integrity corrupted count: {integrity_report['corrupted_files_count']}")
    logger.info(f"Actors present: {integrity_report['actor_count']}")

    if integrity_report["corrupted_files_count"] > 0:
        logger.warning("Corrupted files found. Review data_validation_report.json before training.")

    if args.mode == "validate_data":
        logger.info("Data validation completed. Exiting due to --mode validate_data.")
        return

    train_samples, val_samples, split_info = split_samples(
        samples=samples,
        split_strategy=args.split_strategy,
        train_actor_max=args.train_actor_max,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_counts = count_labels(train_samples, id2label)
    val_counts = count_labels(val_samples, id2label)
    logger.info(f"Train size: {len(train_samples)} | Val size: {len(val_samples)}")
    logger.info(f"Train label distribution: {train_counts}")
    logger.info(f"Val label distribution: {val_counts}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)
    collator = AudioCollator(
        feature_extractor=feature_extractor,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )

    train_dataset = RavdessWaveDataset(
        samples=train_samples,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        augment=not args.disable_augment,
        seed=args.seed,
    )
    val_dataset = RavdessWaveDataset(
        samples=val_samples,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
        augment=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "best_checkpoint"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    if args.mode == "evaluate":
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        model = AutoModelForAudioClassification.from_pretrained(checkpoint_dir).to(device)
        class_weights = build_class_weights(train_samples, num_classes=len(id2label), device=device)
        val_loss, val_metrics, y_true, y_pred = evaluate_model(model, val_loader, device, class_weights)

        eval_report = {
            "mode": "evaluate",
            "checkpoint_dir": str(checkpoint_dir),
            "val_loss": val_loss,
            **val_metrics,
            "split_info": split_info,
            "train_counts": train_counts,
            "val_counts": val_counts,
        }
        save_json(output_dir / "best_emotion_eval_metrics.json", eval_report)
        save_confusion_outputs(output_dir, y_true, y_pred, id2label)
        report = classification_report(
            y_true,
            y_pred,
            labels=list(sorted(id2label.keys())),
            target_names=[id2label[i] for i in sorted(id2label.keys())],
            output_dict=True,
            zero_division=0,
        )
        save_json(output_dir / "best_emotion_classification_report.json", report)
        logger.info(f"Evaluation complete. Metrics: {eval_report}")
        return

    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        finetuning_task="audio-classification",
    )
    model = AutoModelForAudioClassification.from_pretrained(
        args.base_model,
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)

    if args.freeze_feature_encoder_epochs > 0:
        freeze_feature_encoder(model)
        logger.info(f"Feature encoder frozen for first {args.freeze_feature_encoder_epochs} epoch(s).")

    class_weights = build_class_weights(train_samples, num_classes=len(id2label), device=device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1_macro": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
    }

    best_metric = -1.0
    best_epoch = -1
    patience_counter = 0

    logger.info("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        y_true_train: List[int] = []
        y_pred_train: List[int] = []
        optimizer.zero_grad(set_to_none=True)

        if epoch == args.freeze_feature_encoder_epochs + 1 and args.freeze_feature_encoder_epochs > 0:
            unfreeze_all_parameters(model)
            logger.info("Feature encoder unfrozen.")

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                logits = outputs.logits
                raw_loss = F.cross_entropy(logits, labels, weight=class_weights)
                loss = raw_loss / max(args.gradient_accumulation_steps, 1)

            scaler.scale(loss).backward()

            do_step = (
                batch_idx % max(args.gradient_accumulation_steps, 1) == 0
                or batch_idx == len(train_loader)
            )
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            losses.append(float(raw_loss.item()))
            preds = torch.argmax(logits.detach(), dim=-1)
            y_true_train.extend(labels.detach().cpu().tolist())
            y_pred_train.extend(preds.cpu().tolist())

        train_metrics = compute_metrics(y_true_train, y_pred_train)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss, val_metrics, y_true_val, y_pred_val = evaluate_model(model, val_loader, device, class_weights)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_f1_weighted"].append(val_metrics["f1_weighted"])

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        current_metric = val_metrics["f1_macro"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            feature_extractor.save_pretrained(checkpoint_dir)
            save_confusion_outputs(output_dir, y_true_val, y_pred_val, id2label)
            report = classification_report(
                y_true_val,
                y_pred_val,
                labels=list(sorted(id2label.keys())),
                target_names=[id2label[i] for i in sorted(id2label.keys())],
                output_dict=True,
                zero_division=0,
            )
            save_json(output_dir / "best_emotion_classification_report.json", report)
            logger.info(f"New best checkpoint at epoch {epoch} (val_f1_macro={best_metric:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    history_file = output_dir / "best_emotion_training_history.json"
    save_json(history_file, history)

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_metric,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "emotion_mapping": args.emotion_mapping,
        "split_info": split_info,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "device_used": str(device),
        "amp_enabled": use_amp,
    }
    save_json(output_dir / "best_emotion_training_summary.json", summary)
    save_json(
        output_dir / "best_emotion_run_config.json",
        {
            "args": vars(args),
            "id2label": id2label,
            "label2id": label2id,
            "dataset_scan": dataset_scan,
            "integrity_report": integrity_report,
        },
    )

    logger.info("Training complete.")
    logger.info(f"Best checkpoint: {checkpoint_dir}")
    logger.info(f"Best val_f1_macro: {best_metric:.4f} at epoch {best_epoch}")
    logger.info(f"History saved: {history_file}")


if __name__ == "__main__":
    main()
