"""
This script generates realistic call-level data aligned with the project's sale prediction pipeline. It outputs:
1) A full analytics dataset (metadata + features + labels)
2) Optional strict training dataset (only required training columns)
3) Validation report + dataset statistics + plots
Compatible with:
- backend/scripts/train_sale_predictor.py
- Required feature schema in the SalePredictor training pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_sale_training_dataset")


TRAINING_FEATURES: List[str] = [
    "sentiment_mean",
    "sentiment_variance",
    "emotion_neutral",
    "emotion_happiness",
    "emotion_anger",
    "emotion_sadness",
    "emotion_frustration",
    "silence_ratio",
    "interruption_frequency",
    "talk_listen_ratio",
    "turn_taking_frequency",
]

OPTIONAL_FEATURES: List[str] = [
    "filler_word_frequency",
]

LABEL_COLUMN = "sale_outcome"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass
class GeneratorConfig:
    n_samples: int = 10000
    sale_ratio: float = 0.30
    random_seed: int = 42
    feature_noise: float = 0.06
    label_noise: float = 0.75
    contradiction_rate: float = 0.08
    start_date: str = "2025-01-01"
    days: int = 365
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    include_metadata: bool = True


class ProductionSyntheticSalesGenerator:
    """
    Generates high-quality synthetic dataset with:
    - realistic latent factors
    - correlated conversational features
    - valid emotion probabilities
    - calibrated sale probability and target conversion ratio
    - contradictory edge cases for realism
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        self.customer_segments = np.array(
            ["SMB", "MidMarket", "Enterprise", "Consumer"], dtype=object
        )
        self.customer_segment_probs = np.array([0.38, 0.28, 0.17, 0.17], dtype=float)

        self.channels = np.array(["inbound", "outbound"], dtype=object)
        self.channel_probs = np.array([0.62, 0.38], dtype=float)

        self.regions = np.array(
            ["north_america", "europe", "middle_east", "apac", "latam"], dtype=object
        )
        self.region_probs = np.array([0.36, 0.22, 0.12, 0.22, 0.08], dtype=float)

        self.product_lines = np.array(["basic", "pro", "enterprise"], dtype=object)
        self.product_line_probs = np.array([0.45, 0.40, 0.15], dtype=float)

        self.agent_tiers = np.array(["junior", "mid", "senior"], dtype=object)
        self.agent_tier_probs = np.array([0.30, 0.50, 0.20], dtype=float)

        self.segment_effect = {
            "SMB": 0.08,
            "MidMarket": 0.14,
            "Enterprise": 0.20,
            "Consumer": -0.06,
        }
        self.channel_effect = {"inbound": 0.08, "outbound": -0.03}
        self.region_effect = {
            "north_america": 0.05,
            "europe": 0.03,
            "middle_east": 0.00,
            "apac": 0.02,
            "latam": -0.04,
        }

    def _timestamps(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic call timestamps and temporal effect."""
        start = datetime.fromisoformat(self.config.start_date)
        day_offsets = self.rng.integers(0, self.config.days, size=n)
        hour_choices = np.arange(8, 21)
        hour_probs = np.array(
            [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.11, 0.10, 0.10, 0.09, 0.07, 0.05, 0.03],
            dtype=float,
        )
        hour_probs = hour_probs / hour_probs.sum()
        hours = self.rng.choice(hour_choices, size=n, p=hour_probs)
        minutes = self.rng.integers(0, 60, size=n)

        timestamps = np.array(
            [
                start + timedelta(days=int(d), hours=int(h), minutes=int(m))
                for d, h, m in zip(day_offsets, hours, minutes)
            ],
            dtype=object,
        )

        day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps], dtype=float)
        month = np.array([ts.month for ts in timestamps], dtype=int)

        seasonal = 0.16 * np.sin(2.0 * math.pi * day_of_year / 365.25) + 0.07 * np.cos(
            2.0 * math.pi * day_of_year / 365.25
        )
        quarter_boost = np.where(np.isin(month, [11, 12]), 0.12, 0.0) + np.where(
            np.isin(month, [1, 2]), -0.05, 0.0
        )
        temporal_effect = seasonal + quarter_boost

        return timestamps, temporal_effect

    def _sample_categorical(self, n: int) -> Dict[str, np.ndarray]:
        return {
            "customer_segment": self.rng.choice(
                self.customer_segments, size=n, p=self.customer_segment_probs
            ),
            "channel": self.rng.choice(self.channels, size=n, p=self.channel_probs),
            "region": self.rng.choice(self.regions, size=n, p=self.region_probs),
            "product_line": self.rng.choice(
                self.product_lines, size=n, p=self.product_line_probs
            ),
            "agent_tier": self.rng.choice(
                self.agent_tiers, size=n, p=self.agent_tier_probs
            ),
        }

    def _latent_factors(
        self,
        n: int,
        categories: Dict[str, np.ndarray],
        temporal_effect: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Sample project-specific latent factors that drive feature realism."""
        base_affinity = self.rng.normal(0.0, 1.0, n)
        price_sensitivity = clip01(self.rng.beta(2.6, 2.1, size=n))
        urgency = clip01(
            self.rng.beta(2.2, 2.7, size=n)
            + 0.12 * (categories["channel"] == "inbound").astype(float)
        )

        tier_skill_base = np.select(
            [
                categories["agent_tier"] == "junior",
                categories["agent_tier"] == "mid",
                categories["agent_tier"] == "senior",
            ],
            [0.40, 0.57, 0.75],
            default=0.55,
        )
        agent_skill = clip01(tier_skill_base + self.rng.normal(0.0, 0.08, n))

        product_line_fit_bonus = np.select(
            [
                categories["product_line"] == "basic",
                categories["product_line"] == "pro",
                categories["product_line"] == "enterprise",
            ],
            [0.03, 0.08, 0.12],
            default=0.05,
        )

        product_fit = clip01(
            sigmoid(
                0.95 * base_affinity
                - 0.85 * price_sensitivity
                + 0.40 * temporal_effect
                + product_line_fit_bonus
                + self.rng.normal(0.0, 0.60, n)
            )
        )

        trust = clip01(
            sigmoid(
                0.95 * agent_skill
                + 0.65 * base_affinity
                - 0.55 * price_sensitivity
                + self.rng.normal(0.0, 0.55, n)
            )
        )

        objection_intensity = clip01(
            sigmoid(
                0.95 * price_sensitivity
                + 0.85 * (1.0 - product_fit)
                + 0.35 * (categories["channel"] == "outbound").astype(float)
                + self.rng.normal(0.0, 0.65, n)
            )
        )

        complexity = clip01(
            sigmoid(
                0.70 * self.rng.normal(0.0, 1.0, n)
                + 0.65 * objection_intensity
                + 0.20 * (categories["product_line"] == "enterprise").astype(float)
            )
        )

        budget_fit = clip01(
            sigmoid(
                0.95 * (1.0 - price_sensitivity)
                + 0.65 * product_fit
                + self.rng.normal(0.0, 0.55, n)
            )
        )

        intent = clip01(
            sigmoid(
                1.00 * base_affinity
                + 0.78 * urgency
                + 0.42 * trust
                - 0.85 * objection_intensity
                + self.rng.normal(0.0, 0.60, n)
            )
        )

        return {
            "base_affinity": base_affinity,
            "price_sensitivity": price_sensitivity,
            "urgency": urgency,
            "agent_skill": agent_skill,
            "product_fit": product_fit,
            "trust": trust,
            "objection_intensity": objection_intensity,
            "complexity": complexity,
            "budget_fit": budget_fit,
            "intent": intent,
        }

    def _conversation_features(
        self,
        n: int,
        categories: Dict[str, np.ndarray],
        latent: Dict[str, np.ndarray],
        temporal_effect: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate conversational dynamics and sentiment features."""
        fn = self.config.feature_noise
        g = self.rng.normal

        sentiment_raw = (
            1.35 * latent["intent"]
            + 0.95 * latent["trust"]
            + 0.74 * latent["product_fit"]
            - 1.12 * latent["objection_intensity"]
            - 0.58 * latent["price_sensitivity"]
            + 0.26 * temporal_effect
            + g(0.0, 0.35 + fn, n)
        )
        sentiment_mean = np.clip(np.tanh(sentiment_raw / 2.0), -1.0, 1.0)

        sentiment_variance = np.clip(
            0.015
            + 0.16 * latent["complexity"]
            + 0.12 * latent["objection_intensity"]
            + 0.08 * (1.0 - latent["agent_skill"])
            + np.abs(g(0.0, 0.045 + fn / 2.0, n)),
            0.0,
            0.55,
        )

        silence_ratio = np.clip(
            0.06
            + 0.34 * latent["objection_intensity"]
            + 0.18 * latent["complexity"]
            - 0.18 * latent["intent"]
            - 0.11 * latent["agent_skill"]
            + np.abs(g(0.0, 0.06 + fn, n)),
            0.0,
            0.95,
        )

        interruption_frequency = np.clip(
            0.02
            + 0.26 * latent["complexity"]
            + 0.27 * latent["objection_intensity"]
            + 0.05 * (categories["channel"] == "outbound").astype(float)
            - 0.08 * latent["agent_skill"]
            + np.abs(g(0.0, 0.055 + fn, n)),
            0.0,
            0.95,
        )

        talk_listen_ratio = np.clip(
            1.00
            + 0.56 * latent["agent_skill"]
            - 0.24 * latent["objection_intensity"]
            + 0.18 * (categories["channel"] == "outbound").astype(float)
            - 0.10 * (categories["customer_segment"] == "Enterprise").astype(float)
            + g(0.0, 0.18 + fn, n),
            0.2,
            3.0,
        )

        turn_taking_frequency = np.clip(
            0.09
            + 0.43 * latent["intent"]
            + 0.22 * latent["agent_skill"]
            + 0.14 * latent["complexity"]
            - 0.16 * latent["objection_intensity"]
            + g(0.0, 0.08 + fn, n),
            0.0,
            1.0,
        )

        filler_word_frequency = np.clip(
            0.45
            + 2.8 * latent["complexity"]
            + 1.7 * latent["objection_intensity"]
            + 1.2 * (1.0 - latent["agent_skill"])
            + 3.0 * interruption_frequency
            + np.abs(g(0.0, 0.45 + fn, n)),
            0.0,
            12.0,
        )

        call_duration_sec = np.clip(
            g(
                loc=330.0 + 440.0 * latent["complexity"] + 190.0 * latent["objection_intensity"],
                scale=90.0 + 60.0 * latent["complexity"],
                size=n,
            ),
            60.0,
            3600.0,
        )

        return {
            "sentiment_mean": sentiment_mean,
            "sentiment_variance": sentiment_variance,
            "silence_ratio": silence_ratio,
            "interruption_frequency": interruption_frequency,
            "talk_listen_ratio": talk_listen_ratio,
            "turn_taking_frequency": turn_taking_frequency,
            "filler_word_frequency": filler_word_frequency,
            "call_duration_sec": call_duration_sec,
        }

    def _emotion_features(
        self,
        n: int,
        latent: Dict[str, np.ndarray],
        conv: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Generate valid emotion probabilities from latent conversational states."""
        noise = self.rng.normal(0.0, 0.35 + self.config.feature_noise, (n, 5))

        neutral = (
            0.55
            - 0.56 * np.abs(conv["sentiment_mean"])
            + 0.22 * (1.0 - latent["urgency"])
            + 0.15 * (1.0 - latent["objection_intensity"])
        )
        happiness = (
            1.62 * conv["sentiment_mean"]
            + 0.88 * latent["trust"]
            + 0.72 * latent["product_fit"]
            - 0.62 * latent["objection_intensity"]
            + 0.18 * latent["urgency"]
        )
        anger = (
            1.15 * latent["objection_intensity"]
            + 0.85 * conv["interruption_frequency"]
            + 0.72 * latent["price_sensitivity"]
            - 0.72 * latent["agent_skill"]
            - 0.78 * conv["sentiment_mean"]
        )
        sadness = (
            0.92 * (1.0 - latent["intent"])
            + 0.62 * (1.0 - latent["product_fit"])
            + 0.35 * latent["complexity"]
            - 0.30 * latent["trust"]
        )
        frustration = (
            1.12 * latent["objection_intensity"]
            + 0.74 * conv["silence_ratio"]
            + 0.72 * conv["interruption_frequency"]
            + 0.42 * latent["complexity"]
            - 0.60 * latent["agent_skill"]
        )

        logits = np.column_stack([neutral, happiness, anger, sadness, frustration]) + noise
        probs = softmax(logits)

        return {
            "emotion_neutral": probs[:, 0],
            "emotion_happiness": probs[:, 1],
            "emotion_anger": probs[:, 2],
            "emotion_sadness": probs[:, 3],
            "emotion_frustration": probs[:, 4],
        }

    @staticmethod
    def _calibrate_intercept(base_score: np.ndarray, target_ratio: float) -> float:
        """Find intercept so mean(sigmoid(base_score + intercept)) ~= target_ratio."""
        lo, hi = -12.0, 12.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            ratio = float(sigmoid(base_score + mid).mean())
            if ratio < target_ratio:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _sale_probability(
        self,
        categories: Dict[str, np.ndarray],
        latent: Dict[str, np.ndarray],
        conv: Dict[str, np.ndarray],
        emotions: Dict[str, np.ndarray],
        temporal_effect: np.ndarray,
    ) -> np.ndarray:
        seg_eff = np.array([self.segment_effect[s] for s in categories["customer_segment"]], dtype=float)
        ch_eff = np.array([self.channel_effect[c] for c in categories["channel"]], dtype=float)
        reg_eff = np.array([self.region_effect[r] for r in categories["region"]], dtype=float)

        base = (
            2.25 * conv["sentiment_mean"]
            - 0.62 * conv["sentiment_variance"]
            + 1.52 * emotions["emotion_happiness"]
            - 1.22 * emotions["emotion_anger"]
            - 1.40 * emotions["emotion_frustration"]
            - 0.74 * emotions["emotion_sadness"]
            + 0.43 * emotions["emotion_neutral"]
            - 1.12 * conv["silence_ratio"]
            - 0.92 * conv["interruption_frequency"]
            + 0.47 * np.log1p(conv["talk_listen_ratio"])
            + 0.86 * conv["turn_taking_frequency"]
            - 0.28 * np.log1p(conv["filler_word_frequency"])
            + 1.04 * latent["intent"]
            + 0.95 * latent["product_fit"]
            + 0.72 * latent["budget_fit"]
            + 0.64 * latent["trust"]
            + 0.48 * latent["agent_skill"]
            + 0.56 * latent["urgency"]
            + 0.52 * conv["sentiment_mean"] * emotions["emotion_happiness"]
            - 0.61 * emotions["emotion_frustration"] * conv["interruption_frequency"]
            + 0.42 * temporal_effect
            + seg_eff
            + ch_eff
            + reg_eff
        )

        noisy_score = base + self.rng.normal(0.0, self.config.label_noise, len(base))
        intercept = self._calibrate_intercept(noisy_score, self.config.sale_ratio)
        p = sigmoid(noisy_score + intercept)
        return np.clip(p, 0.001, 0.999)

    def _inject_contradictions(self, labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Inject realistic contradictory outcomes to avoid overly clean separability."""
        n = len(labels)
        flips = int(self.config.contradiction_rate * n)
        if flips <= 0:
            return labels

        y = labels.copy()
        high_conf_sales = np.where((y == 1) & (probabilities >= 0.80))[0]
        high_conf_no_sales = np.where((y == 0) & (probabilities <= 0.20))[0]

        n_flip_sales = min(len(high_conf_sales), flips // 2)
        n_flip_no_sales = min(len(high_conf_no_sales), flips - n_flip_sales)

        if n_flip_sales > 0:
            idx = self.rng.choice(high_conf_sales, size=n_flip_sales, replace=False)
            y[idx] = 0
        if n_flip_no_sales > 0:
            idx = self.rng.choice(high_conf_no_sales, size=n_flip_no_sales, replace=False)
            y[idx] = 1

        return y

    def _split_column(self, n: int) -> np.ndarray:
        """Create train/val/test split marker column."""
        train_r = self.config.train_ratio
        val_r = self.config.val_ratio
        test_r = 1.0 - train_r - val_r
        if test_r <= 0:
            test_r = 0.15
            val_r = 1.0 - train_r - test_r
        probs = np.array([train_r, val_r, test_r], dtype=float)
        probs = probs / probs.sum()
        return self.rng.choice(np.array(["train", "val", "test"], dtype=object), size=n, p=probs)

    def generate(self) -> pd.DataFrame:
        """Generate full production-style synthetic dataset."""
        n = self.config.n_samples
        categories = self._sample_categorical(n)
        timestamps, temporal_effect = self._timestamps(n)
        latent = self._latent_factors(n, categories, temporal_effect)
        conv = self._conversation_features(n, categories, latent, temporal_effect)
        emotions = self._emotion_features(n, latent, conv)
        probabilities = self._sale_probability(categories, latent, conv, emotions, temporal_effect)
        labels = self.rng.binomial(1, probabilities, size=n).astype(int)
        labels = self._inject_contradictions(labels, probabilities)

        call_ids = np.array([f"CALL_{i:07d}" for i in range(1, n + 1)], dtype=object)
        split = self._split_column(n)

        df = pd.DataFrame(
            {
                "call_id": call_ids,
                "call_timestamp": pd.to_datetime(timestamps),
                "customer_segment": categories["customer_segment"],
                "channel": categories["channel"],
                "region": categories["region"],
                "product_line": categories["product_line"],
                "agent_tier": categories["agent_tier"],
                "data_split": split,
                "true_sale_probability": probabilities,
                "call_duration_sec": conv["call_duration_sec"],
                "sentiment_mean": conv["sentiment_mean"],
                "sentiment_variance": conv["sentiment_variance"],
                "emotion_neutral": emotions["emotion_neutral"],
                "emotion_happiness": emotions["emotion_happiness"],
                "emotion_anger": emotions["emotion_anger"],
                "emotion_sadness": emotions["emotion_sadness"],
                "emotion_frustration": emotions["emotion_frustration"],
                "silence_ratio": conv["silence_ratio"],
                "interruption_frequency": conv["interruption_frequency"],
                "talk_listen_ratio": conv["talk_listen_ratio"],
                "turn_taking_frequency": conv["turn_taking_frequency"],
                "filler_word_frequency": conv["filler_word_frequency"],
                LABEL_COLUMN: labels,
            }
        )

        if not self.config.include_metadata:
            keep = TRAINING_FEATURES + OPTIONAL_FEATURES + [LABEL_COLUMN]
            df = df[keep].copy()

        df = df.sample(frac=1.0, random_state=self.config.random_seed).reset_index(drop=True)
        return df


def validate_dataset(df: pd.DataFrame, target_ratio: float) -> Dict[str, object]:
    """Validate data integrity and modeling usefulness."""
    results: Dict[str, object] = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "statistics": {},
    }

    for col in TRAINING_FEATURES + [LABEL_COLUMN]:
        if col not in df.columns:
            results["errors"].append(f"Missing required column: {col}")
    if results["errors"]:
        results["valid"] = False
        return results

    if df.isnull().any().any():
        nulls = df.isnull().sum()
        bad = {k: int(v) for k, v in nulls[nulls > 0].to_dict().items()}
        results["errors"].append(f"Null values detected: {bad}")

    label_values = set(df[LABEL_COLUMN].unique().tolist())
    if not label_values.issubset({0, 1}):
        results["errors"].append(f"Label values must be binary 0/1, found: {sorted(label_values)}")

    if {"emotion_neutral", "emotion_happiness", "emotion_anger", "emotion_sadness", "emotion_frustration"}.issubset(
        set(df.columns)
    ):
        emotion_sum = (
            df["emotion_neutral"]
            + df["emotion_happiness"]
            + df["emotion_anger"]
            + df["emotion_sadness"]
            + df["emotion_frustration"]
        )
        max_dev = float(np.abs(emotion_sum - 1.0).max())
        if max_dev > 1e-3:
            results["warnings"].append(
                f"Emotion probability sums deviate from 1.0 (max deviation={max_dev:.6f})"
            )

    bounded_01 = [
        "sentiment_variance",
        "emotion_neutral",
        "emotion_happiness",
        "emotion_anger",
        "emotion_sadness",
        "emotion_frustration",
        "silence_ratio",
        "interruption_frequency",
        "turn_taking_frequency",
    ]
    for col in bounded_01:
        if col in df.columns:
            low = float(df[col].min())
            high = float(df[col].max())
            if low < -1e-6 or high > 1.0 + 1e-6:
                results["errors"].append(f"{col} out of expected [0,1] range: min={low:.4f}, max={high:.4f}")

    if "sentiment_mean" in df.columns:
        low = float(df["sentiment_mean"].min())
        high = float(df["sentiment_mean"].max())
        if low < -1.0 - 1e-6 or high > 1.0 + 1e-6:
            results["errors"].append(f"sentiment_mean out of expected [-1,1] range: min={low:.4f}, max={high:.4f}")

    sale_ratio = float(df[LABEL_COLUMN].mean())
    results["statistics"]["sale_ratio"] = sale_ratio
    results["statistics"]["target_sale_ratio"] = target_ratio
    results["statistics"]["label_distribution"] = {
        str(int(k)): int(v) for k, v in df[LABEL_COLUMN].value_counts().to_dict().items()
    }
    if abs(sale_ratio - target_ratio) > 0.06:
        results["warnings"].append(
            f"Sale ratio differs from target by > 0.06 (target={target_ratio:.3f}, actual={sale_ratio:.3f})"
        )

    correlations = {
        feature: float(df[feature].corr(df[LABEL_COLUMN])) for feature in TRAINING_FEATURES
    }
    results["statistics"]["feature_correlations"] = correlations

    # Quick sanity: dataset should be learnable but not perfectly separable
    try:
        x = df[TRAINING_FEATURES].values
        y = df[LABEL_COLUMN].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=y
        )
        model = LogisticRegression(max_iter=1200)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
        results["statistics"]["logistic_auc"] = auc
        if auc < 0.62:
            results["warnings"].append(
                f"Low predictive signal detected (logistic AUC={auc:.3f}). Increase sample size or lower noise."
            )
        if auc > 0.96:
            results["warnings"].append(
                f"Dataset may be too easy / overly separable (logistic AUC={auc:.3f}). Increase contradiction_rate."
            )
    except Exception as exc:
        results["warnings"].append(f"Could not compute logistic AUC sanity check: {exc}")

    if results["errors"]:
        results["valid"] = False

    return results


def build_statistics(df: pd.DataFrame, config: GeneratorConfig, validation: Dict[str, object]) -> Dict[str, object]:
    """Build JSON-serializable statistics payload."""
    stats: Dict[str, object] = {
        "generator_config": asdict(config),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "label_distribution": validation.get("statistics", {}).get("label_distribution", {}),
        "sale_ratio": validation.get("statistics", {}).get("sale_ratio", 0.0),
        "target_sale_ratio": validation.get("statistics", {}).get("target_sale_ratio", 0.0),
        "logistic_auc": validation.get("statistics", {}).get("logistic_auc", None),
        "feature_summary": {},
        "feature_correlations": validation.get("statistics", {}).get("feature_correlations", {}),
        "warnings": validation.get("warnings", []),
    }

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        s = df[col]
        stats["feature_summary"][col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.quantile(0.50)),
            "q75": float(s.quantile(0.75)),
        }

    if "customer_segment" in df.columns:
        by_segment = (
            df.groupby("customer_segment")[LABEL_COLUMN]
            .agg(["count", "mean"])
            .rename(columns={"mean": "conversion_rate"})
            .reset_index()
        )
        stats["conversion_by_segment"] = by_segment.to_dict(orient="records")

    if "channel" in df.columns:
        by_channel = (
            df.groupby("channel")[LABEL_COLUMN]
            .agg(["count", "mean"])
            .rename(columns={"mean": "conversion_rate"})
            .reset_index()
        )
        stats["conversion_by_channel"] = by_channel.to_dict(orient="records")

    if "call_timestamp" in df.columns:
        ts = pd.to_datetime(df["call_timestamp"])
        monthly = (
            df.assign(month=ts.dt.to_period("M").astype(str))
            .groupby("month")[LABEL_COLUMN]
            .mean()
            .reset_index(name="conversion_rate")
        )
        stats["monthly_conversion"] = monthly.to_dict(orient="records")

    return stats


def save_plots(df: pd.DataFrame, stats_dir: Path) -> None:
    """Generate analytics plots for reporting and QA."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries unavailable; skipping plot generation.")
        return

    stats_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    dist_cols = TRAINING_FEATURES + OPTIONAL_FEATURES
    n_cols = 3
    n_rows = int(math.ceil(len(dist_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.8 * n_rows))
    axes = np.array(axes).reshape(-1)
    for idx, feature in enumerate(dist_cols):
        ax = axes[idx]
        if feature not in df.columns:
            ax.axis("off")
            continue
        df[df[LABEL_COLUMN] == 1][feature].hist(bins=40, alpha=0.6, label="sale", ax=ax)
        df[df[LABEL_COLUMN] == 0][feature].hist(bins=40, alpha=0.6, label="no_sale", ax=ax)
        ax.set_title(feature)
        ax.legend()
    for idx in range(len(dist_cols), len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    fig.savefig(stats_dir / "feature_distributions_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    corr_cols = [c for c in TRAINING_FEATURES + OPTIONAL_FEATURES + [LABEL_COLUMN] if c in df.columns]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, linewidths=0.5, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(stats_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    if "call_timestamp" in df.columns:
        month_df = (
            df.assign(month=pd.to_datetime(df["call_timestamp"]).dt.to_period("M").astype(str))
            .groupby("month")[LABEL_COLUMN]
            .mean()
            .reset_index(name="conversion_rate")
        )
        plt.figure(figsize=(12, 5))
        plt.plot(month_df["month"], month_df["conversion_rate"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Conversion rate")
        plt.title("Monthly Conversion Trend")
        plt.tight_layout()
        plt.savefig(stats_dir / "monthly_conversion_trend.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "customer_segment" in df.columns:
        seg = (
            df.groupby("customer_segment")[LABEL_COLUMN]
            .mean()
            .sort_values(ascending=False)
            .reset_index(name="conversion_rate")
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=seg, x="customer_segment", y="conversion_rate")
        plt.title("Conversion Rate by Customer Segment")
        plt.tight_layout()
        plt.savefig(stats_dir / "conversion_by_segment.png", dpi=150, bbox_inches="tight")
        plt.close()


def save_outputs(
    df: pd.DataFrame,
    output_path: Path,
    training_output_path: Path | None,
    stats_dir: Path,
    config: GeneratorConfig,
    validation: Dict[str, object],
) -> None:
    """Persist generated datasets and reports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info("Saved full dataset to %s (%d rows, %d columns)", output_path, len(df), df.shape[1])

    train_df = df[TRAINING_FEATURES + [LABEL_COLUMN]].copy()
    if training_output_path is not None:
        training_output_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(training_output_path, index=False)
        logger.info(
            "Saved strict training dataset to %s (%d rows, %d columns)",
            training_output_path,
            len(train_df),
            train_df.shape[1],
        )

    stats = build_statistics(df, config, validation)
    with open(stats_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(stats_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    if "customer_segment" in df.columns:
        by_segment = (
            df.groupby("customer_segment")[LABEL_COLUMN]
            .agg(["count", "mean"])
            .rename(columns={"mean": "conversion_rate"})
            .reset_index()
        )
        by_segment.to_csv(stats_dir / "conversion_by_segment.csv", index=False)

    if "channel" in df.columns:
        by_channel = (
            df.groupby("channel")[LABEL_COLUMN]
            .agg(["count", "mean"])
            .rename(columns={"mean": "conversion_rate"})
            .reset_index()
        )
        by_channel.to_csv(stats_dir / "conversion_by_channel.csv", index=False)

    if "call_timestamp" in df.columns:
        by_month = (
            df.assign(month=pd.to_datetime(df["call_timestamp"]).dt.to_period("M").astype(str))
            .groupby("month")[LABEL_COLUMN]
            .agg(["count", "mean"])
            .rename(columns={"mean": "conversion_rate"})
            .reset_index()
        )
        by_month.to_csv(stats_dir / "conversion_by_month.csv", index=False)

    save_plots(df, stats_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate production-grade synthetic sales dataset for analytics and model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full dataset + stats (default)
  python backend/scripts/generate_sale_training_dataset.py \\
    --n_samples 20000 \\
    --sale_ratio 0.30 \\
    --output data/sale_training_data.csv \\
    --stats_dir data/dataset_stats \\
    --overwrite

  # Also export strict training schema for train_sale_predictor.py
  python backend/scripts/generate_sale_training_dataset.py \\
    --n_samples 12000 \\
    --sale_ratio 0.28 \\
    --output data/synthetic_sales_full.csv \\
    --training_output data/sale_training_data.csv \\
    --overwrite
""",
    )

    parser.add_argument("--n_samples", type=int, default=10000, help="Number of rows to generate.")
    parser.add_argument(
        "--sale_ratio",
        type=float,
        default=0.30,
        help="Target conversion rate for sale_outcome=1 (0-1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sale_training_data.csv",
        help="Output path for full synthetic dataset CSV.",
    )
    parser.add_argument(
        "--training_output",
        type=str,
        default=None,
        help="Optional path for strict training dataset (required features + label only).",
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        default=None,
        help="Directory to save statistics JSON, validation report, and plots.",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--feature_noise", type=float, default=0.06, help="Feature noise strength (higher = noisier features)."
    )
    parser.add_argument(
        "--label_noise",
        type=float,
        default=0.75,
        help="Noise in sale logit generation (higher = harder classification).",
    )
    parser.add_argument(
        "--contradiction_rate",
        type=float,
        default=0.08,
        help="Fraction of high-confidence outcomes to flip for realism.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2025-01-01",
        help="Start date for synthetic call timeline, format YYYY-MM-DD.",
    )
    parser.add_argument("--days", type=int, default=365, help="Timeline span in days.")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio for data_split marker.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio for data_split marker.")
    parser.add_argument(
        "--without_metadata",
        action="store_true",
        help="Output only model-centric features (drops metadata columns).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_samples < 300:
        raise ValueError("n_samples must be >= 300 for meaningful training and analytics.")
    if not (0.0 < args.sale_ratio < 1.0):
        raise ValueError("sale_ratio must be strictly between 0 and 1.")
    if not (0.0 <= args.feature_noise <= 1.0):
        raise ValueError("feature_noise must be in [0, 1].")
    if not (0.0 <= args.label_noise <= 2.0):
        raise ValueError("label_noise must be in [0, 2].")
    if not (0.0 <= args.contradiction_rate <= 0.30):
        raise ValueError("contradiction_rate must be in [0, 0.30].")
    if args.days < 30:
        raise ValueError("days must be >= 30.")
    if not (0.4 <= args.train_ratio <= 0.9):
        raise ValueError("train_ratio should be in [0.4, 0.9].")
    if not (0.05 <= args.val_ratio <= 0.4):
        raise ValueError("val_ratio should be in [0.05, 0.4].")
    if args.train_ratio + args.val_ratio >= 0.98:
        raise ValueError("train_ratio + val_ratio must leave room for test split.")
    datetime.fromisoformat(args.start_date)


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except Exception as exc:
        logger.error("Invalid arguments: %s", exc)
        sys.exit(1)

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        logger.error("Output file already exists: %s. Use --overwrite to replace it.", output_path)
        sys.exit(1)

    training_output_path = Path(args.training_output) if args.training_output else None
    if training_output_path and training_output_path.exists() and not args.overwrite:
        logger.error(
            "Training output file already exists: %s. Use --overwrite to replace it.",
            training_output_path,
        )
        sys.exit(1)

    stats_dir = Path(args.stats_dir) if args.stats_dir else output_path.parent / "dataset_stats"

    config = GeneratorConfig(
        n_samples=args.n_samples,
        sale_ratio=args.sale_ratio,
        random_seed=args.random_seed,
        feature_noise=args.feature_noise,
        label_noise=args.label_noise,
        contradiction_rate=args.contradiction_rate,
        start_date=args.start_date,
        days=args.days,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_metadata=not args.without_metadata,
    )

    logger.info("=" * 88)
    logger.info("Production synthetic sales dataset generation started")
    logger.info("Configuration: %s", json.dumps(asdict(config), indent=2))
    logger.info("Output: %s", output_path)
    if training_output_path:
        logger.info("Training output: %s", training_output_path)
    logger.info("Stats dir: %s", stats_dir)
    logger.info("=" * 88)

    generator = ProductionSyntheticSalesGenerator(config)
    try:
        df = generator.generate()
        validation = validate_dataset(df, target_ratio=config.sale_ratio)

        if not validation["valid"]:
            logger.error("Dataset validation failed.")
            for err in validation["errors"]:
                logger.error("  - %s", err)
            sys.exit(1)

        if validation["warnings"]:
            logger.warning("Validation warnings:")
            for warning in validation["warnings"]:
                logger.warning("  - %s", warning)

        save_outputs(df, output_path, training_output_path, stats_dir, config, validation)

        sale_ratio = float(df[LABEL_COLUMN].mean())
        logger.info("Generation complete.")
        logger.info("Rows: %d, Columns: %d", len(df), df.shape[1])
        logger.info("Achieved conversion ratio: %.4f", sale_ratio)
        logger.info("Recommended training command:")
        if training_output_path:
            logger.info(
                "python backend/scripts/train_sale_predictor.py --csv_path %s --output_dir backend/models --early_stopping_rounds 20 --optimize_threshold",
                training_output_path,
            )
        else:
            logger.info(
                "python backend/scripts/train_sale_predictor.py --csv_path %s --output_dir backend/models --early_stopping_rounds 20 --optimize_threshold",
                output_path,
            )
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
