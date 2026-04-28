"""Learned pattern detector for high-frequency OHLCV windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class LearnedPatternPrediction:
    """Prediction payload from learned detector."""

    pattern_name: str
    confidence: float
    direction: str
    probabilities: Dict[str, float]


class LearnedPatternDetector:
    """A compact supervised model that learns pattern labels from windowed OHLCV."""

    PATTERN_CLASSES = [
        "none",
        "triangle",
        "wedge_rising",
        "wedge_falling",
        "flag",
        "channel",
        "head_shoulders",
    ]

    BULLISH = {"triangle", "wedge_falling", "flag", "channel"}
    BEARISH = {"wedge_rising", "head_shoulders"}

    def __init__(self, lookback: int, min_confidence: float) -> None:
        self.lookback = lookback
        self.min_confidence = min_confidence
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=120,
            random_state=42,
        )
        self.is_trained = False

    def fit(
        self,
        df: pd.DataFrame,
        rule_pattern_provider,
        max_samples: int = 3000,
        forward_horizon: int = 6,
        forward_return_threshold: float = 0.0012,
    ) -> bool:
        """Train classifier on rule-labeled windows."""
        if df.empty or len(df) < self.lookback + 5:
            return False

        X, y = self._build_dataset(
            df,
            rule_pattern_provider,
            max_samples=max_samples,
            forward_horizon=forward_horizon,
            forward_return_threshold=forward_return_threshold,
        )
        if len(X) < 100:
            return False

        X, y = self._rebalance_dataset(X, y)
        if len(X) < 100:
            return False

        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.is_trained = True
        return True

    def save(self, model_path: str) -> None:
        """Persist the learned model and scaler to disk."""
        payload = {
            "lookback": self.lookback,
            "min_confidence": self.min_confidence,
            "scaler": self.scaler,
            "model": self.model,
            "is_trained": self.is_trained,
        }
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, model_path: str) -> bool:
        """Load previously saved learned model and scaler."""
        path = Path(model_path)
        if not path.exists():
            return False

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.lookback = int(payload["lookback"])
        self.min_confidence = float(payload["min_confidence"])
        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.is_trained = bool(payload["is_trained"])
        return self.is_trained

    def predict(self, window: pd.DataFrame) -> LearnedPatternPrediction:
        """Predict the most likely pattern for a single window."""
        if (not self.is_trained) or len(window) < self.lookback:
            return LearnedPatternPrediction(
                pattern_name="none",
                confidence=0.0,
                direction="neutral",
                probabilities={c: 0.0 for c in self.PATTERN_CLASSES},
            )

        x = self._window_to_features(window).reshape(1, -1)
        xs = self.scaler.transform(x)
        probs = self.model.predict_proba(xs)[0]
        classes = list(self.model.classes_)

        prob_map = {c: 0.0 for c in self.PATTERN_CLASSES}
        for idx, cls in enumerate(classes):
            prob_map[str(cls)] = float(probs[idx])

        pattern_name = max(prob_map, key=prob_map.get)
        confidence = prob_map[pattern_name]

        if pattern_name in self.BULLISH:
            direction = "bullish"
        elif pattern_name in self.BEARISH:
            direction = "bearish"
        else:
            direction = "neutral"

        return LearnedPatternPrediction(
            pattern_name=pattern_name,
            confidence=confidence,
            direction=direction,
            probabilities=prob_map,
        )

    def feature_vector(self, window: pd.DataFrame) -> np.ndarray:
        """Return compact learned features for RL state (8 dims)."""
        pred = self.predict(window)

        # 6 pattern probs (excluding 'none') + confidence + direction
        probs = pred.probabilities
        vec = [
            probs.get("triangle", 0.0),
            probs.get("wedge_rising", 0.0),
            probs.get("wedge_falling", 0.0),
            probs.get("flag", 0.0),
            probs.get("channel", 0.0),
            probs.get("head_shoulders", 0.0),
            pred.confidence,
            1.0 if pred.direction == "bullish" else (-1.0 if pred.direction == "bearish" else 0.0),
        ]
        return np.asarray(vec, dtype=np.float32)

    def _build_dataset(
        self,
        df: pd.DataFrame,
        rule_pattern_provider,
        max_samples: int = 3000,
        forward_horizon: int = 6,
        forward_return_threshold: float = 0.0012,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised dataset from rolling windows with rule-generated labels."""
        X: List[np.ndarray] = []
        y: List[str] = []

        # Evenly subsample long histories to keep fit time short.
        start = self.lookback
        end = len(df) - max(1, forward_horizon)
        step = max(1, int((end - start) / max_samples))

        for i in range(start, end, step):
            window = df.iloc[i - self.lookback : i]
            if len(window) < self.lookback:
                continue

            x = self._window_to_features(window)
            pattern_name = self._label_from_rules(window, rule_pattern_provider)
            pattern_name = self._apply_forward_return_filter(
                df,
                i,
                pattern_name,
                forward_horizon=forward_horizon,
                forward_return_threshold=forward_return_threshold,
            )
            X.append(x)
            y.append(pattern_name)

            if len(X) >= max_samples:
                break

        if not X:
            return np.empty((0, 1)), np.empty((0,))

        return np.vstack(X), np.asarray(y)

    def _label_from_rules(self, window: pd.DataFrame, rule_pattern_provider) -> str:
        """Use existing rules to produce training labels for distillation."""
        patterns = rule_pattern_provider.detect_patterns(window)
        if not patterns:
            return "none"
        return patterns[-1].name if patterns[-1].name in self.PATTERN_CLASSES else "none"

    def _apply_forward_return_filter(
        self,
        df: pd.DataFrame,
        index_pos: int,
        label: str,
        forward_horizon: int,
        forward_return_threshold: float,
    ) -> str:
        """Drop pattern labels that are not supported by near-term forward returns."""
        if label == "none":
            return label

        cur = float(df["close"].iloc[index_pos - 1])
        fut = float(df["close"].iloc[index_pos - 1 + forward_horizon])
        if cur <= 0:
            return "none"
        fwd_ret = (fut - cur) / cur

        if label in self.BULLISH and fwd_ret < forward_return_threshold:
            return "none"
        if label in self.BEARISH and fwd_ret > -forward_return_threshold:
            return "none"

        return label

    def _rebalance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mitigate class imbalance so classifier does not collapse to one class."""
        if len(X) == 0:
            return X, y

        labels, counts = np.unique(y, return_counts=True)
        if len(labels) <= 1:
            return X, y

        target = int(np.median(counts))
        target = max(80, target)

        keep_idx: List[int] = []
        for lab, count in zip(labels, counts):
            idx = np.where(y == lab)[0]
            if count > target:
                sampled = np.random.choice(idx, size=target, replace=False)
                keep_idx.extend(sampled.tolist())
            else:
                keep_idx.extend(idx.tolist())

        keep_idx = np.asarray(sorted(keep_idx), dtype=int)
        return X[keep_idx], y[keep_idx]

    def _window_to_features(self, window: pd.DataFrame) -> np.ndarray:
        """Convert OHLCV window to a fixed-size feature vector."""
        w = window.tail(self.lookback).copy()
        close = w["close"].to_numpy(dtype=np.float64)
        high = w["high"].to_numpy(dtype=np.float64)
        low = w["low"].to_numpy(dtype=np.float64)
        open_ = w["open"].to_numpy(dtype=np.float64)
        volume = w["volume"].to_numpy(dtype=np.float64)

        # Price normalization anchored to first close.
        base = close[0] if close[0] != 0 else 1.0
        close_n = close / base - 1.0
        open_n = open_ / base - 1.0
        high_n = high / base - 1.0
        low_n = low / base - 1.0

        # Candle shape and momentum features.
        body = (close - open_) / np.maximum(open_, 1e-8)
        upper_wick = (high - np.maximum(open_, close)) / np.maximum(close, 1e-8)
        lower_wick = (np.minimum(open_, close) - low) / np.maximum(close, 1e-8)
        returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)

        vol_scale = np.median(volume) if np.median(volume) > 0 else 1.0
        vol_n = volume / vol_scale - 1.0

        feat = np.concatenate(
            [
                close_n,
                open_n,
                high_n,
                low_n,
                body,
                upper_wick,
                lower_wick,
                returns,
                vol_n,
            ]
        )
        return feat.astype(np.float32)
