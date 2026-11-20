"""Load the trusted fitted pipeline once and provide strict feature-schema inference."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


class ChurnPredictor:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or os.getenv("AURA_MODEL_PATH", ROOT / "ml/artifacts/best_model.pkl"))
        if not self.model_path.is_absolute(): self.model_path = ROOT / self.model_path
        if not self.model_path.exists(): raise FileNotFoundError(f"Trained model artifact not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        metadata_path = self.model_path.parent / "model_metadata.json"
        schema_path = self.model_path.parent / "feature_schema.json"
        if not metadata_path.exists() or not schema_path.exists(): raise FileNotFoundError("Model metadata/schema missing")
        self.metadata = json.loads(metadata_path.read_text())
        schema = json.loads(schema_path.read_text())
        self.features = schema["features"]
        self.numeric_features = set(schema.get("numeric", []))
        configured = os.getenv("AURA_PREDICTION_THRESHOLD", "").strip()
        try:
            self.threshold = float(configured) if configured else float(self.metadata.get("threshold", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError("AURA_PREDICTION_THRESHOLD must be a number between 0 and 1") from exc
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("AURA_PREDICTION_THRESHOLD must be a finite number between 0 and 1")

    def predict(self, customer_id: str, features: dict) -> dict:
        if not isinstance(customer_id, str) or not customer_id.strip(): raise ValueError("customer_id must be a non-empty string")
        if not isinstance(features, dict): raise ValueError("features must be an object")
        missing = sorted(set(self.features) - set(features))
        extra = sorted(set(features) - set(self.features))
        if missing or extra: raise ValueError(f"Invalid feature schema; missing={missing}, extra={extra}")
        for name in self.numeric_features.intersection(self.features):
            try:
                value = float(features[name])
                if not np.isfinite(value): raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"Invalid numeric feature: {name}") from None
        frame = pd.DataFrame([{key: features[key] for key in self.features}], columns=self.features)
        probability = float(self.model.predict_proba(frame)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1: raise RuntimeError("Model returned an invalid probability")
        level = "CRITICAL" if probability >= max(0.85, self.threshold) else "HIGH" if probability >= self.threshold else "MEDIUM" if probability >= self.threshold * 0.75 else "LOW"
        return {"customer_id": customer_id, "churn_probability": probability, "risk_score": round(probability * 100, 2),
                "risk_level": level, "predicted_churn": probability >= self.threshold,
                "decision_threshold": self.threshold, "model_version": self.metadata["model_version"],
                "prediction_timestamp": datetime.now(timezone.utc).isoformat()}
