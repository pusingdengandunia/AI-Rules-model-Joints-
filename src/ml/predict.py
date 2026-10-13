"""Langkah 5 / spec Step 10 — inference: one window of features -> status in the output contract.

    predictor = Predictor(mode="rules")          # demo default (Target A)
    predictor = Predictor(mode="model")          # WESAD stress model (Target B, experimental)
    result = predictor.predict(features)          # features from compute_window_features()

Responsibilities (spec Step 10):
  1. accept the features of one window
  2. check required features (order taken from models/model_card.json for the model)
  3. return insufficient_data for poor-quality windows instead of a guess
  4. run the rules or the model
  5. map the result onto the categories of docs/ml_target.md Bagian 3
  6. always return model_version and data_quality

If the model or its card cannot be loaded, or they disagree on the feature order, the
predictor falls back to rules_v1 (reason "fallback_rules") instead of crashing.
The output format is identical in both modes, so the app never needs to know which ran.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd

from src.ml.rules import RULES_CONFIG, RuleEngine, assess_window

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
OUTPUT_KEYS = ["patient_id", "window_end", "risk_level", "risk_score", "reasons", "model_version", "data_quality"]


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class Predictor:
    def __init__(self, mode: str = "rules", model_dir: Path = MODELS_DIR, rules_config: dict | None = None):
        if mode not in ("rules", "model"):
            raise ValueError("mode must be 'rules' or 'model'")
        self.requested_mode = mode
        self.mode = mode
        self.rules = RuleEngine(config=dict(rules_config or RULES_CONFIG))
        self.model = None
        self.card: dict | None = None
        self.fallback_reason: str | None = None
        if mode == "model":
            self._load(Path(model_dir))

    def _load(self, model_dir: Path) -> None:
        try:
            card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
            model = joblib.load(model_dir / "model.joblib")
            trained_order = list(getattr(model, "feature_names_in_", card["features"]))
            if trained_order != list(card["features"]):
                raise ValueError(f"feature order mismatch: card {card['features']} vs model {trained_order}")
            self.model, self.card = model, card
        except Exception as exc:  # missing file, corrupt pickle, mismatch: never crash the service
            self.mode = "rules"
            self.fallback_reason = f"{type(exc).__name__}: {exc}"

    @property
    def model_version(self) -> str:
        return self.card["model_version"] if self.mode == "model" else "rules_v1"

    @property
    def required_features(self) -> list[str]:
        return list(self.card["features"]) if self.mode == "model" else ["hr_delta_pct", "accel_magnitude_std"]

    def reset(self, patient_id) -> None:
        self.rules.reset(patient_id)

    def predict(self, features: dict) -> dict:
        out = self.rules.predict(features) if self.mode == "rules" else self._predict_model(features)
        if self.fallback_reason:
            out["reasons"] = list(out["reasons"]) + ["fallback_rules"]
        return {k: out[k] for k in OUTPUT_KEYS}

    def _predict_model(self, f: dict) -> dict:
        out = {"patient_id": f.get("patient_id"),
               "window_end": str(f["window_end"]) if f.get("window_end") is not None else None,
               "risk_score": None,  # predict_proba is not calibrated: never shown as risk
               "model_version": self.model_version, "data_quality": f.get("data_quality")}
        kind, reasons = assess_window(f)
        if kind == "insufficient" and reasons != ["no_baseline"]:
            return {**out, "risk_level": "insufficient_data", "reasons": reasons}
        missing = [n for n in self.card["features"] if _missing(f.get(n))]
        if missing:
            return {**out, "risk_level": "insufficient_data", "reasons": [f"missing_feature:{n}" for n in missing]}
        X = pd.DataFrame([[f[n] for n in self.card["features"]]], columns=self.card["features"])
        stress = self.model.predict_proba(X)[0, 1] >= self.card["threshold"]
        return {**out, "risk_level": "monitor" if stress else "normal",
                "reasons": ["model_stress_pattern"] if stress else []}


_default: Predictor | None = None


def predict_risk(features: dict, predictor: Predictor | None = None) -> dict:
    """Spec Step 10 entry point. Uses a shared rules predictor unless one is given."""
    global _default
    if predictor is None:
        _default = _default or Predictor("rules")
        predictor = _default
    return predictor.predict(features)
