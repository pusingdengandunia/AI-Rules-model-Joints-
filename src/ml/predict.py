"""Langkah 5 / spec Step 10 — inference: one window of features -> status in the output contract.

    predictor = Predictor(mode="rules")   # demo default (Target A)
    predictor = Predictor(mode="model")   # WESAD stress model only (Target B, experimental)
    predictor = Predictor(mode="mix")     # rules decide; the model may add a "monitor" the rules missed
    result = predictor.predict(features)  # features from compute_window_features()

Responsibilities (spec Step 10):
  1. accept the features of one window
  2. check required features (order taken from models/model_card.json for the model)
  3. return insufficient_data for poor-quality windows instead of a guess
  4. run the rules and/or the model
  5. map the result onto the categories of docs/ml_target.md Bagian 3
  6. always return model_version and data_quality

Mix mode, in order:
  - rules say insufficient_data / monitor / elevated -> keep the rules result
    (reason "model_agrees" is added when the model also sees stress)
  - rules say normal because of movement, post-exercise recovery or night time -> keep normal:
    the WESAD model reads movement as stress and knows nothing about sleep
  - rules say normal, patient truly still (accel_magnitude_std < MIX_MODEL_MAX_ACCEL_STD) and the
    model sees a stress pattern -> monitor ("model_stress_pattern")
  The model alone never raises "elevated" and never triggers an alert notification by itself.

If the model or its card cannot be loaded, or they disagree on the feature order, "model" and
"mix" fall back to rules_v1 (reason "fallback_rules") instead of crashing.
The output format (OUTPUT_KEYS) is identical in every mode.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd

from src.ml.rules import RULES_CONFIG, RULES_VERSION, RuleEngine, assess_window, in_sleep_window

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
OUTPUT_KEYS = ["patient_id", "window_end", "risk_level", "reasons", "model_version", "data_quality"]
# No numeric score on purpose: rules have none, and the model's predict_proba is not calibrated,
# so any number would be read as a clinical "risk" (spec Step 10/12).
MODES = ("rules", "model", "mix")
MODEL_IGNORED_REASONS = {"activity", "post_activity"}
# The WESAD model learned "movement = stress" (subjects gesture during the TSST), so in mix mode it
# may only add a hint when the patient is truly still. Rest is ~0.01 g; light walking ~0.05-0.1 g.
MIX_MODEL_MAX_ACCEL_STD = 0.03


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class Predictor:
    def __init__(self, mode: str = "rules", model_dir: Path = MODELS_DIR, rules_config: dict | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.requested_mode = mode
        self.mode = mode
        self.rules = RuleEngine(config=dict(rules_config or RULES_CONFIG))
        self.model = None
        self.card: dict | None = None
        self.fallback_reason: str | None = None
        if mode in ("model", "mix"):
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
        if self.mode == "model":
            return self.card["model_version"]
        if self.mode == "mix":
            return f"mix:{RULES_VERSION}+{self.card['model_version']}"
        return RULES_VERSION

    @property
    def required_features(self) -> list[str]:
        rules = ["hr_delta_pct", "hrv_delta_pct", "accel_magnitude_std"]
        if self.mode == "rules":
            return rules
        if self.mode == "model":
            return list(self.card["features"])
        return rules + [f for f in self.card["features"] if f not in rules]

    def reset(self, patient_id) -> None:
        self.rules.reset(patient_id)

    def predict(self, features: dict) -> dict:
        if self.mode == "rules":
            out = self.rules.predict(features)
        elif self.mode == "model":
            out = self._predict_model(features)
        else:
            out = self._predict_mix(features)
        if self.fallback_reason:
            out["reasons"] = list(out["reasons"]) + ["fallback_rules"]
        out["model_version"] = self.model_version
        return {k: out[k] for k in OUTPUT_KEYS}

    # --- model -------------------------------------------------------------------------

    def _model_vote(self, f: dict) -> tuple[bool | None, list[str]]:
        """(stress?, reasons). None when the model cannot judge this window."""
        kind, reasons = assess_window(f)
        if kind == "insufficient" and reasons != ["no_baseline"]:
            return None, reasons
        missing = [n for n in self.card["features"] if _missing(f.get(n))]
        if missing:
            return None, [f"missing_feature:{n}" for n in missing]
        X = pd.DataFrame([[f[n] for n in self.card["features"]]], columns=self.card["features"])
        return bool(self.model.predict_proba(X)[0, 1] >= self.card["threshold"]), []

    def _predict_model(self, f: dict) -> dict:
        out = {"patient_id": f.get("patient_id"),
               "window_end": str(f["window_end"]) if f.get("window_end") is not None else None,
               "data_quality": f.get("data_quality")}
        stress, reasons = self._model_vote(f)
        if stress is None:
            return {**out, "risk_level": "insufficient_data", "reasons": reasons}
        return {**out, "risk_level": "monitor" if stress else "normal",
                "reasons": ["model_stress_pattern"] if stress else []}

    # --- mix -----------------------------------------------------------------------------

    def _predict_mix(self, f: dict) -> dict:
        out = self.rules.predict(f)  # always run: keeps the rules' streaks consistent
        level, reasons = out["risk_level"], list(out["reasons"])
        if level == "insufficient_data":
            return out
        night = in_sleep_window(f.get("window_end"), self.rules.config, f.get("timezone"))
        if level in ("monitor", "elevated_physiological_anomaly"):
            stress, _ = self._model_vote(f) if not night else (None, [])
            if stress:
                out["reasons"] = reasons + ["model_agrees"]
            return out
        if night or MODEL_IGNORED_REASONS & set(reasons):
            return out
        accel = f.get("accel_magnitude_std")
        if _missing(accel) or accel >= MIX_MODEL_MAX_ACCEL_STD:
            return out
        stress, _ = self._model_vote(f)
        if stress:
            out["risk_level"], out["reasons"] = "monitor", ["model_stress_pattern"]
        return out


_default: Predictor | None = None


def predict_risk(features: dict, predictor: Predictor | None = None) -> dict:
    """Spec Step 10 entry point. Uses a shared rules predictor unless one is given."""
    global _default
    if predictor is None:
        _default = _default or Predictor("rules")
        predictor = _default
    return predictor.predict(features)
