"""Inference: contract, insufficient_data handling, feature order, fallback (spec §8 ML checklist)."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.ml.dataset import FEATURE_SETS
from src.ml.predict import OUTPUT_KEYS, Predictor, predict_risk

FEATURES = FEATURE_SETS["eda"]
T0 = pd.Timestamp("2026-01-05T08:10:00Z")


def window(i=0, quality="good", **overrides):
    f = dict(patient_id="P1", window_end=T0 + pd.Timedelta(minutes=i), data_quality=quality, staleness_s=1.0,
             hr_mean=70.0, hr_std=2.0, hr_delta_pct=0.0, hrv_delta_pct=0.0, accel_magnitude_std=0.01,
             eda_mean=2.0, eda_delta=0.0, skin_temp_delta=0.0)
    f.update(overrides)
    return f


@pytest.fixture
def model_dir(tmp_path):
    """A tiny model trained on fake data: high eda_delta = stress."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (200, len(FEATURES))), columns=FEATURES)
    y = (X["eda_delta"] > 0.5).astype(int)
    joblib.dump(LogisticRegression().fit(X, y), tmp_path / "model.joblib")
    (tmp_path / "model_card.json").write_text(json.dumps(
        {"model_version": "test_model_v1", "features": FEATURES, "threshold": 0.5}))
    return tmp_path


def test_rules_output_contract():
    r = Predictor("rules").predict(window())
    assert list(r) == OUTPUT_KEYS
    assert r["model_version"] == "rules_v1" and r["risk_score"] is None


def test_model_output_contract_is_identical(model_dir):
    r = Predictor("model", model_dir=model_dir).predict(window())
    assert list(r) == OUTPUT_KEYS
    assert r["model_version"] == "test_model_v1" and r["risk_score"] is None


@pytest.mark.parametrize("mode", ["rules", "model"])
def test_bad_quality_gives_insufficient_data(model_dir, mode):
    p = Predictor(mode, model_dir=model_dir)
    r = p.predict(window(quality="insufficient_data", staleness_s=300.0, eda_delta=5.0, hr_delta_pct=50.0))
    assert r["risk_level"] == "insufficient_data" and r["data_quality"] == "insufficient_data"


def test_model_missing_feature_gives_insufficient_data(model_dir):
    r = Predictor("model", model_dir=model_dir).predict(window(eda_delta=None))
    assert r["risk_level"] == "insufficient_data"
    assert "missing_feature:eda_delta" in r["reasons"]


def test_model_predictions_follow_signal(model_dir):
    p = Predictor("model", model_dir=model_dir)
    assert p.predict(window(eda_delta=3.0))["risk_level"] == "monitor"
    assert p.predict(window(eda_delta=-1.0))["risk_level"] == "normal"


def test_feature_order_comes_from_card(model_dir):
    p = Predictor("model", model_dir=model_dir)
    assert p.required_features == FEATURES == list(p.model.feature_names_in_)


def test_missing_model_falls_back_to_rules(tmp_path):
    p = Predictor("model", model_dir=tmp_path)
    assert p.mode == "rules" and p.fallback_reason
    r = p.predict(window())
    assert r["model_version"] == "rules_v1" and "fallback_rules" in r["reasons"]


def test_feature_order_mismatch_falls_back(model_dir):
    card = json.loads((model_dir / "model_card.json").read_text())
    card["features"] = list(reversed(card["features"]))
    (model_dir / "model_card.json").write_text(json.dumps(card))
    p = Predictor("model", model_dir=model_dir)
    assert p.mode == "rules" and "mismatch" in p.fallback_reason


def test_predict_risk_entry_point():
    assert predict_risk(window())["risk_level"] == "normal"


def test_real_model_card_matches_training(tmp_path):
    from src.ml.predict import MODELS_DIR
    if not (MODELS_DIR / "model.joblib").exists():
        pytest.skip("model not trained (python -m src.ml.train)")
    p = Predictor("model")
    assert p.mode == "model", p.fallback_reason
    assert p.required_features == list(p.model.feature_names_in_)
