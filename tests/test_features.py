"""Step 5 checklist: feature table matches the contract; batch and online give identical features."""

import json

import numpy as np
import pandas as pd

from src.features.engineer import (
    apply_baseline,
    build_feature_table,
    compute_window_features,
    load_baselines,
    save_baselines,
    summarize_window,
)
from src.ingestion.validate import validate_sensor_data
from src.preprocessing.clean import clean_sensor_data
from src.schema import FEATURE_TABLE_COLUMNS, MODEL_FEATURES
from tests.conftest import make_records


def features_for(df, **kw):
    clean = clean_sensor_data(validate_sensor_data(df))
    return clean, *build_feature_table(clean, **kw)


def test_feature_table_columns_match_contract():
    _, feats, _ = features_for(make_records(n=3600))
    assert list(feats.columns[: len(FEATURE_TABLE_COLUMNS)]) == FEATURE_TABLE_COLUMNS
    assert set(MODEL_FEATURES) <= set(feats.columns)
    assert set(feats["data_quality"]) <= {"good", "insufficient_data"}


def test_feature_values_are_sensible():
    _, feats, base = features_for(make_records(n=3600, hr=70))
    good = feats[feats["data_quality"] == "good"]
    assert good["hr_mean"].between(68, 72).all()
    assert good["valid_sample_ratio"].between(0.99, 1.0).all()
    assert good["accel_magnitude_std"].lt(0.05).all()
    assert base.iloc[0]["valid"] and abs(base.iloc[0]["hr_baseline"] - 70) < 1


def test_delta_pct_relative_to_personal_baseline():
    calm = make_records(n=1800, hr=60)
    excited = make_records(n=1800, hr=90, start="2026-01-05T08:30:00Z")
    _, feats, base = features_for(pd.concat([calm, excited], ignore_index=True))
    late = feats[feats["window_start"] >= pd.Timestamp("2026-01-05T08:35:00Z")]
    assert abs(base.iloc[0]["hr_baseline"] - 60) < 1
    assert late["hr_delta_pct"].between(45, 55).all()  # 90 vs 60 bpm = +50%
    assert feats.loc[feats["is_baseline_period"], "window_end"].max() <= pd.Timestamp("2026-01-05T08:30:00Z")


def test_too_few_samples_gives_none_not_zero():
    rows = make_records(n=1)
    f = summarize_window(rows, expected_samples=300)
    assert f["hr_mean"] is None and f["hr_std"] is None
    assert f["data_quality"] == "insufficient_data"


def test_no_baseline_gives_none_delta():
    f = apply_baseline({"hr_mean": 80.0, "hrv_rmssd_mean": 40.0}, None)
    assert f["hr_delta_pct"] is None and f["hrv_delta_pct"] is None
    f = apply_baseline({"hr_mean": 80.0, "hrv_rmssd_mean": 40.0}, {"valid": False, "hr_baseline": 70})
    assert f["hr_delta_pct"] is None


def test_online_and_batch_features_are_identical(tmp_path):
    clean, feats, base = features_for(make_records(n=2400))
    save_baselines(base, tmp_path / "b.json")
    baseline = load_baselines(tmp_path / "b.json")["P1"]
    row = feats.iloc[-1]
    window = clean[(clean["timestamp_utc"] >= row["window_start"]) & (clean["timestamp_utc"] < row["window_end"])]
    online = compute_window_features(window, baseline, window_end=row["window_end"], expected_samples=300)
    for col in MODEL_FEATURES + ["data_quality", "staleness_s"]:
        a, b = online[col], row[col]
        assert (a == b) or (isinstance(a, float) and np.isclose(a, b)), col


def test_baselines_json_round_trip(tmp_path):
    _, _, base = features_for(make_records(n=2400))
    save_baselines(base, tmp_path / "b.json")
    data = json.loads((tmp_path / "b.json").read_text())
    assert data[0]["patient_id"] == "P1" and data[0]["method"] == "calibration"
