"""Step 4 checklist: missing data is never turned into a normal reading."""

import numpy as np
import pandas as pd

from src.features.engineer import build_feature_table
from src.ingestion.validate import validate_sensor_data
from src.preprocessing.clean import clean_sensor_data, detect_gaps, window_bounds
from src.schema import WINDOW_S
from tests.conftest import make_records


def pipeline(df):
    clean = clean_sensor_data(validate_sensor_data(df))
    return clean, build_feature_table(clean)[0]


def test_unusable_records_are_excluded(records):
    df = records.copy()
    df.loc[5, "timestamp"] = "garbage"
    df.loc[6, "patient_id"] = ""
    clean, _ = pipeline(df)
    assert len(clean) == len(df) - 2


def test_invalid_values_become_nan_not_replaced(records):
    df = records.copy()
    df.loc[9, "heart_rate"] = 300
    clean, _ = pipeline(df)
    row = clean[clean["ingest_seq"] == 9].iloc[0]
    assert np.isnan(row["heart_rate"])
    assert row["hrv_rmssd"] == records.loc[9, "hrv_rmssd"]  # other signals of the record survive


def test_missing_values_are_not_filled_with_zero(records):
    df = records.copy()
    df.loc[100:400, "heart_rate"] = None
    clean, _ = pipeline(df)
    assert clean["heart_rate"].isna().sum() == 301
    assert (clean["heart_rate"].dropna() > 0).all()


def test_disconnect_window_is_insufficient_not_normal():
    df = make_records(n=1800)
    df = df[(df.index < 600) | (df.index >= 1200)]  # 10 minutes without any record
    clean, feats = pipeline(df.reset_index(drop=True))
    gaps = detect_gaps(clean)
    assert len(gaps) == 1 and gaps.iloc[0]["duration_s"] == 601
    inside = feats[(feats["window_start"] >= pd.Timestamp("2026-01-05T08:10:00Z"))
                   & (feats["window_end"] <= pd.Timestamp("2026-01-05T08:20:00Z"))]
    assert len(inside) > 0
    assert (inside["data_quality"] == "insufficient_data").all()
    assert inside["hr_mean"].isna().all()


def test_recent_disconnect_is_insufficient_even_with_enough_old_samples():
    from src.features.engineer import compute_window_features
    clean, _ = pipeline(make_records(n=600))
    now = clean["timestamp_utc"].max() + pd.Timedelta(seconds=120)  # sensor silent for 2 min
    window = clean[clean["timestamp_utc"] >= now - pd.Timedelta(seconds=300)]
    f = compute_window_features(window, None, window_end=now)
    assert f["valid_sample_ratio"] >= 0.5          # 3 of 5 minutes have data...
    assert f["data_quality"] == "insufficient_data"  # ...but nothing recent: not "normal"
    assert f["staleness_s"] >= 120


def test_off_wrist_nulls_make_windows_insufficient():
    df = make_records(n=1200)
    df.loc[300:900, ["heart_rate", "hrv_rmssd"]] = None  # records arrive, physiology missing
    _, feats = pipeline(df)
    mid = feats[(feats["window_start"] >= pd.Timestamp("2026-01-05T08:05:00Z"))
                & (feats["window_end"] <= pd.Timestamp("2026-01-05T08:15:00Z"))]
    assert (mid["data_quality"] == "insufficient_data").all()


def test_sessions_derived_from_long_gaps_when_absent():
    a = make_records(n=600, start="2026-01-05T08:00:00Z", session=None)
    b = make_records(n=600, start="2026-01-05T12:00:00Z", session=None)
    clean, _ = pipeline(pd.concat([a, b], ignore_index=True))
    assert clean["session_id"].nunique() == 2


def test_windows_are_clock_aligned_and_inside_session():
    start = pd.Timestamp("2026-01-05T08:00:30Z")
    end = pd.Timestamp("2026-01-05T08:20:00Z")
    wins = window_bounds(start, end)
    assert wins[0][0] == pd.Timestamp("2026-01-05T08:01:00Z")
    assert all(ws >= start and we <= end + pd.Timedelta(seconds=1) for ws, we in wins)
    assert all((we - ws).total_seconds() == WINDOW_S for ws, we in wins)
