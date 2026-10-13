"""WESAD training dataset: filters and leakage guards."""

import pandas as pd

from src.ml.dataset import EXCLUDED, FEATURE_SETS, MIN_PURITY, build_wesad_dataset


def fake_features():
    rows = []
    for pid in ("WESAD_S2", "WESAD_S3"):
        for i, (label, purity, calib) in enumerate([
            ("baseline", 1.0, True), ("baseline", 1.0, False), ("stress", 1.0, False),
            ("stress", 0.6, False), ("transient", 1.0, False), ("meditation", 0.95, False),
        ]):
            rows.append(dict(patient_id=pid, window_start=pd.Timestamp("2017-01-01") + pd.Timedelta(minutes=i),
                             window_end=pd.Timestamp("2017-01-01") + pd.Timedelta(minutes=i + 5),
                             gt_label=label, gt_label_purity=purity, is_baseline_period=calib,
                             data_quality="good", hr_mean=70.0, hr_std=2.0, hr_delta_pct=0.0,
                             accel_magnitude_std=0.01, eda_mean=2.0 + i, skin_temp_mean=33.0,
                             eda_delta=float(i), skin_temp_delta=0.0,
                             valid_sample_ratio=1.0, hrv_rmssd_mean=50.0))
    return pd.DataFrame(rows)


def test_filters_purity_classes_and_calibration():
    ds = build_wesad_dataset(fake_features())
    assert set(ds["gt_label"]) == {"baseline", "stress", "meditation"}
    assert (ds["gt_label_purity"] >= MIN_PURITY).all()
    assert len(ds) == 2 * 3  # per subject: non-calib baseline, pure stress, meditation


def test_target_is_stress():
    ds = build_wesad_dataset(fake_features())
    assert (ds["y"] == (ds["gt_label"] == "stress")).all()


def test_no_leaky_or_ground_truth_features():
    for name, feats in FEATURE_SETS.items():
        assert not any(f.startswith("gt_") for f in feats), name
        assert not set(feats) & {k for k in EXCLUDED if not k.endswith("*")}, name


def test_all_feature_set_columns_present():
    ds = build_wesad_dataset(fake_features())
    for feats in FEATURE_SETS.values():
        assert set(feats) <= set(ds.columns)
