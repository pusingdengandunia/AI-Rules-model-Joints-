"""Step 5 — window features and personal baselines.

The same two functions serve batch (feature table for the ML engineer) and online
(API scoring one window):

    feats = summarize_window(rows, expected_samples)   # raw window statistics
    feats = apply_baseline(feats, baseline)            # adds hr_delta_pct / hrv_delta_pct

so training and inference cannot drift apart. `compute_window_features` wraps both.

Personal baseline = median of hr_mean / hrv_rmssd_mean over good windows of an explicit
calibration period. The period is configurable and recorded with the baseline:
  "calibration": first 30 min of each patient's first session (default, needs no labels)
  "label":       first half of the good windows fully inside the protocol "baseline"
                 condition (WESAD). The rest of the baseline windows stay available for evaluation.
Windows used to compute a baseline are marked is_baseline_period=True; exclude them from
model evaluation, their deltas are ~0 by construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.clean import iter_windows, window_quality
from src.schema import (
    BASELINE_CALIBRATION_S,
    BASELINE_MIN_WINDOWS,
    FEATURE_TABLE_COLUMNS,
    GROUND_TRUTH_PREFIX,
    WINDOW_S,
)

SKIP_GT = {"gt_t_s", "gt_injected"}


def _num(rows: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(rows[col], errors="coerce").dropna() if col in rows.columns else pd.Series(dtype=float)


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def summarize_window(rows: pd.DataFrame, expected_samples: float = WINDOW_S,
                     window_end: pd.Timestamp | None = None) -> dict:
    """Statistics for one window. Values are None when there is not enough data, never 0.

    window_end (online: "now") is used for staleness: seconds since the newest valid HR sample.
    """
    expected = max(float(expected_samples), 1.0)
    hr = _num(rows, "heart_rate")
    staleness = None
    if window_end is not None:
        if len(hr):
            newest = rows.loc[hr.index, "timestamp_utc"].max()
            staleness = round((window_end - newest).total_seconds(), 3)
        else:
            staleness = float(WINDOW_S)
    hrv = _num(rows, "hrv_rmssd")
    if {"accel_x", "accel_y", "accel_z"} <= set(rows.columns):
        acc = rows[["accel_x", "accel_y", "accel_z"]].apply(pd.to_numeric, errors="coerce").dropna()
        mag = np.sqrt((acc ** 2).sum(axis=1))
    else:
        mag = pd.Series(dtype=float)
    eda, temp = _num(rows, "eda"), _num(rows, "skin_temp")

    valid_ratio = min(len(hr) / expected, 1.0)
    return {
        "hr_mean": _f(hr.mean()) if len(hr) >= 2 else None,
        "hr_std": _f(hr.std()) if len(hr) >= 2 else None,
        "hrv_rmssd_mean": _f(hrv.mean()) if len(hrv) >= 2 else None,
        "accel_magnitude_std": _f(mag.std()) if len(mag) >= 2 else None,
        "valid_sample_ratio": round(valid_ratio, 4),
        "hrv_valid_ratio": round(min(len(hrv) / expected, 1.0), 4),
        "accel_valid_ratio": round(min(len(mag) / expected, 1.0), 4),
        "eda_mean": _f(eda.mean()) if len(eda) else None,
        "skin_temp_mean": _f(temp.mean()) if len(temp) else None,
        "n_samples": int(len(rows)),
        "staleness_s": staleness,
        "data_quality": window_quality(valid_ratio, staleness),
    }


def _pct_change(value, base) -> float | None:
    if value is None or base is None or not base or (isinstance(base, float) and math.isnan(base)):
        return None
    return (value - base) / base * 100.0


def _diff(value, base) -> float | None:
    if value is None or base is None or (isinstance(base, float) and math.isnan(base)):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value - base


def apply_baseline(feats: dict, baseline: dict | None) -> dict:
    """Add deltas relative to a personal baseline.

    hr_delta_pct / hrv_delta_pct (percent) need a *valid* HR baseline.
    eda_delta (uS) / skin_temp_delta (degC) are absolute differences; they only need their own
    baseline value, because EDA and temperature do not depend on the PPG quality gate.
    """
    out = dict(feats)
    b = baseline or {}
    ok = bool(b) and b.get("valid", False)
    out["hr_delta_pct"] = _pct_change(feats.get("hr_mean"), b.get("hr_baseline")) if ok else None
    out["hrv_delta_pct"] = _pct_change(feats.get("hrv_rmssd_mean"), b.get("hrv_baseline")) if ok else None
    out["eda_delta"] = _diff(feats.get("eda_mean"), b.get("eda_baseline"))
    out["skin_temp_delta"] = _diff(feats.get("skin_temp_mean"), b.get("skin_temp_baseline"))
    return out


def compute_window_features(rows: pd.DataFrame, baseline: dict | None, window_end: pd.Timestamp,
                            expected_samples: float = WINDOW_S) -> dict:
    """Online entry point: clean rows of [window_end - 5 min, window_end) -> feature dict.

    Same code as the batch table. Pass window_end = now, so a sensor that stopped
    sending is reported as insufficient_data instead of its last good values.
    """
    return apply_baseline(summarize_window(rows, expected_samples, window_end), baseline)


def summarize_labels(rows: pd.DataFrame) -> dict:
    """Aggregate ground-truth columns of a window, for evaluation only."""
    out = {}
    for col in rows.columns:
        if not col.startswith(GROUND_TRUTH_PREFIX) or col in SKIP_GT:
            continue
        s = rows[col].dropna()
        if s.empty:
            out[col] = None
        elif col == "gt_expected_alert":
            v = pd.to_numeric(s)
            out["gt_expected_alert"] = int(v.max())
            out["gt_expected_alert_frac"] = round(float(v.mean()), 4)
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = round(float(s.mean()), 4)
        else:
            counts = s.value_counts()
            out[col] = counts.index[0]
            out[f"{col}_purity"] = round(float(counts.iloc[0] / len(s)), 4)
    return out


def select_baseline_windows(features: pd.DataFrame, mode: str) -> pd.Series:
    """Boolean mask of windows forming each patient's calibration period."""
    mask = pd.Series(False, index=features.index)
    if mode == "calibration":
        first_session = features.sort_values("window_start").groupby("patient_id")["session_id"].first()
        in_first = features["session_id"] == features["patient_id"].map(first_session)
        session_start = features.groupby("session_id")["window_start"].transform("min")
        mask = in_first & (features["window_end"] <= session_start + pd.Timedelta(seconds=BASELINE_CALIBRATION_S))
    elif mode == "label":
        # First half of the good windows fully inside the protocol baseline (at least
        # BASELINE_MIN_WINDOWS when available); the rest stay available for evaluation.
        pure = (features.get("gt_label") == "baseline") & (features.get("gt_label_purity") == 1.0)
        cand = features[pure & (features["data_quality"] == "good")].sort_values("window_start")
        for _, g in cand.groupby("patient_id"):
            take = max(math.ceil(len(g) / 2), min(BASELINE_MIN_WINDOWS, len(g)))
            mask.loc[g.index[:take]] = True
    else:
        raise ValueError(f"unknown baseline mode: {mode}")
    return mask


def compute_baselines(features: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.Series]:
    """Per-patient baseline table, plus the mask of windows used for it."""
    period = select_baseline_windows(features, mode)
    usable = period & (features["data_quality"] == "good")
    rows = []
    for pid in features["patient_id"].unique():
        g = features[usable & (features["patient_id"] == pid)]
        p = features[period & (features["patient_id"] == pid)]
        hr_b = _f(g["hr_mean"].median()) if len(g) else None
        hrv_vals = g["hrv_rmssd_mean"].dropna()
        rows.append(dict(
            patient_id=pid,
            hr_baseline=hr_b,
            hrv_baseline=_f(hrv_vals.median()) if len(hrv_vals) >= BASELINE_MIN_WINDOWS else None,
            eda_baseline=_f(g["eda_mean"].median()) if g["eda_mean"].notna().any() else None,
            skin_temp_baseline=_f(g["skin_temp_mean"].median()) if g["skin_temp_mean"].notna().any() else None,
            n_windows=int(len(g)),
            valid=bool(len(g) >= BASELINE_MIN_WINDOWS and hr_b is not None),
            method=mode,
            period_start=str(p["window_start"].min()) if len(p) else None,
            period_end=str(p["window_end"].max()) if len(p) else None,
            note="prototype baseline for demo/experiments, not a clinical reference",
        ))
    return pd.DataFrame(rows), period


def build_feature_table(clean: pd.DataFrame, baseline_mode: str = "calibration",
                        baselines: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch: clean series -> (feature table, baselines). Pass `baselines` to reuse existing ones."""
    records = []
    for pid, sid, ws, we, rows, expected in iter_windows(clean):
        rec = {"patient_id": pid, "session_id": sid, "window_start": ws, "window_end": we,
               "source": rows["source"].mode().iloc[0] if "source" in rows and rows["source"].notna().any() else None}
        rec.update(summarize_window(rows, expected, we))
        rec.update(summarize_labels(rows))
        records.append(rec)
    feats = pd.DataFrame.from_records(records)
    if feats.empty:
        return pd.DataFrame(columns=FEATURE_TABLE_COLUMNS), pd.DataFrame()

    if baselines is None:
        baselines, period = compute_baselines(feats, baseline_mode)
    else:
        period = pd.Series(False, index=feats.index)
    feats["is_baseline_period"] = period.to_numpy()
    lookup = baselines.set_index("patient_id").to_dict("index")
    base_cols = ["patient_id", "hr_mean", "hrv_rmssd_mean", "eda_mean", "skin_temp_mean"]
    deltas = [apply_baseline(r, lookup.get(r["patient_id"])) for r in feats[base_cols].to_dict("records")]
    for col in ("hr_delta_pct", "hrv_delta_pct", "eda_delta", "skin_temp_delta"):
        feats[col] = [d[col] for d in deltas]

    gt_cols = [c for c in feats.columns if c.startswith(GROUND_TRUTH_PREFIX)]
    return feats[FEATURE_TABLE_COLUMNS + gt_cols], baselines


def save_baselines(baselines: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baselines.to_dict("records"), indent=2), encoding="utf-8")


def load_baselines(path: Path) -> dict[str, dict]:
    """patient_id -> baseline dict, for the online API."""
    return {b["patient_id"]: b for b in json.loads(Path(path).read_text(encoding="utf-8"))}
