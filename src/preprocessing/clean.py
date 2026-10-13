"""Step 4 — time-series preprocessing: clean series, gaps, sessions and windows.

Order (spec Step 4):
  1. sort by patient and time
  2. drop records validation marked unusable (duplicates, bad timestamp, no patient)
  3. null out values validation marked invalid (never replaced with a "plausible" value)
  4. assign sessions and find gaps / abnormal sampling intervals
  5. no filtering/smoothing: HR and RMSSD already arrive as windowed device values
  6. build sliding windows and mark the ones without enough valid samples

Missing data is never filled with zero or interpolated: a window without enough
samples is labelled insufficient_data so the app cannot show it as "normal".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import (
    GAP_FACTOR,
    MAX_STALENESS_S,
    MIN_VALID_RATIO,
    NUMERIC_COLUMNS,
    SESSION_GAP_S,
    WINDOW_S,
    WINDOW_STEP_S,
)


def clean_sensor_data(validated: pd.DataFrame) -> pd.DataFrame:
    """Usable records only, invalid values set to NaN, sorted, with session_id and t_rel_s."""
    df = validated[validated["is_usable"]].copy()
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df["quality_issues"].str.contains(f"(?:out_of_range|non_numeric):{col}(?:;|$)", regex=True)
        df.loc[bad, col] = np.nan
    df = df.sort_values(["patient_id", "timestamp_utc", "ingest_seq"]).reset_index(drop=True)
    df["session_id"] = assign_sessions(df)
    start = df.groupby("session_id")["timestamp_utc"].transform("min")
    df["t_rel_s"] = (df["timestamp_utc"] - start).dt.total_seconds()
    return df


def assign_sessions(df: pd.DataFrame) -> pd.Series:
    """Keep a provided session_id; otherwise start a new session after a gap > SESSION_GAP_S."""
    dt = df.groupby("patient_id")["timestamp_utc"].diff().dt.total_seconds()
    new = dt.isna() | (dt > SESSION_GAP_S)
    derived = df["patient_id"].astype(str) + "_s" + new.groupby(df["patient_id"]).cumsum().astype(str)
    if "session_id" in df.columns:
        given = df["session_id"].astype("string").str.strip().replace("", pd.NA)
        return given.fillna(derived).astype(str)
    return derived


def nominal_interval_s(times: pd.Series) -> float:
    """Expected sampling interval of a session (median spacing, rounded to the ms)."""
    dt = times.diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    return float(round(dt.median(), 3)) if len(dt) else 1.0


def detect_gaps(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per period without records (interval > GAP_FACTOR x nominal)."""
    rows = []
    for (pid, sid), g in clean.groupby(["patient_id", "session_id"], sort=False):
        t = g["timestamp_utc"].reset_index(drop=True)
        step = nominal_interval_s(t)
        dt = t.diff().dt.total_seconds()
        for i in np.flatnonzero(dt.to_numpy() > GAP_FACTOR * step):
            rows.append(dict(patient_id=pid, session_id=sid, gap_start=t[i - 1], gap_end=t[i],
                             duration_s=float(dt[i]), nominal_interval_s=step))
    return pd.DataFrame(rows, columns=["patient_id", "session_id", "gap_start", "gap_end", "duration_s", "nominal_interval_s"])


def interval_profile(clean: pd.DataFrame) -> pd.DataFrame:
    """Per session: nominal interval and counts of too-short / too-long spacings."""
    rows = []
    for (pid, sid), g in clean.groupby(["patient_id", "session_id"], sort=False):
        t = g["timestamp_utc"]
        step = nominal_interval_s(t)
        dt = t.diff().dt.total_seconds().dropna()
        rows.append(dict(
            patient_id=pid, session_id=sid, records=len(g), start=t.min(), end=t.max(),
            nominal_interval_s=step,
            too_dense=int((dt < 0.5 * step).sum()),
            gaps=int((dt > GAP_FACTOR * step).sum()),
            hr_missing_ratio=round(float(g["heart_rate"].isna().mean()), 4),
        ))
    return pd.DataFrame(rows)


def window_bounds(session_start: pd.Timestamp, session_end: pd.Timestamp,
                  window_s: int = WINDOW_S, step_s: int = WINDOW_STEP_S) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Clock-aligned window starts (multiples of step_s) fully inside the session."""
    first = session_start.ceil(f"{step_s}s")
    last_start = session_end - pd.Timedelta(seconds=window_s) + pd.Timedelta(seconds=1)
    if first > last_start:
        return []
    starts = pd.date_range(first, last_start, freq=f"{step_s}s")
    return [(s, s + pd.Timedelta(seconds=window_s)) for s in starts]


def iter_windows(clean: pd.DataFrame, window_s: int = WINDOW_S, step_s: int = WINDOW_STEP_S):
    """Yield (patient_id, session_id, window_start, window_end, rows, expected_samples).

    Windows are half-open [start, end). Expected samples come from the session's nominal
    interval, so a window that lost records to a gap shows a low valid_sample_ratio.
    """
    for (pid, sid), g in clean.groupby(["patient_id", "session_id"], sort=False):
        t = g["timestamp_utc"]
        step = nominal_interval_s(t)
        expected = window_s / step
        tv = t.dt.tz_localize(None).to_numpy()  # naive UTC datetime64 for searchsorted
        for ws, we in window_bounds(t.min(), t.max(), window_s, step_s):
            lo = np.searchsorted(tv, ws.tz_localize(None).to_datetime64(), side="left")
            hi = np.searchsorted(tv, we.tz_localize(None).to_datetime64(), side="left")
            yield pid, sid, ws, we, g.iloc[lo:hi], expected


def window_quality(valid_sample_ratio: float | None, staleness_s: float | None = None) -> str:
    """good only if the window has enough valid HR samples AND recent ones."""
    if valid_sample_ratio is None or np.isnan(valid_sample_ratio) or valid_sample_ratio < MIN_VALID_RATIO:
        return "insufficient_data"
    if staleness_s is not None and staleness_s > MAX_STALENESS_S:
        return "insufficient_data"
    return "good"
