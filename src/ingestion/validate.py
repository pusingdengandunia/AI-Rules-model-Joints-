"""Step 3 — record-level validation.

Every record gets a primary `quality_flag` and a full `quality_issues` list.
Nothing is dropped or silently corrected here: invalid values stay as received
(in their original column) so problems stay traceable. Cleaning happens in
src/preprocessing/clean.py, driven by these flags.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.schema import (
    FUTURE_TOLERANCE_S,
    LATE_THRESHOLD_S,
    MIN_TIMESTAMP,
    NUMERIC_COLUMNS,
    QUALITY_FLAGS,
    REQUIRED_COLUMNS,
    UNUSABLE_FLAGS,
    VALID_RANGES,
)

# ISO-8601 string ending in Z or a +hh:mm / -hh:mm offset.
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def _parse_utc(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse to UTC. Returns (timestamps, naive_mask). Naive inputs are assumed UTC and reported."""
    if pd.api.types.is_datetime64_any_dtype(values):
        naive = pd.Series(values.dt.tz is None, index=values.index)
        ts = values.dt.tz_localize("UTC") if values.dt.tz is None else values.dt.tz_convert("UTC")
        return ts, naive
    text = values.astype("string").str.strip()
    naive = text.notna() & (text != "") & ~text.str.contains(_TZ_SUFFIX, na=False)
    ts = pd.to_datetime(text, utc=True, errors="coerce", format="ISO8601")
    return ts, naive.fillna(False).astype(bool)


def validate_sensor_data(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Flag every record. Returns a copy sorted by patient, timestamp, arrival order.

    Added columns:
      ingest_seq     arrival order (row order of the input)
      timestamp_utc  parsed timestamp (NaT if invalid)
      received_utc   parsed received_at (NaT if absent)
      quality_issues ';'-joined list of every issue found
      quality_flag   highest-priority issue, or 'good'
      is_usable      False for records the clean series must exclude
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)
    out["ingest_seq"] = np.arange(len(out))
    issues: dict[str, list[pd.Series]] = {}

    def add(name: str, mask: pd.Series) -> None:
        issues.setdefault(name, []).append(mask.fillna(False).astype(bool))

    # Timestamp: parseable, timezone-aware, plausible.
    ts, naive = _parse_utc(out["timestamp"])
    if "received_at" in out.columns:
        recv, _ = _parse_utc(out["received_at"])
    else:
        recv = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    reference = recv.fillna(now)
    future = ts > reference + pd.Timedelta(seconds=FUTURE_TOLERANCE_S)
    bad_ts = ts.isna() | (ts < pd.Timestamp(MIN_TIMESTAMP)) | future
    add("invalid_timestamp", bad_ts)
    add("naive_timezone", naive & ~bad_ts)  # informational: assumed UTC
    out["timestamp_utc"] = ts.where(~bad_ts)
    out["received_utc"] = recv

    # Patient id.
    pid = out["patient_id"].astype("string").str.strip()
    add("missing_patient_id", pid.isna() | (pid == ""))
    out["patient_id"] = pid.replace("", pd.NA)

    # Sensor values: numeric and within technical bounds. Original text is kept on failure.
    value_issue = pd.Series(False, index=out.index)
    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            continue
        raw = out[col]
        num = pd.to_numeric(raw, errors="coerce")
        given = raw.notna() & (raw.astype("string").str.strip() != "")
        non_numeric = given & num.isna()
        lo, hi = VALID_RANGES[col]
        out_of_range = num.notna() & ~num.between(lo, hi)
        add(f"non_numeric:{col}", non_numeric)
        add(f"out_of_range:{col}", out_of_range)
        value_issue |= non_numeric | out_of_range
        if not non_numeric.any():
            out[col] = num  # safe to store as numbers; out-of-range values are kept as received
    add("invalid_value", value_issue)
    hr_num = pd.to_numeric(out["heart_rate"], errors="coerce")
    hr_given = out["heart_rate"].notna() & (out["heart_rate"].astype("string").str.strip() != "")
    add("missing_value", ~hr_given)

    # Duplicates: same patient + timestamp. The last-received record wins (spec: keep="last");
    # earlier copies are flagged, not deleted.
    key_ok = out["timestamp_utc"].notna() & out["patient_id"].notna()
    order = out.assign(_recv=recv.fillna(pd.Timestamp.min.tz_localize("UTC"))).sort_values(["_recv", "ingest_seq"])
    dup_mask = pd.Series(False, index=out.index)
    keyed = order[key_ok.loc[order.index]]
    dup_mask.loc[keyed.index] = keyed.duplicated(subset=["patient_id", "timestamp_utc"], keep="last")
    add("duplicate", dup_mask)
    # A duplicate group whose heart_rate values disagree is worth knowing about.
    if dup_mask.any():
        n_values = hr_num.loc[keyed.index].groupby([keyed["patient_id"], keyed["timestamp_utc"]]).transform(
            lambda s: s.nunique(dropna=False)
        )
        add("duplicate_conflict", (dup_mask.loc[keyed.index] & (n_values > 1)).reindex(out.index, fill_value=False))

    # Late arrival and out-of-order arrival (per patient, in arrival order).
    lag = (recv - out["timestamp_utc"]).dt.total_seconds()
    add("late", lag > LATE_THRESHOLD_S)
    # Arrival order = received_at when known, else input row order.
    arrival = out.assign(_recv=recv)[key_ok].sort_values(["_recv", "ingest_seq"], na_position="first")
    prev_max = arrival.groupby("patient_id")["timestamp_utc"].cummax().groupby(arrival["patient_id"]).shift(1)
    ooo = arrival["timestamp_utc"] < prev_max
    add("out_of_order", ooo.reindex(out.index, fill_value=False))

    # Combine.
    merged = {name: np.logical_or.reduce([m.to_numpy() for m in masks]) for name, masks in issues.items()}
    names = list(merged)
    matrix = np.column_stack([merged[n] for n in names]) if names else np.zeros((len(out), 0), bool)
    out["quality_issues"] = [";".join(n for n, hit in zip(names, row) if hit) for row in matrix]
    flag = np.full(len(out), "good", dtype=object)
    for f in reversed(QUALITY_FLAGS[:-1]):  # lowest priority first, higher priority overwrites
        if f in merged:
            flag[merged[f]] = f
    out["quality_flag"] = flag
    out["is_usable"] = ~out["quality_flag"].isin(UNUSABLE_FLAGS)

    return out.sort_values(["patient_id", "timestamp_utc", "ingest_seq"], na_position="last").reset_index(drop=True)


def quality_summary(validated: pd.DataFrame) -> dict:
    """Counts per flag and per issue, for the data quality report."""
    issue_counts = validated["quality_issues"].str.split(";").explode()
    issue_counts = issue_counts[issue_counts != ""].value_counts()
    return {
        "records": int(len(validated)),
        "usable": int(validated["is_usable"].sum()),
        "flags": validated["quality_flag"].value_counts().to_dict(),
        "issues": issue_counts.to_dict(),
    }
