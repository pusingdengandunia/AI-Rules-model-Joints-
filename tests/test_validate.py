"""Step 3 checklist: valid payload accepted, duplicates detected, invalid timestamps flagged,
bad values flagged without being altered, every record has a quality flag."""

import pandas as pd
import pytest

from src.ingestion.validate import validate_sensor_data
from tests.conftest import make_records


def flags(v: pd.DataFrame) -> pd.Series:
    return v.set_index("ingest_seq")["quality_flag"].sort_index()


def test_valid_payload_is_accepted(records):
    v = validate_sensor_data(records)
    assert len(v) == len(records)
    assert (v["quality_flag"] == "good").all()
    assert v["is_usable"].all()
    assert str(v["timestamp_utc"].dt.tz) == "UTC"


def test_missing_required_column_raises(records):
    with pytest.raises(ValueError, match="heart_rate"):
        validate_sensor_data(records.drop(columns="heart_rate"))


def test_every_record_gets_a_flag(records):
    df = records.copy()
    df.loc[3, "heart_rate"] = None
    df.loc[5, "timestamp"] = "garbage"
    v = validate_sensor_data(df)
    assert v["quality_flag"].notna().all()
    assert set(v["quality_flag"]) <= {"good", "missing_value", "invalid_timestamp"}


def test_duplicate_keeps_last_received_and_flags_the_rest(records):
    df = records.copy()
    resent = df.iloc[[10]].copy()
    resent["received_at"] = "2026-01-05T09:00:00Z"
    resent["heart_rate"] = 99.0
    df = pd.concat([df, resent], ignore_index=True)
    v = validate_sensor_data(df)
    f = flags(v)
    assert f[10] == "duplicate"           # earlier copy flagged, not deleted
    assert f[len(df) - 1] != "duplicate"  # later copy wins
    assert "duplicate_conflict" in v.set_index("ingest_seq").loc[10, "quality_issues"]
    assert not v.set_index("ingest_seq").loc[10, "is_usable"]


@pytest.mark.parametrize("bad", ["not-a-time", "", "1970-01-01T00:00:00Z", "2099-01-01T00:00:00Z", "2026-13-45T25:61:00Z"])
def test_invalid_timestamp_is_flagged(records, bad):
    df = records.copy()
    df.loc[7, "timestamp"] = bad
    f = flags(validate_sensor_data(df))
    assert f[7] == "invalid_timestamp"


def test_naive_timestamp_is_assumed_utc_and_reported(records):
    df = records.copy()
    df.loc[4, "timestamp"] = df.loc[4, "timestamp"].rstrip("Z")
    v = validate_sensor_data(df).set_index("ingest_seq")
    assert "naive_timezone" in v.loc[4, "quality_issues"]
    assert v.loc[4, "timestamp_utc"] == pd.Timestamp(records.loc[4, "timestamp"])


def test_offset_timestamps_are_converted_to_utc(records):
    df = records.copy()
    df.loc[0, "timestamp"] = "2026-01-05T15:00:00+07:00"  # same instant as 08:00:00Z
    v = validate_sensor_data(df).set_index("ingest_seq")
    assert v.loc[0, "timestamp_utc"] == pd.Timestamp("2026-01-05T08:00:00Z")


def test_missing_patient_id_is_flagged(records):
    df = records.copy()
    df.loc[2, "patient_id"] = "  "
    assert flags(validate_sensor_data(df))[2] == "missing_patient_id"


@pytest.mark.parametrize("col,val,issue", [
    ("heart_rate", 0, "out_of_range:heart_rate"),
    ("heart_rate", 300, "out_of_range:heart_rate"),
    ("hrv_rmssd", 900, "out_of_range:hrv_rmssd"),
    ("accel_x", 40, "out_of_range:accel_x"),
])
def test_implausible_value_is_flagged_but_not_changed(records, col, val, issue):
    df = records.copy()
    df.loc[9, col] = val
    v = validate_sensor_data(df).set_index("ingest_seq")
    assert v.loc[9, "quality_flag"] == "invalid_value"
    assert issue in v.loc[9, "quality_issues"]
    assert float(v.loc[9, col]) == val  # kept as received for traceability


def test_non_numeric_value_is_flagged(records):
    df = records.copy()
    df["heart_rate"] = df["heart_rate"].astype(object)
    df.loc[9, "heart_rate"] = "abc"
    v = validate_sensor_data(df).set_index("ingest_seq")
    assert v.loc[9, "quality_flag"] == "invalid_value"
    assert "non_numeric:heart_rate" in v.loc[9, "quality_issues"]


def test_missing_heart_rate_is_missing_not_good(records):
    df = records.copy()
    df.loc[11, "heart_rate"] = None
    assert flags(validate_sensor_data(df))[11] == "missing_value"


def test_late_and_out_of_order_arrival(records):
    df = records.copy()
    df.loc[100, "received_at"] = "2026-01-05T08:30:00Z"  # measured 08:01:40, arrives 28 min later
    v = validate_sensor_data(df).set_index("ingest_seq")
    assert v.loc[100, "quality_flag"] == "late"
    assert "out_of_order" in v.loc[100, "quality_issues"]
    assert v.loc[100, "is_usable"]  # late data is still real data


def test_output_sorted_by_patient_and_time(records):
    other = make_records(n=50, patient="P0", session="P0_s")
    v = validate_sensor_data(pd.concat([records.iloc[::-1], other], ignore_index=True))
    assert list(v["patient_id"].unique()) == ["P0", "P1"]
    assert v.groupby("patient_id")["timestamp_utc"].apply(lambda s: s.is_monotonic_increasing).all()
