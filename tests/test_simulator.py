"""Simulator: each scenario contains the pattern it claims, output is reproducible,
and every injected fault in bad_quality is caught by the validator."""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.validate import validate_sensor_data
from src.simulator.generate import generate_all


@pytest.fixture(scope="module")
def sessions():
    frames, events, _ = generate_all(seed=42, n_patients=2)
    return {(f.patient_id.dropna().iloc[0], f.gt_scenario.iloc[0]): f for f in frames}


def clean_rows(df):
    return df[df.gt_injected.eq("")].assign(heart_rate=lambda d: pd.to_numeric(d.heart_rate, errors="coerce"))


def baseline(df):
    d = clean_rows(df)
    return d[(d.gt_state == "baseline") & (d.gt_sensor_status == "ok")]


def peak(df, state):
    rows = clean_rows(df)
    rows = rows[rows.gt_state == state]
    return rows[rows.gt_t_s >= rows.gt_t_s.min() + 60]  # let the 60 s RMSSD window catch up


@pytest.mark.parametrize("pid", ["SIM01", "SIM02"])
def test_arousal_is_hr_up_hrv_down_at_rest(sessions, pid):
    df = sessions[(pid, "hr_elevation")]
    b, p = baseline(df), peak(df, "arousal")
    assert p.heart_rate.mean() - b.heart_rate.mean() >= 18
    assert p.hrv_rmssd.mean() <= 0.8 * b.hrv_rmssd.mean()
    mag = np.sqrt(p.accel_x**2 + p.accel_y**2 + p.accel_z**2)
    assert mag.std() < 0.05
    assert p.gt_expected_alert.eq(1).all()


@pytest.mark.parametrize("pid", ["SIM01", "SIM02"])
def test_hrv_drop_with_small_hr_change(sessions, pid):
    df = sessions[(pid, "hrv_drop")]
    b, p = baseline(df), peak(df, "hrv_drop")
    assert p.hrv_rmssd.mean() <= 0.65 * b.hrv_rmssd.mean()
    assert p.heart_rate.mean() - b.heart_rate.mean() < 15
    assert p.gt_alert_level.eq("warning").all()


@pytest.mark.parametrize("pid", ["SIM01", "SIM02"])
def test_activity_is_a_confounder(sessions, pid):
    df = sessions[(pid, "high_activity")]
    b, p = baseline(df), peak(df, "activity")
    assert p.heart_rate.mean() - b.heart_rate.mean() >= 35
    mag = np.sqrt(p.accel_x**2 + p.accel_y**2 + p.accel_z**2)
    assert mag.std() >= 0.25
    assert df.gt_expected_alert.eq(0).all()


def test_missing_data_scenario(sessions):
    df = sessions[("SIM01", "missing_data")]
    assert len(df) == 3600 - 120  # disconnect: records absent
    off = df[df.gt_sensor_status == "off_wrist"]
    assert pd.to_numeric(off.heart_rate).isna().all() and off.skin_temp.iloc[-1] < 30
    art = df[df.gt_state == "artifact"]
    assert (pd.to_numeric(art.heart_rate) > 170).all()
    assert df.gt_expected_alert.eq(0).all()


def test_every_injected_fault_is_detected(sessions):
    raw = sessions[("SIM01", "bad_quality")]
    v = validate_sensor_data(raw)
    inj = v[v.gt_injected != ""]
    assert set(inj.gt_injected) == {"late", "duplicate", "invalid_timestamp", "missing_patient_id",
                                    "invalid_value", "naive_timezone"}
    for expected, issues in zip(inj.gt_injected, inj.quality_issues):
        assert expected in issues.split(";"), (expected, issues)
    # Untouched records stay good, except the HR warm-up at session start and the re-sent
    # copies, which legitimately arrive out of order.
    rest = v[(v.gt_injected == "") & v.quality_flag.ne("good")]
    assert set(rest.quality_flag) <= {"missing_value", "out_of_order"}
    resent_ts = set(v.loc[v.gt_injected == "duplicate", "timestamp_utc"])
    assert set(rest.loc[rest.quality_flag == "out_of_order", "timestamp_utc"]) <= resent_ts


def test_reproducible_and_seed_dependent():
    a, _, _ = generate_all(seed=42, n_patients=1)
    b, _, _ = generate_all(seed=42, n_patients=1)
    c, _, _ = generate_all(seed=7, n_patients=1)
    assert all(x.equals(y) for x, y in zip(a, b))
    assert not a[0].equals(c[0])


def test_patients_have_different_baselines(sessions):
    hrs = [baseline(sessions[(p, "normal")]).heart_rate.mean() for p in ("SIM01", "SIM02")]
    assert abs(hrs[0] - hrs[1]) > 1


def test_patient_baseline_is_stable_across_sessions(sessions):
    # Resting HR of one patient may drift a little day to day, not jump like a different person.
    for pid in ("SIM01", "SIM02"):
        hrs = [baseline(sessions[(pid, sc)]).heart_rate.median()
               for sc in ("normal", "hr_elevation", "hrv_drop", "missing_data", "bad_quality")]
        assert max(hrs) - min(hrs) < 7, (pid, hrs)
