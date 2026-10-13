"""Rule-based baseline: precedence, activity handling, persistence and output contract."""

import pandas as pd
import pytest

from src.ml.rules import LEVELS, RULES_CONFIG, RuleEngine, assess_window

T0 = pd.Timestamp("2026-01-05T08:10:00Z")
CONTRACT_KEYS = {"patient_id", "window_end", "risk_level", "risk_score", "reasons", "model_version", "data_quality"}


def window(i=0, hr=0.0, hrv=0.0, accel=0.01, quality="good", pid="P1", staleness=1.0):
    return dict(patient_id=pid, window_end=T0 + pd.Timedelta(minutes=i), hr_delta_pct=hr, hrv_delta_pct=hrv,
                accel_magnitude_std=accel, data_quality=quality, staleness_s=staleness)


def run(engine, windows):
    return [engine.predict(w)["risk_level"] for w in windows]


def test_output_follows_contract():
    r = RuleEngine().predict(window())
    assert set(r) == CONTRACT_KEYS
    assert r["risk_level"] in LEVELS and r["risk_score"] is None and r["model_version"] == "rules_v1"


def test_insufficient_data_always_wins():
    r = RuleEngine().predict(window(hr=80, quality="insufficient_data", staleness=200))
    assert r["risk_level"] == "insufficient_data"
    assert r["reasons"] == ["stale_data"]


def test_missing_baseline_is_insufficient_not_normal():
    r = RuleEngine().predict(window(hr=None, hrv=None))
    assert r["risk_level"] == "insufficient_data" and r["reasons"] == ["no_baseline"]


def test_hrv_optional_hr_alone_can_flag():
    kind, reasons = assess_window(window(hr=30, hrv=None))
    assert kind == "abnormal" and "hr_above_baseline" in reasons


def test_persistence_escalates_monitor_then_elevated():
    levels = run(RuleEngine(), [window(i, hr=30) for i in range(4)])
    assert levels == ["monitor", "monitor", "elevated_physiological_anomaly", "elevated_physiological_anomaly"]


def test_hrv_drop_alone_is_abnormal():
    levels = run(RuleEngine(), [window(i, hr=5, hrv=-50) for i in range(3)])
    assert levels[-1] == "elevated_physiological_anomaly"


def test_exercise_is_not_an_alert():
    levels = run(RuleEngine(), [window(i, hr=70, hrv=-60, accel=0.4) for i in range(5)])
    assert set(levels) == {"normal"}


def test_post_activity_cooldown_then_alerts_again():
    eng = RuleEngine()
    run(eng, [window(0, hr=70, accel=0.4)])
    cool = RULES_CONFIG["post_activity_cooldown_s"] // 60
    during = run(eng, [window(i, hr=30) for i in range(1, cool + 1)])
    after = run(eng, [window(i, hr=30) for i in range(cool + 1, cool + 4)])
    assert set(during) == {"normal"}
    assert after[-1] == "elevated_physiological_anomaly"


def test_gap_in_windows_resets_streak():
    eng = RuleEngine()
    run(eng, [window(0, hr=30), window(1, hr=30)])
    assert eng.predict(window(10, hr=30))["risk_level"] == "monitor"  # 9 minutes unseen: start over


def test_insufficient_window_resets_streak():
    eng = RuleEngine()
    levels = run(eng, [window(0, hr=30), window(1, hr=30), window(2, quality="insufficient_data"), window(3, hr=30)])
    assert levels == ["monitor", "monitor", "insufficient_data", "monitor"]


def test_patients_do_not_share_state():
    eng = RuleEngine()
    run(eng, [window(i, hr=30, pid="A") for i in range(3)])
    assert eng.predict(window(3, hr=30, pid="B"))["risk_level"] == "monitor"


@pytest.mark.parametrize("hr,hrv", [(5.0, -10.0), (-10.0, 20.0)])
def test_small_changes_are_normal(hr, hrv):
    assert run(RuleEngine(), [window(i, hr=hr, hrv=hrv) for i in range(5)]) == ["normal"] * 5
