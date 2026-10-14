"""Rule-based baseline: precedence, activity handling, persistence and output contract."""

import pandas as pd
import pytest

from src.ml.rules import LEVELS, RULES_CONFIG, RuleEngine, assess_window

T0 = pd.Timestamp("2026-01-05T08:10:00Z")
CONTRACT_KEYS = {"patient_id", "window_end", "risk_level", "reasons", "model_version", "data_quality"}


def window(i=0, hr=0.0, hrv=0.0, accel=0.01, quality="good", pid="P1", staleness=1.0):
    return dict(patient_id=pid, window_end=T0 + pd.Timedelta(minutes=i), hr_delta_pct=hr, hrv_delta_pct=hrv,
                accel_magnitude_std=accel, data_quality=quality, staleness_s=staleness)


def run(engine, windows):
    return [engine.predict(w)["risk_level"] for w in windows]


def test_output_follows_contract():
    r = RuleEngine().predict(window())
    assert set(r) == CONTRACT_KEYS
    assert r["risk_level"] in LEVELS and r["model_version"] == "rules_v1"
    assert "risk_score" not in r


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


# --- Sleeptime anomaly --------------------------------------------------------------------

from src.ml.rules import in_sleep_window  # noqa: E402

WIB = "Asia/Jakarta"


def at(local: str, minutes: int = 0, **kw):
    """A window whose window_end is `local` (Asia/Jakarta clock) + minutes, expressed in UTC."""
    end = (pd.Timestamp(local, tz=WIB) + pd.Timedelta(minutes=minutes)).tz_convert("UTC")
    w = window(**kw)
    w["window_end"] = end
    return w


def test_sleep_window_uses_patient_local_time():
    assert in_sleep_window(pd.Timestamp("2026-01-05T19:00Z"))       # 02:00 WIB
    assert not in_sleep_window(pd.Timestamp("2026-01-05T03:00Z"))   # 10:00 WIB
    assert in_sleep_window(pd.Timestamp("2026-01-05T15:00Z"))       # 22:00 WIB (start inclusive)
    assert not in_sleep_window(pd.Timestamp("2026-01-05T23:00Z"))   # 06:00 WIB (end exclusive)
    assert in_sleep_window(pd.Timestamp("2026-01-05T03:00Z"), timezone="America/New_York")  # 22:00 EST


def test_1_night_movement_is_sleep_disruption_not_activity():
    # Spec test 1: high movement at 02:00 -> elevated, never "activity".
    eng = RuleEngine()
    results = [eng.predict(at("2026-01-06 02:00", 5 * i, accel=0.4)) for i in range(2)]
    assert results[0]["risk_level"] == "monitor" and results[0]["reasons"] == ["sleep_disruption_candidate"]
    assert results[1]["risk_level"] == "elevated_physiological_anomaly"
    assert results[1]["reasons"] == ["sleep_disruption", "high_night_restlessness"]
    assert all("activity" not in r["reasons"] for r in results)


def test_2_night_hr_spike_monitor_then_elevated():
    # Spec test 2: HR spike at 03:00, still -> monitor on window 1, elevated on window 2.
    eng = RuleEngine()
    first = eng.predict(at("2026-01-06 03:00", 0, hr=25, accel=0.01))
    second = eng.predict(at("2026-01-06 03:00", 5, hr=25, accel=0.01))
    assert first["risk_level"] == "monitor" and first["reasons"] == ["sleep_disruption_candidate"]
    assert second["risk_level"] == "elevated_physiological_anomaly"
    assert second["reasons"] == ["sleep_disruption", "sustained_night_arousal"]


def test_3_day_movement_stays_activity():
    # Spec test 3: high movement at 10:00 -> normal (activity).
    eng = RuleEngine()
    results = [eng.predict(at("2026-01-06 10:00", 5 * i, accel=0.4, hr=60)) for i in range(3)]
    assert [r["risk_level"] for r in results] == ["normal"] * 3
    assert all(r["reasons"] == ["activity"] for r in results)


def test_brief_wake_on_sliding_windows_stays_monitor():
    # Guardrail: a ~4-minute wake-up is seen by ~8 overlapping 1-minute-step windows,
    # but covers < 10 minutes, so it must not reach elevated.
    eng = RuleEngine()
    levels = [eng.predict(at("2026-01-06 01:00", i, accel=0.2))["risk_level"] for i in range(8)]
    levels += [eng.predict(at("2026-01-06 01:00", i))["risk_level"] for i in range(8, 12)]
    assert "elevated_physiological_anomaly" not in levels
    assert levels[:8] == ["monitor"] * 8 and levels[8:] == ["normal"] * 4


def test_sustained_disruption_on_sliding_windows_reaches_elevated_after_10_min():
    eng = RuleEngine()
    levels = [eng.predict(at("2026-01-06 01:00", i, accel=0.2))["risk_level"] for i in range(12)]
    assert levels[:9] == ["monitor"] * 9 and levels[9] == "elevated_physiological_anomaly"


def test_calm_sleep_is_normal_even_with_low_hr():
    r = RuleEngine().predict(at("2026-01-06 02:00", hr=-10, accel=0.005))
    assert r["risk_level"] == "normal" and r["reasons"] == []


def test_night_insufficient_data_still_wins():
    r = RuleEngine().predict(at("2026-01-06 02:00", accel=0.4, quality="insufficient_data", staleness=200))
    assert r["risk_level"] == "insufficient_data"


def test_night_without_baseline_is_insufficient():
    r = RuleEngine().predict(at("2026-01-06 02:00", accel=0.4, hr=None))
    assert r["risk_level"] == "insufficient_data" and r["reasons"] == ["no_baseline"]


def test_sleep_reasons_are_not_diagnostic():
    allowed = {"sleep_disruption", "sleep_disruption_candidate", "high_night_restlessness", "sustained_night_arousal"}
    eng = RuleEngine()
    reasons = set()
    for i in range(3):
        reasons |= set(eng.predict(at("2026-01-06 02:00", 5 * i, accel=0.4))["reasons"])
    assert reasons <= allowed
    assert not any(w in r for r in reasons for w in ("ptsd", "diagnos", "episode"))
